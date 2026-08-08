"""Unit tests for Trainer class (Issue #19).

Tests verify training loop execution, optimizer integration, scheduler stepping,
gradient clipping, AMP mixed-precision, TensorBoard logging, memory management,
and CheckpointManager integration on CPU using synthetic data.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.engine.checkpoint import CheckpointManager
from src.engine.trainer import Trainer

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model: single Conv2d that preserves spatial dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SyntheticSEMDataset(Dataset):
    """Tiny synthetic dataset mimicking SEM batch structure.

    Returns dict with ``"input"``, ``"target"``, ``"filename"`` keys.
    Input and target are same spatial size for simplicity in unit tests
    (Trainer does not enforce the 2x SR spatial relationship).
    """

    def __init__(self, num_samples: int = 8, size: int = 32) -> None:
        self.num_samples = num_samples
        self.size = size
        # Fixed seed data for reproducibility
        gen = torch.Generator().manual_seed(42)
        self.inputs = [
            torch.rand(1, size, size, generator=gen) for _ in range(num_samples)
        ]
        self.targets = [
            torch.rand(1, size, size, generator=gen) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input": self.inputs[index],
            "target": self.targets[index],
            "filename": f"sample_{index:03d}",
        }


def _collate_fn(batch: list) -> Dict[str, Any]:
    """Minimal collate matching sem_collate batch dict structure."""
    return {
        "input": torch.stack([s["input"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
        "filename": [s["filename"] for s in batch],
    }


def _make_trainer(
    val_loader: Optional[DataLoader] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    writer: Optional[SummaryWriter] = None,
    grad_clip_norm: Optional[float] = None,
    use_amp: bool = False,
    amp_dtype: str = "float16",
    epochs: int = 2,
) -> Trainer:
    """Helper to construct a Trainer with synthetic components."""
    model = TinyModel()
    dataset = SyntheticSEMDataset(num_samples=8, size=32)
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=_collate_fn
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    return Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        val_loader=val_loader,
        checkpoint_manager=checkpoint_manager,
        writer=writer,
        device="cpu",
        epochs=epochs,
        grad_clip_norm=grad_clip_norm,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        val_freq=1,
        log_freq=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainerConstruction:
    """Test 1: Trainer can be instantiated with correct attributes."""

    def test_trainer_construction(self) -> None:
        trainer = _make_trainer()

        assert isinstance(trainer.model, nn.Module)
        assert isinstance(trainer.optimizer, optim.AdamW)
        assert trainer.scheduler is not None
        assert trainer.device == torch.device("cpu")
        assert trainer.epochs == 2
        assert trainer.grad_clip_norm is None
        assert trainer.use_amp is False
        assert trainer.global_step == 0


class TestSingleTrainingEpoch:
    """Test 2: Single training epoch completes with finite loss and parameter updates."""

    def test_single_training_epoch(self) -> None:
        trainer = _make_trainer()

        # Capture initial parameter values
        initial_params = {
            name: param.clone().detach()
            for name, param in trainer.model.named_parameters()
        }

        avg_loss = trainer.train_epoch(epoch=1)

        # Loss must be finite
        assert isinstance(avg_loss, float)
        assert torch.isfinite(torch.tensor(avg_loss)), f"Loss is not finite: {avg_loss}"

        # Parameters must have changed after optimizer update
        params_changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.equal(param.data, initial_params[name]):
                params_changed = True
                break

        assert params_changed, "Model parameters did not change after training epoch"


class TestOptimizerIntegration:
    """Test 3: Verify optimizer is AdamW."""

    def test_optimizer_is_adamw(self) -> None:
        trainer = _make_trainer()
        assert isinstance(trainer.optimizer, optim.AdamW)


class TestSchedulerLRUpdate:
    """Test 4: CosineAnnealingLR decreases learning rate across epochs."""

    def test_scheduler_lr_update(self) -> None:
        trainer = _make_trainer(epochs=10)

        assert trainer.scheduler is not None
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Run a few epochs and step scheduler
        for epoch in range(1, 6):
            trainer.train_epoch(epoch)
            trainer.scheduler.step()

        mid_lr = trainer.optimizer.param_groups[0]["lr"]

        # After several epochs of cosine annealing, LR should have decreased
        assert (
            mid_lr < initial_lr
        ), f"LR did not decrease: initial={initial_lr}, mid={mid_lr}"


class TestGradientClipping:
    """Test 5: Gradient clipping is applied when grad_clip_norm is set."""

    def test_gradient_clipping(self) -> None:
        max_norm = 0.1
        trainer = _make_trainer(grad_clip_norm=max_norm)

        # Run one epoch with clipping
        avg_loss = trainer.train_epoch(epoch=1)
        assert torch.isfinite(torch.tensor(avg_loss))

        # Verify clipping config is stored
        assert trainer.grad_clip_norm == max_norm

    def test_gradient_clipping_bounds_norms(self) -> None:
        """Verify that after clipping, gradient norms are bounded."""
        max_norm = 0.01
        trainer = _make_trainer(grad_clip_norm=max_norm)

        # Do a manual forward-backward to check norms
        batch = next(iter(trainer.train_loader))
        input_tensor = batch["input"].to(trainer.device)
        target = batch["target"].to(trainer.device)

        trainer.optimizer.zero_grad(set_to_none=True)
        output = trainer.model(input_tensor)
        loss = trainer.criterion(output, target)

        scaled_loss = trainer.scaler.scale(loss)
        if isinstance(scaled_loss, torch.Tensor):
            scaled_loss.backward()
        else:
            for sl in scaled_loss:
                sl.backward()
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm)

        # Check that total grad norm is bounded
        total_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), float("inf")
        )
        assert (
            total_norm <= max_norm * 1.1
        ), f"Gradient norm {total_norm} exceeds max_norm {max_norm}"


class TestAMPDisabled:
    """Test 6: Standard FP32 training on CPU works correctly."""

    def test_amp_disabled_fp32(self) -> None:
        trainer = _make_trainer(use_amp=False)

        assert trainer.use_amp is False
        assert trainer.scaler.is_enabled() is False

        avg_loss = trainer.train_epoch(epoch=1)
        assert torch.isfinite(torch.tensor(avg_loss))


class TestAMPFP16CUDA:
    """Test 7: FP16 autocast + GradScaler on CUDA."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for FP16 AMP test"
    )
    def test_amp_fp16_cuda(self) -> None:
        model = TinyModel()
        dataset = SyntheticSEMDataset(num_samples=4, size=32)
        train_loader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=_collate_fn
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cuda",
            epochs=1,
            use_amp=True,
            amp_dtype="float16",
        )

        assert trainer.use_amp is True
        assert trainer.scaler.is_enabled() is True

        avg_loss = trainer.train_epoch(epoch=1)
        assert torch.isfinite(torch.tensor(avg_loss))


class TestAMPBF16:
    """Test 8: BF16 autocast without GradScaler."""

    def test_amp_bf16_no_scaler(self) -> None:
        """BF16 on CPU: autocast may be supported, GradScaler must be disabled."""
        trainer = _make_trainer(use_amp=True, amp_dtype="bfloat16")

        # BF16 should never enable GradScaler
        assert trainer.scaler.is_enabled() is False
        assert trainer.amp_dtype_torch == torch.bfloat16

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for BF16 test"
    )
    def test_amp_bf16_cuda_no_scaler(self) -> None:
        """BF16 on CUDA: GradScaler must be disabled."""
        model = TinyModel()
        dataset = SyntheticSEMDataset(num_samples=4, size=32)
        train_loader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=_collate_fn
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cuda",
            epochs=1,
            use_amp=True,
            amp_dtype="bfloat16",
        )

        assert trainer.use_amp is True
        assert trainer.scaler.is_enabled() is False


class TestAMPFP16CPUAutoDisable:
    """Test: CPU + FP16 AMP request is auto-disabled."""

    def test_cpu_fp16_amp_disabled(self) -> None:
        trainer = _make_trainer(use_amp=True, amp_dtype="float16")

        # CPU + FP16 should be auto-disabled
        assert trainer.use_amp is False


class TestTensorBoardLogging:
    """Test 9: SummaryWriter logs scalars to temporary directory."""

    def test_tensorboard_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(log_dir=tmpdir)
            val_dataset = SyntheticSEMDataset(num_samples=4, size=32)
            val_loader = DataLoader(
                val_dataset, batch_size=2, shuffle=False, collate_fn=_collate_fn
            )

            trainer = _make_trainer(val_loader=val_loader, writer=writer, epochs=2)
            result = trainer.fit(start_epoch=1)

            # Writer should have been closed by fit()
            # Verify TensorBoard event files were created
            tb_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
            assert len(tb_files) > 0, "No TensorBoard event files created"

            # Verify training completed
            assert result["epochs_completed"] == 2


class TestNoGraphRetention:
    """Test 10: Training accumulates only float scalars, not autograd tensors."""

    def test_no_graph_retention(self) -> None:
        trainer = _make_trainer(epochs=3)

        # Run multiple epochs
        losses = []
        for epoch in range(1, 4):
            avg_loss = trainer.train_epoch(epoch)
            losses.append(avg_loss)

        # All accumulated losses must be plain Python floats
        for loss_val in losses:
            assert isinstance(
                loss_val, float
            ), f"Loss is {type(loss_val)}, expected float"
            # Must not be a tensor with grad
            assert not isinstance(loss_val, torch.Tensor)


class TestCheckpointIntegration:
    """Test 11: CheckpointManager.save() is invoked during fit()."""

    def test_checkpoint_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(checkpoint_dir=tmpdir)
            val_dataset = SyntheticSEMDataset(num_samples=4, size=32)
            val_loader = DataLoader(
                val_dataset, batch_size=2, shuffle=False, collate_fn=_collate_fn
            )

            trainer = _make_trainer(
                val_loader=val_loader,
                checkpoint_manager=ckpt_mgr,
                epochs=2,
            )

            result = trainer.fit(start_epoch=1)

            # Checkpoint files should exist after fit()
            ckpt_dir = Path(tmpdir)
            epoch_files = list(ckpt_dir.glob("checkpoint_epoch_*.pth"))
            assert len(epoch_files) > 0, "No checkpoint files created"

            # Best model should exist (first epoch always becomes best)
            best_path = ckpt_dir / "best_model.pth"
            assert best_path.exists(), "best_model.pth was not created"

            # Verify checkpoint content structure
            checkpoint = torch.load(best_path, weights_only=False)
            assert set(checkpoint.keys()) == {
                "epoch",
                "model",
                "optimizer",
                "scheduler",
                "best_metric",
            }

            assert result["epochs_completed"] == 2


class TestValidation:
    """Test validation method produces expected metric keys."""

    def test_validate_returns_metrics(self) -> None:
        val_dataset = SyntheticSEMDataset(num_samples=4, size=32)
        val_loader = DataLoader(
            val_dataset, batch_size=2, shuffle=False, collate_fn=_collate_fn
        )
        trainer = _make_trainer(val_loader=val_loader)

        metrics = trainer.validate(epoch=1)

        assert "val_loss" in metrics
        assert "val_psnr" in metrics
        assert "val_ssim" in metrics

        assert isinstance(metrics["val_loss"], float)
        assert isinstance(metrics["val_psnr"], float)
        assert isinstance(metrics["val_ssim"], float)

        assert torch.isfinite(torch.tensor(metrics["val_loss"]))

    def test_validate_without_val_loader(self) -> None:
        trainer = _make_trainer(val_loader=None)
        metrics = trainer.validate(epoch=1)

        assert metrics == {"val_loss": 0.0, "val_psnr": 0.0, "val_ssim": 0.0}


class TestFit:
    """Test full training loop via fit()."""

    def test_fit_completes(self) -> None:
        val_dataset = SyntheticSEMDataset(num_samples=4, size=32)
        val_loader = DataLoader(
            val_dataset, batch_size=2, shuffle=False, collate_fn=_collate_fn
        )
        trainer = _make_trainer(val_loader=val_loader, epochs=3)

        result = trainer.fit(start_epoch=1)

        assert result["epochs_completed"] == 3
        assert isinstance(result["final_train_loss"], float)
        assert isinstance(result["best_val_psnr"], float)
        assert isinstance(result["history"], list)
        assert len(result["history"]) == 3
