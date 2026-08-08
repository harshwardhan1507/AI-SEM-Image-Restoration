"""Pytest unit test suite for CheckpointManager (src/engine/checkpoint.py)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.engine import CheckpointManager


class DummyModel(nn.Module):
    """Simple model for checkpoint testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def test_save_structure(tmp_path: Path) -> None:
    """Test 1: Verify saved checkpoint contains exact required keys."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    ckpt_path = mgr.save(
        epoch=1, model=model, optimizer=optimizer, scheduler=scheduler, metric=30.5
    )
    assert ckpt_path.exists()

    raw_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(raw_data, dict)
    assert set(raw_data.keys()) == {
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "best_metric",
    }
    assert raw_data["epoch"] == 1
    assert raw_data["best_metric"] == pytest.approx(30.5)


def test_model_restoration(tmp_path: Path) -> None:
    """Test 2: Save model state, mutate model, and load into fresh model, verifying original parameters restored."""
    mgr = CheckpointManager(tmp_path)
    model1 = DummyModel()
    optimizer1 = optim.AdamW(model1.parameters(), lr=1e-3)

    orig_params = [p.clone() for p in model1.parameters()]

    # Save original state
    ckpt_path = mgr.save(epoch=5, model=model1, optimizer=optimizer1)

    # Mutate parameters of model1
    with torch.no_grad():
        for p in model1.parameters():
            p.add_(1.0)

    # Load checkpoint into model1 (restore)
    mgr.load(ckpt_path, model=model1)
    for p, orig in zip(model1.parameters(), orig_params, strict=True):
        assert torch.equal(p, orig)

    # Create fresh model2 and load checkpoint
    model2 = DummyModel()
    mgr.load(ckpt_path, model=model2)
    for p2, orig in zip(model2.parameters(), orig_params, strict=True):
        assert torch.equal(p2, orig)


def test_optimizer_restoration(tmp_path: Path) -> None:
    """Test 3: Take training step to populate optimizer state, save, load into fresh optimizer."""
    mgr = CheckpointManager(tmp_path)
    model1 = DummyModel()
    optimizer1 = optim.AdamW(model1.parameters(), lr=1e-3)

    # Perform a step to build Adam momentum/variance state
    x = torch.rand(2, 1, 8, 8)
    loss = model1(x).sum()
    loss.backward()
    optimizer1.step()

    ckpt_path = mgr.save(epoch=1, model=model1, optimizer=optimizer1)

    # Create fresh model and optimizer
    model2 = DummyModel()
    model2.load_state_dict(model1.state_dict())
    optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3)

    assert len(optimizer2.state) == 0  # no state before load

    mgr.load(ckpt_path, optimizer=optimizer2)
    assert len(optimizer2.state) > 0  # state restored


def test_scheduler_restoration(tmp_path: Path) -> None:
    """Test 4: Advance scheduler, save, load into fresh scheduler (and test None scheduler)."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    optimizer.step()
    scheduler1.step()  # lr becomes 1e-3
    assert scheduler1.get_last_lr()[0] == pytest.approx(1e-3)

    ckpt_path = mgr.save(
        epoch=2, model=model, optimizer=optimizer, scheduler=scheduler1
    )

    # Restore into fresh scheduler
    optimizer2 = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.1)

    mgr.load(ckpt_path, scheduler=scheduler2)
    assert scheduler2.get_last_lr()[0] == pytest.approx(1e-3)

    # Test saving and loading when scheduler is None
    ckpt_path_no_sched = mgr.save(
        epoch=3, model=model, optimizer=optimizer, scheduler=None
    )
    raw = torch.load(ckpt_path_no_sched, map_location="cpu", weights_only=False)
    assert raw["scheduler"] is None

    # Restoring scheduler=None should not crash
    mgr.load(ckpt_path_no_sched, scheduler=scheduler2)


def test_epoch_restoration(tmp_path: Path) -> None:
    """Test 5: Save at known epoch, load, verify epoch recovered."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = mgr.save(epoch=42, model=model, optimizer=optimizer)
    loaded = mgr.load(ckpt_path)

    assert loaded["epoch"] == 42


def test_best_metric_restoration(tmp_path: Path) -> None:
    """Test 6: Save checkpoint with known best PSNR, load, verify metric recovered."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = mgr.save(epoch=10, model=model, optimizer=optimizer, metric=34.2)
    loaded = mgr.load(ckpt_path)

    assert loaded["best_metric"] == pytest.approx(34.2)
    assert mgr.best_metric == pytest.approx(34.2)


def test_best_model_replacement(tmp_path: Path) -> None:
    """Test 7: Save with PSNR A, save with higher PSNR B, verify best_model.pth stores B."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    mgr.save(epoch=1, model=model, optimizer=optimizer, metric=25.0)
    best_path = mgr.get_best_checkpoint_path()
    assert best_path is not None
    assert torch.load(best_path, weights_only=False)["best_metric"] == pytest.approx(
        25.0
    )

    mgr.save(epoch=2, model=model, optimizer=optimizer, metric=28.5)
    assert torch.load(best_path, weights_only=False)["best_metric"] == pytest.approx(
        28.5
    )
    assert torch.load(best_path, weights_only=False)["epoch"] == 2


def test_lower_psnr_does_not_replace_best(tmp_path: Path) -> None:
    """Test 8: Save with PSNR A, attempt save with lower PSNR B, verify best remains A."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    mgr.save(epoch=1, model=model, optimizer=optimizer, metric=30.0)
    best_path = mgr.get_best_checkpoint_path()

    # Attempt lower metric
    mgr.save(epoch=2, model=model, optimizer=optimizer, metric=27.0)

    best_data = torch.load(best_path, weights_only=False)
    assert best_data["best_metric"] == pytest.approx(30.0)
    assert best_data["epoch"] == 1


def test_cpu_loading(tmp_path: Path) -> None:
    """Test 9: Save checkpoint, load with map_location='cpu', verify successful restoration."""
    mgr = CheckpointManager(tmp_path)
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = mgr.save(epoch=1, model=model, optimizer=optimizer, metric=20.0)

    loaded = mgr.load(ckpt_path, map_location="cpu")
    assert isinstance(loaded, dict)
    assert loaded["epoch"] == 1


def test_invalid_checkpoint(tmp_path: Path) -> None:
    """Test 10: Verify loading non-existent file or corrupted dictionary raises exception."""
    mgr = CheckpointManager(tmp_path)

    # File missing
    with pytest.raises(FileNotFoundError):
        mgr.load(tmp_path / "non_existent.pth")

    # Not a dictionary
    not_dict_path = tmp_path / "not_dict.pth"
    torch.save(["not", "a", "dict"], not_dict_path)
    with pytest.raises(ValueError, match="Expected dictionary"):
        mgr.load(not_dict_path)

    # Missing required keys
    incomplete_path = tmp_path / "incomplete.pth"
    torch.save({"epoch": 1, "model": {}}, incomplete_path)
    with pytest.raises(KeyError, match="missing required key"):
        mgr.load(incomplete_path)
