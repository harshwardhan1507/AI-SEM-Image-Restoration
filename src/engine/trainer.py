"""PyTorch Trainer module for SEM image restoration training execution.

This module provides ``Trainer``, the central training orchestration class
that executes epoch/batch training loops with AdamW optimizer, CosineAnnealingLR
scheduler, PyTorch AMP mixed-precision, gradient clipping, TensorBoard logging,
validation metric computation, and CheckpointManager integration.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.engine.checkpoint import CheckpointManager
from src.metrics.psnr_ssim import calculate_psnr, calculate_ssim


class Trainer:
    """Training execution engine for SEM image restoration with NAFNet.

    Orchestrates training epoch loops, AMP mixed-precision scaling, gradient
    clipping, validation, TensorBoard logging, and checkpoint persistence.

    The Trainer receives pre-constructed dependencies (model, optimizer,
    scheduler, loss, data loaders) and orchestrates their interaction.
    It does not construct these components internally.

    Args:
        model: PyTorch model (e.g., NAFNet nn.Module).
        train_loader: Training DataLoader yielding batch dicts with
            ``"input"`` and ``"target"`` tensor keys.
        criterion: Loss function module (e.g., CharbonnierLoss, PSNRLoss).
        optimizer: Pre-constructed PyTorch optimizer (e.g., AdamW).
        scheduler: Optional pre-constructed LR scheduler. Stepped once
            per completed training epoch.
        val_loader: Optional validation DataLoader.
        checkpoint_manager: Optional CheckpointManager for periodic and
            best-model checkpointing. CheckpointManager owns best-model
            selection logic.
        writer: Optional TensorBoard SummaryWriter.
        device: Target device string or torch.device (default ``"cpu"``).
        epochs: Total number of training epochs. API fallback default only;
            callers should provide the actual experiment value.
        grad_clip_norm: Maximum gradient norm for clipping. ``None`` disables
            gradient clipping.
        use_amp: Whether to enable AMP mixed-precision training.
        amp_dtype: AMP precision type: ``"float16"`` or ``"bfloat16"``.
            GradScaler is only enabled for float16 on CUDA.
            CPU + float16 AMP is automatically disabled.
        val_freq: Run validation every ``val_freq`` epochs.
        log_freq: Log batch-level training loss every ``log_freq`` batches.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        val_loader: Optional[DataLoader] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        writer: Optional[SummaryWriter] = None,
        device: Union[str, torch.device] = "cpu",
        epochs: int = 100,
        grad_clip_norm: Optional[float] = None,
        use_amp: bool = False,
        amp_dtype: str = "float16",
        val_freq: int = 1,
        log_freq: int = 10,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.val_loader = val_loader
        self.checkpoint_manager = checkpoint_manager
        self.writer = writer
        self.device = torch.device(device)
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.val_freq = val_freq
        self.log_freq = log_freq

        # Determine device type for AMP autocast
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"

        # Resolve AMP dtype
        self.amp_dtype_torch = (
            torch.float16 if amp_dtype == "float16" else torch.bfloat16
        )

        # Determine effective AMP enablement:
        # CPU + float16 is not a valid AMP training mode — disable AMP
        if (
            use_amp
            and self.device_type == "cpu"
            and self.amp_dtype_torch == torch.float16
        ):
            self.use_amp = False
        else:
            self.use_amp = use_amp

        # GradScaler: only for float16 on CUDA
        scaler_enabled = (
            self.use_amp
            and self.device_type == "cuda"
            and self.amp_dtype_torch == torch.float16
        )
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=scaler_enabled)

        # Move model to device
        self.model = self.model.to(self.device)

        # Global step counter for batch-level TensorBoard logging
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """Execute a single training epoch.

        Args:
            epoch: Current epoch number (1-indexed, for logging purposes).

        Returns:
            float: Average training loss for the epoch (sample-weighted).
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_tensor = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            batch_size = input_tensor.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            # Autocast context created per-iteration
            with torch.amp.autocast(
                device_type=self.device_type,
                dtype=self.amp_dtype_torch,
                enabled=self.use_amp,
            ):
                output = self.model(input_tensor)
                loss = self.criterion(output, target)
            scaled_loss = self.scaler.scale(loss)
            if isinstance(scaled_loss, torch.Tensor):
                scaled_loss.backward()
            else:
                for sl in scaled_loss:
                    sl.backward()
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate detached scalar only — no graph retention
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1

            # Batch-level TensorBoard logging
            if self.writer is not None and (batch_idx + 1) % self.log_freq == 0:
                self.writer.add_scalar("Train/BatchLoss", loss.item(), self.global_step)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """Execute validation over the validation DataLoader.

        Computes validation loss, PSNR, and SSIM using the existing
        metric implementations from ``src.metrics.psnr_ssim``.

        Args:
            epoch: Current epoch number (for logging purposes).

        Returns:
            Dict with keys ``"val_loss"``, ``"val_psnr"``, ``"val_ssim"``.
        """
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_psnr": 0.0, "val_ssim": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_tensor = batch["input"].to(self.device)
                target = batch["target"].to(self.device)
                batch_size = input_tensor.size(0)

                with torch.amp.autocast(
                    device_type=self.device_type,
                    dtype=self.amp_dtype_torch,
                    enabled=self.use_amp,
                ):
                    output = self.model(input_tensor)
                    loss = self.criterion(output, target)

                total_loss += loss.item() * batch_size

                # Metrics — calculate_psnr and calculate_ssim handle
                # detaching and per-sample averaging internally
                psnr_val = calculate_psnr(output, target, data_range=1.0)
                ssim_val = calculate_ssim(output, target, data_range=1.0)

                total_psnr += psnr_val * batch_size
                total_ssim += ssim_val * batch_size
                total_samples += batch_size

        # Restore training mode
        self.model.train()

        if total_samples == 0:
            return {"val_loss": 0.0, "val_psnr": 0.0, "val_ssim": 0.0}

        return {
            "val_loss": total_loss / total_samples,
            "val_psnr": total_psnr / total_samples,
            "val_ssim": total_ssim / total_samples,
        }

    def fit(self, start_epoch: int = 1) -> Dict[str, Any]:
        """Execute the full training loop across all epochs.

        For each epoch:
            1. Run training epoch.
            2. Step scheduler (epoch-level).
            3. If validation epoch: run validation, log metrics, save checkpoint.

        Args:
            start_epoch: Starting epoch number (1-indexed). Used for resume.

        Returns:
            Dict containing training summary with keys:
                ``"epochs_completed"``, ``"final_train_loss"``,
                ``"best_val_psnr"``, ``"history"``.
        """
        history: List[Dict[str, Any]] = []
        final_train_loss = 0.0
        best_val_psnr = float("-inf")

        for epoch in range(start_epoch, self.epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            final_train_loss = train_loss

            # Epoch-level scheduler stepping
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            # Validation and checkpointing on val_freq
            if self.val_loader is not None and epoch % self.val_freq == 0:
                val_metrics = self.validate(epoch)
                epoch_record.update(val_metrics)

                if val_metrics["val_psnr"] > best_val_psnr:
                    best_val_psnr = val_metrics["val_psnr"]

                # TensorBoard epoch-level logging
                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss", train_loss, epoch)
                    self.writer.add_scalar("Val/Loss", val_metrics["val_loss"], epoch)
                    self.writer.add_scalar("Val/PSNR", val_metrics["val_psnr"], epoch)
                    self.writer.add_scalar("Val/SSIM", val_metrics["val_ssim"], epoch)
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("Train/LR", lr, epoch)

                # Checkpoint — CheckpointManager owns best-model logic
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save(
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metric=val_metrics["val_psnr"],
                    )

            history.append(epoch_record)

        if self.writer is not None:
            self.writer.close()

        return {
            "epochs_completed": self.epochs - start_epoch + 1,
            "final_train_loss": final_train_loss,
            "best_val_psnr": best_val_psnr,
            "history": history,
        }
