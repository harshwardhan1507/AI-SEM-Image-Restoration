"""Persistent model checkpoint management module for SEM image restoration.

This module provides ``CheckpointManager``, handling model state, optimizer state,
learning rate scheduler state, epoch numbers, and best validation metric tracking.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Manager for persistent model checkpoint saving, loading, and best-model tracking.

    Required Checkpoint Dictionary Keys:
        - 'epoch': Current training epoch integer.
        - 'model': Model state dictionary (model.state_dict()).
        - 'optimizer': Optimizer state dictionary (optimizer.state_dict()).
        - 'scheduler': Scheduler state dictionary or None.
        - 'best_metric': Peak validation metric (PSNR) score observed.

    Args:
        checkpoint_dir: Directory path where checkpoints will be saved.
    """

    REQUIRED_KEYS = {"epoch", "model", "optimizer", "scheduler", "best_metric"}

    def __init__(self, checkpoint_dir: Union[str, Path]) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float = float("-inf")

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        metric: Optional[float] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> Path:
        """Save a training checkpoint and optionally update best_model.pth.

        Args:
            epoch: Current training epoch number.
            model: PyTorch model module.
            optimizer: PyTorch optimizer instance.
            scheduler: Optional learning rate scheduler instance.
            metric: Optional current validation metric (PSNR) value.
            is_best: Explicit flag forcing best_model.pth update.
            filename: Custom filename for periodic checkpoint. Defaults to 'checkpoint_epoch_{epoch:03d}.pth'.

        Returns:
            Path: Path to saved periodic checkpoint.
        """
        new_best = False
        if metric is not None and metric > self.best_metric:
            self.best_metric = float(metric)
            new_best = True

        if is_best:
            new_best = True

        state: Dict[str, Any] = {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": float(self.best_metric),
        }

        save_filename = (
            filename if filename is not None else f"checkpoint_epoch_{epoch:03d}.pth"
        )
        periodic_path = self.checkpoint_dir / save_filename
        torch.save(state, periodic_path)

        if new_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)

        return periodic_path

    def save_best(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        metric: float = 0.0,
    ) -> bool:
        """Convenience method to save checkpoint if validation metric improves best_metric.

        Args:
            epoch: Current epoch number.
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            scheduler: Optional LR scheduler.
            metric: Current validation metric value (PSNR).

        Returns:
            bool: True if new best metric achieved and best_model.pth saved, False otherwise.
        """
        if metric > self.best_metric:
            self.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metric=metric,
                is_best=True,
            )
            return True
        return False

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Load checkpoint state dictionary and restore model/optimizer/scheduler parameters.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            model: Optional PyTorch model instance to restore weights into.
            optimizer: Optional PyTorch optimizer instance to restore state into.
            scheduler: Optional LR scheduler instance to restore state into.
            map_location: Device mapping target for torch.load (default 'cpu').

        Returns:
            Dict[str, Any]: Full loaded checkpoint dictionary.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If loaded object is not a dictionary.
            KeyError: If required key is missing in checkpoint dictionary.
        """
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Invalid checkpoint format in {path}. Expected dictionary."
            )

        missing_keys = self.REQUIRED_KEYS - set(checkpoint.keys())
        if missing_keys:
            raise KeyError(
                f"Checkpoint at {path} is missing required key(s): {sorted(missing_keys)}."
            )

        if model is not None:
            model.load_state_dict(checkpoint["model"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        loaded_best = float(checkpoint["best_metric"])
        if loaded_best > self.best_metric:
            self.best_metric = loaded_best

        return checkpoint

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Return path to best_model.pth if it exists, else None."""
        best_path = self.checkpoint_dir / "best_model.pth"
        return best_path if best_path.exists() else None

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Return path to highest-epoch periodic checkpoint file if any exists, else None."""
        epoch_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not epoch_files:
            return None
        return max(epoch_files, key=lambda p: p.stat().st_mtime)
