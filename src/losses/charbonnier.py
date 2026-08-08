"""Charbonnier loss function module for SEM image restoration.

This module provides ``CharbonnierLoss``, a differentiable smooth L1 variant
loss function using PyTorch autograd.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (differentiable L1 variant).

    Formula:
        L(P, T) = reduction( sqrt( (P - T)^2 + eps^2 ) )

    Args:
        eps: Small constant for numerical stability under square root. Default 1e-3.
        reduction: Specifies the reduction to apply to output: 'mean' | 'sum' | 'none'.
            Default 'mean'.
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be a positive float, got {eps}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Expected 'mean', 'sum', or 'none'."
            )

        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss between prediction and target tensors.

        Args:
            prediction: Predicted tensor of shape (..., H, W).
            target: Ground truth target tensor of same shape as prediction.

        Returns:
            torch.Tensor: Reduced scalar loss or elementwise loss tensor if reduction='none'.

        Raises:
            ValueError: If prediction and target shapes do not match.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch between prediction {prediction.shape} and target {target.shape}."
            )

        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
