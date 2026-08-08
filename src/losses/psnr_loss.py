"""Differentiable PSNR loss function module for SEM image restoration.

This module provides ``PSNRLoss``, a PyTorch autograd loss function
derived from Peak Signal-to-Noise Ratio (PSNR).
"""

import math

import torch
import torch.nn as nn


class PSNRLoss(nn.Module):
    """Differentiable PSNR Loss.

    Computes per-sample MSE across non-batch dimensions, converts to stabilized
    negative PSNR, and applies reduction:
        MSE_b = mean((prediction_b - target_b)^2)
        L_b = (10 / ln(10)) * ln((MSE_b + eps) / data_range^2)
        L = reduction(L_b)

    Args:
        data_range: Maximum pixel intensity dynamic range. Default 1.0.
        eps: Small constant for zero-MSE numerical stabilization. Default 1e-8.
        reduction: Specifies the reduction to apply to output: 'mean' | 'sum' | 'none'.
            Default 'mean'.
    """

    def __init__(
        self, data_range: float = 1.0, eps: float = 1e-8, reduction: str = "mean"
    ) -> None:
        super().__init__()
        if data_range <= 0:
            raise ValueError(f"data_range must be a positive float, got {data_range}")
        if eps <= 0:
            raise ValueError(f"eps must be a positive float, got {eps}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Expected 'mean', 'sum', or 'none'."
            )

        self.data_range = float(data_range)
        self.eps = float(eps)
        self.reduction = reduction
        self._log10_factor = 10.0 / math.log(10.0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute PSNR loss between prediction and target tensors.

        Args:
            prediction: Predicted tensor of shape (B, C, H, W), (1, H, W), or (H, W).
            target: Ground truth target tensor of same shape as prediction.

        Returns:
            torch.Tensor: Reduced scalar loss, or per-sample loss tensor of shape (B,) if reduction='none'.

        Raises:
            ValueError: If prediction and target shapes do not match.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch between prediction {prediction.shape} and target {target.shape}."
            )

        diff_sq = (prediction - target) ** 2

        if diff_sq.ndim >= 3:
            non_batch_dims = tuple(range(1, diff_sq.ndim))
            mse = torch.mean(diff_sq, dim=non_batch_dims)
        else:
            mse = torch.mean(diff_sq)

        scale = self.data_range**2
        loss_b = self._log10_factor * torch.log((mse + self.eps) / scale)

        if self.reduction == "mean":
            return torch.mean(loss_b)
        elif self.reduction == "sum":
            return torch.sum(loss_b)
        else:
            return loss_b
