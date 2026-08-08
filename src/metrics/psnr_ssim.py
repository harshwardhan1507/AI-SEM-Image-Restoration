"""Evaluation metrics module for full-reference SEM image restoration (PSNR & SSIM).

This module provides reference-verified implementation of PSNR (Peak Signal-to-Noise Ratio)
and SSIM (Structural Similarity Index Measure) supporting NumPy arrays and PyTorch tensors.
"""

from typing import Tuple, Union

import numpy as np
import skimage.metrics
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _prepare_input(array: ArrayLike) -> np.ndarray:
    """Validate, detach, convert precision, and convert ArrayLike input to a float32 NumPy array.

    Args:
        array: Input array as np.ndarray or torch.Tensor.

    Returns:
        np.ndarray: float32 NumPy array representation.

    Raises:
        TypeError: If array is neither np.ndarray nor torch.Tensor.
        ValueError: If array contains NaN or Inf values.
    """
    if isinstance(array, torch.Tensor):
        arr_np = array.detach().cpu().to(torch.float32).numpy()
    elif isinstance(array, np.ndarray):
        arr_np = array.astype(np.float32, copy=False)
    else:
        raise TypeError(
            f"Unsupported array type '{type(array).__name__}'. Expected numpy.ndarray or torch.Tensor."
        )

    if not np.isfinite(arr_np).all():
        raise ValueError("Input image contains non-finite values (NaN or Inf).")

    return arr_np


def _validate_inputs(
    prediction: ArrayLike, target: ArrayLike, data_range: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate data range, shapes, and dimensions for metric calculation.

    Args:
        prediction: Predicted image array or tensor.
        target: Target image array or tensor.
        data_range: Dynamic range of input images (must be > 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Prepared (prediction, target) float32 NumPy arrays.

    Raises:
        ValueError: If data_range <= 0, shapes mismatch, or array dimensions are unsupported.
    """
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}.")

    pred_np = _prepare_input(prediction)
    target_np = _prepare_input(target)

    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"Shape mismatch between prediction {pred_np.shape} and target {target_np.shape}."
        )

    ndim = pred_np.ndim
    if ndim not in (2, 3, 4):
        raise ValueError(
            f"Unsupported image dimensions: {ndim}. Expected 2D (H, W), 3D (1, H, W), or 4D (B, 1, H, W)."
        )

    if ndim == 3 and pred_np.shape[0] != 1:
        raise ValueError(
            f"Expected single-channel 3D image shape (1, H, W), got {pred_np.shape}."
        )

    if ndim == 4 and pred_np.shape[1] != 1:
        raise ValueError(
            f"Expected single-channel 4D batched image shape (B, 1, H, W), got {pred_np.shape}."
        )

    return pred_np, target_np


def calculate_psnr(
    prediction: ArrayLike,
    target: ArrayLike,
    data_range: float = 1.0,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between prediction and target.

    For batched inputs (B, 1, H, W), metrics are calculated per sample and averaged.

    Args:
        prediction: Predicted image array or tensor.
        target: Ground truth target image array or tensor.
        data_range: Dynamic range of image pixel values (default 1.0 for [0, 1]).

    Returns:
        float: Scalar PSNR value in decibels (dB), or float("inf") if images are identical.
    """
    pred_np, target_np = _validate_inputs(prediction, target, data_range)

    if pred_np.ndim == 4:
        batch_size = pred_np.shape[0]
        psnr_vals = [
            calculate_psnr(pred_np[i], target_np[i], data_range=data_range)
            for i in range(batch_size)
        ]
        return float(np.mean(psnr_vals))

    if np.array_equal(pred_np, target_np):
        return float("inf")

    psnr_val = skimage.metrics.peak_signal_noise_ratio(
        target_np, pred_np, data_range=data_range
    )
    return float(psnr_val)


def calculate_ssim(
    prediction: ArrayLike,
    target: ArrayLike,
    data_range: float = 1.0,
) -> float:
    """Calculate Structural Similarity Index (SSIM) between prediction and target.

    For batched inputs (B, 1, H, W), metrics are calculated per sample and averaged.

    Args:
        prediction: Predicted image array or tensor.
        target: Ground truth target image array or tensor.
        data_range: Dynamic range of image pixel values (default 1.0 for [0, 1]).

    Returns:
        float: Scalar SSIM value in [-1.0, 1.0].
    """
    pred_np, target_np = _validate_inputs(prediction, target, data_range)

    if pred_np.ndim == 4:
        batch_size = pred_np.shape[0]
        ssim_vals = [
            calculate_ssim(pred_np[i], target_np[i], data_range=data_range)
            for i in range(batch_size)
        ]
        return float(np.mean(ssim_vals))

    pred_2d = pred_np[0] if pred_np.ndim == 3 else pred_np
    target_2d = target_np[0] if target_np.ndim == 3 else target_np

    ssim_val = skimage.metrics.structural_similarity(
        target_2d, pred_2d, data_range=data_range
    )
    return float(ssim_val)
