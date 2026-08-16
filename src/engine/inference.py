"""Sliding-window inference engine module for full-resolution SEM image restoration.

This module provides ``SlidingWindowInference`` and ``slide_window_inference``, enabling
tile-based sliding-window inference with overlapping patches, 2D Gaussian spatial blending,
and memory-bounded mini-batch execution.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _generate_gaussian_weights(
    tile_h: int,
    tile_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    min_weight: float = 1e-3,
) -> torch.Tensor:
    """Generate a deterministic 2D Gaussian spatial weighting map for tile blending.

    The weighting map has maximum magnitude at the tile center and tapers off towards
    the tile boundaries, preventing visible spatial seams when accumulating overlapping
    tile predictions.

    Args:
        tile_h: Output tile height in pixels.
        tile_w: Output tile width in pixels.
        device: Target PyTorch execution device.
        dtype: Tensor data type (default torch.float32).
        min_weight: Minimum weight floor to ensure numerical stability at corners.

    Returns:
        Tensor of shape ``(1, 1, tile_h, tile_w)``.
    """
    center_y = (tile_h - 1) / 2.0
    center_x = (tile_w - 1) / 2.0

    sigma_y = max(1.0, tile_h / 4.0)
    sigma_x = max(1.0, tile_w / 4.0)

    y_coords = torch.arange(tile_h, dtype=dtype, device=device)
    x_coords = torch.arange(tile_w, dtype=dtype, device=device)

    gauss_y = torch.exp(-0.5 * ((y_coords - center_y) / sigma_y) ** 2)
    gauss_x = torch.exp(-0.5 * ((x_coords - center_x) / sigma_x) ** 2)

    weight_2d = torch.outer(gauss_y, gauss_x)
    # Apply minimum floor so corners maintain positive non-zero weight
    weight_2d = torch.clamp(weight_2d, min=min_weight)

    return weight_2d.unsqueeze(0).unsqueeze(0)


def _calculate_tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    """Calculate 1D tile start coordinates guaranteeing 100% boundary coverage.

    Args:
        length: Total dimension length (padded).
        tile_size: Tile size along dimension.
        stride: Step stride between adjacent tiles.

    Returns:
        List of integer start coordinates.
    """
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


class SlidingWindowInference:
    """Sliding-window tile inference engine with Gaussian spatial blending.

    Args:
        model: PyTorch neural network model (e.g. NAFNet nn.Module).
        tile_size: Input spatial tile size in pixels (default 512).
        overlap: Fractional overlap ratio between adjacent tiles, 0.0 <= overlap < 1.0 (default 0.25).
        tile_batch_size: Maximum number of tiles to process in a single model forward pass (default 1).
        device: Target execution device. If None, uses model's current device.

    Raises:
        ValueError: If parameters fail validation checks.
        TypeError: If model is not an instance of nn.Module.
    """

    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 512,
        overlap: float = 0.25,
        tile_batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be a torch.nn.Module instance, got {type(model).__name__}."
            )
        if tile_size <= 0:
            raise ValueError(f"tile_size must be a positive integer, got {tile_size}.")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(
                f"overlap must satisfy 0.0 <= overlap < 1.0, got {overlap}."
            )
        if tile_batch_size <= 0:
            raise ValueError(
                f"tile_batch_size must be a positive integer, got {tile_batch_size}."
            )

        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.tile_batch_size = tile_batch_size

        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        # Derive model upscale factor from existing architecture contract
        self.upscale = getattr(model, "upscale", 1)

        # Calculate stride from tile_size and overlap
        self.stride = max(1, math.floor(tile_size * (1.0 - overlap)))

    def infer(
        self,
        x: torch.Tensor,
        use_gaussian: bool = True,
    ) -> torch.Tensor:
        """Execute sliding-window inference on input image tensor.

        Args:
            x: Input image tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
            use_gaussian: If True, applies 2D Gaussian spatial blending map across tile overlaps.
                If False, uses uniform weighting.

        Returns:
            Restored output tensor of shape ``(C, H * upscale, W * upscale)`` or ``(B, C, H * upscale, W * upscale)``.

        Raises:
            ValueError: If input tensor dimensionality or channel layout is invalid,
                or if model output spatial dimensions mismatch expected tile sizes.
        """
        if x.dim() not in (3, 4):
            raise ValueError(
                f"Input tensor must be 3D (C, H, W) or 4D (B, C, H, W), got {x.dim()}D tensor."
            )

        is_3d = x.dim() == 3
        if is_3d:
            inp = x.unsqueeze(0)
        else:
            inp = x

        B, C, H, W = inp.shape

        # 1. Safe padding if image is smaller than configured tile_size
        pad_h = max(0, self.tile_size - H)
        pad_w = max(0, self.tile_size - W)

        if pad_h > 0 or pad_w > 0:
            # Reflect padding requires input dimensions to be strictly greater than padding amount
            pad_mode = (
                "reflect"
                if (H > pad_h and W > pad_w and H >= 2 and W >= 2)
                else "constant"
            )
            padded_inp = F.pad(inp, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            padded_inp = inp


        _, _, H_pad, W_pad = padded_inp.shape

        # 2. Output space dimension calculation
        H_out = H_pad * self.upscale
        W_out = W_pad * self.upscale
        tile_out_h = self.tile_size * self.upscale
        tile_out_w = self.tile_size * self.upscale

        # 3. Calculate 1D grid tile starts
        starts_y = _calculate_tile_starts(H_pad, self.tile_size, self.stride)
        starts_x = _calculate_tile_starts(W_pad, self.tile_size, self.stride)

        # 4. Construct blending weight map at output resolution
        exec_device = self.device
        if use_gaussian:
            weight_map = _generate_gaussian_weights(
                tile_h=tile_out_h,
                tile_w=tile_out_w,
                device=exec_device,
                dtype=torch.float32,
            )
        else:
            weight_map = torch.ones(
                (1, 1, tile_out_h, tile_out_w),
                dtype=torch.float32,
                device=exec_device,
            )

        # 5. Move model to execution device and set evaluation mode
        self.model.eval()
        self.model.to(exec_device)

        batch_outputs: List[torch.Tensor] = []

        # Process each image in batch B
        with torch.inference_mode():
            for b in range(B):
                single_img = padded_inp[b : b + 1].to(exec_device)

                weighted_accum = torch.zeros(
                    (1, C, H_out, W_out), dtype=torch.float32, device=exec_device
                )
                weight_sum_accum = torch.zeros(
                    (1, 1, H_out, W_out), dtype=torch.float32, device=exec_device
                )

                # Build tile coordinate list
                tile_coords: List[Tuple[int, int, int, int]] = []
                for y in starts_y:
                    for x_coord in starts_x:
                        y_out = y * self.upscale
                        x_out = x_coord * self.upscale
                        tile_coords.append((y, x_coord, y_out, x_out))

                # Process tiles in mini-batches
                for i in range(0, len(tile_coords), self.tile_batch_size):
                    batch_coords = tile_coords[i : i + self.tile_batch_size]

                    tile_tensors = [
                        single_img[:, :, y : y + self.tile_size, x : x + self.tile_size]
                        for y, x, _, _ in batch_coords
                    ]
                    tiles_batch = torch.cat(tile_tensors, dim=0)

                    # Pad tiles_batch to multiple of 8 for NAFNet U-Net requirements
                    _, _, t_h, t_w = tiles_batch.shape
                    pad_h = (8 - t_h % 8) % 8
                    pad_w = (8 - t_w % 8) % 8
                    
                    if pad_h > 0 or pad_w > 0:
                        padded_tiles = F.pad(tiles_batch, (0, pad_w, 0, pad_h), mode="reflect")
                        pred_batch_padded = self.model(padded_tiles)
                        
                        out_h = t_h * self.upscale
                        out_w = t_w * self.upscale
                        pred_batch = pred_batch_padded[:, :, :out_h, :out_w]
                    else:
                        pred_batch = self.model(tiles_batch)

                    # Validate model output spatial dimensions against expected contract
                    if pred_batch.shape[2:] != (tile_out_h, tile_out_w):
                        raise ValueError(
                            f"Model output tile spatial dimensions {tuple(pred_batch.shape[2:])} "
                            f"do not match expected tile dimensions ({tile_out_h}, {tile_out_w})."
                        )

                    weighted_preds = pred_batch * weight_map

                    for idx, (_, _, y_out, x_out) in enumerate(batch_coords):
                        weighted_accum[
                            :, :, y_out : y_out + tile_out_h, x_out : x_out + tile_out_w
                        ] += weighted_preds[idx : idx + 1]
                        weight_sum_accum[
                            :, :, y_out : y_out + tile_out_h, x_out : x_out + tile_out_w
                        ] += weight_map

                # Normalize accumulated predictions by total weight sum
                norm_output = weighted_accum / torch.clamp(
                    weight_sum_accum, min=1e-8
                )

                # Unpad output back to exact original target spatial dimensions
                final_h = H * self.upscale
                final_w = W * self.upscale
                unpadded_output = norm_output[:, :, :final_h, :final_w]

                batch_outputs.append(unpadded_output)

        result = torch.cat(batch_outputs, dim=0)

        if is_3d:
            return result.squeeze(0)
        return result


def slide_window_inference(
    model: nn.Module,
    x: torch.Tensor,
    tile_size: int = 512,
    overlap: float = 0.25,
    tile_batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    use_gaussian: bool = True,
    force_tile: bool = False,
) -> torch.Tensor:
    """Helper function to execute inference, defaulting to a single full-pass to avoid tiling artifacts.

    If the image is extremely large and OOMs are expected, set `force_tile=True` to 
    fall back to the sliding-window tiled inference engine.

    Args:
        model: PyTorch model.
        x: Input image tensor (C, H, W) or (B, C, H, W).
        tile_size: Tile size in pixels (default 512).
        overlap: Tile overlap ratio (default 0.25).
        tile_batch_size: Tile mini-batch size (default 1).
        device: Target device.
        use_gaussian: Whether to apply 2D Gaussian spatial blending (if tiling).
        force_tile: Force sliding window inference.

    Returns:
        Restored output tensor of shape (C, H*scale, W*scale) or (B, C, H*scale, W*scale).
    """
    is_3d = x.dim() == 3
    inp = x.unsqueeze(0) if is_3d else x
    _, _, h, w = inp.shape

    # Default to single padded forward pass to avoid border hallucination artifacts
    if not force_tile:
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            pad_mode = (
                "reflect"
                if (h > pad_h and w > pad_w and h >= 2 and w >= 2)
                else "constant"
            )
            padded_inp = F.pad(inp, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            padded_inp = inp

        if device is not None:
            exec_device = torch.device(device)
        else:
            try:
                exec_device = next(model.parameters()).device
            except StopIteration:
                exec_device = torch.device("cpu")

        model.eval()
        model.to(exec_device)
        padded_inp = padded_inp.to(exec_device)
        
        with torch.inference_mode():
            out = model(padded_inp)
            
        upscale = getattr(model, "upscale", 1)
        final_h = h * upscale
        final_w = w * upscale
        out = out[:, :, :final_h, :final_w]
        
        return out.squeeze(0) if is_3d else out

    # Fallback to sliding window for massive OOM-prone images
    engine = SlidingWindowInference(
        model=model,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        device=device,
    )
    return engine.infer(x, use_gaussian=use_gaussian)
