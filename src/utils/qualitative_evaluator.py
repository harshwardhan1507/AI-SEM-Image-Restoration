"""Qualitative restoration failure analysis visualization module.

This module provides ``QualitativeEvaluator`` to create reproducible, publication-ready
visual comparison grids and crop zoom views for SEM image restoration evaluation.
It enforces fixed [0.0, 1.0] intensity mapping across all panels, preserves spatial aspect
ratios, and handles missing/unavailable baseline or improved model predictions gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger("QualitativeEvaluator")


class QualitativeEvaluator:
    """Evaluator for generating qualitative comparison grids and crop zoom views.

    Args:
        output_dir: Path to directory where output figure PNGs will be saved.
        dpi: Dots per inch resolution for rendered matplotlib figures (default 150).
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "results/images/qualitative_analysis",
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    @staticmethod
    def load_array(
        input_data: Union[str, Path, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Load and convert input image data to a 2D float32 NumPy array.

        Args:
            input_data: File path (.npy), NumPy array, or PyTorch Tensor.

        Returns:
            np.ndarray: 2D float32 array.

        Raises:
            TypeError: If input data type is unsupported.
            FileNotFoundError: If path string/Path does not exist on disk.
        """
        if isinstance(input_data, (str, Path)):
            path = Path(input_data).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image array file not found: {path}")
            arr = np.load(path)
        elif isinstance(input_data, torch.Tensor):
            arr = input_data.detach().cpu().numpy()
        elif isinstance(input_data, np.ndarray):
            arr = input_data.copy()
        else:
            raise TypeError(
                f"Unsupported input data type: {type(input_data)}. Expected Path, str, np.ndarray, or torch.Tensor."
            )

        arr = np.array(arr, dtype=np.float32)

        # Squeeze leading singleton channel/batch dimensions e.g. (1, 1, H, W) or (1, H, W) -> (H, W)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D image array after dimension squeezing, got shape {arr.shape}."
            )

        return arr

    @staticmethod
    def normalize_fixed(arr: np.ndarray) -> np.ndarray:
        return arr

    @staticmethod
    def align_spatial_dimensions(
        arr: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Align spatial resolution of an array to target (H, W) via bicubic interpolation.

        Used solely for visual grid alignment (e.g. upsampling 128x128 Raw input to 256x256 GT size)
        while preserving original aspect ratio and pixel intensity range.

        Args:
            arr: Input 2D image array.
            target_shape: Target (height, width) tuple.

        Returns:
            np.ndarray: Resized 2D array of shape target_shape.
        """
        if arr.shape == target_shape:
            return arr

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        resized_tensor = torch.nn.functional.interpolate(
            tensor,
            size=target_shape,
            mode="bicubic",
            align_corners=False,
        )
        return resized_tensor.squeeze(0).squeeze(0).numpy()

    def render_comparison_grid(
        self,
        raw_input: Union[str, Path, np.ndarray, torch.Tensor],
        gt_target: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        baseline_pred: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        improved_pred: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        bicubic_ref: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        sample_id: str = "sample",
        save_name: Optional[str] = None,
        baseline_status_msg: str = "exp001 Baseline\n(Unavailable)",
        improved_status_msg: str = "Pending Issue #38\ntraining results",
    ) -> Path:
        """Render a publication-ready multi-column comparison figure.

        Columns:
            1. Raw Noisy Input
            2. Ground Truth (or "Ground Truth N/A" if None)
            3. Baseline NAFNet Output (or baseline_status_msg if None)
            4. Improved Model Output (or improved_status_msg if None)
            5. [Optional] Bicubic Reference (if bicubic_ref is provided)

        Args:
            raw_input: Raw noisy input image array/path.
            gt_target: Ground truth target array/path (optional).
            baseline_pred: Verified exp001 baseline prediction array/path (optional).
            improved_pred: Verified Issue #38 improved prediction array/path (optional).
            bicubic_ref: Optional separate bicubic reference array/path.
            sample_id: Sample identifier string for title/metadata.
            save_name: Optional custom filename for output PNG.
            baseline_status_msg: Display text when baseline_pred is None.
            improved_status_msg: Display text when improved_pred is None.

        Returns:
            Path: Absolute path to rendered figure PNG.
        """
        raw_arr = self.normalize_fixed(self.load_array(raw_input))
        target_shape = (256, 256)

        # Align Raw input to GT size for visual comparison if needed
        raw_aligned = self.align_spatial_dimensions(raw_arr, target_shape)

        panels: List[Dict[str, Any]] = [
            {
                "title": f"Raw Noisy Input\n({raw_arr.shape[0]}x{raw_arr.shape[1]})",
                "data": raw_aligned,
                "available": True,
            }
        ]

        # Ground Truth Panel
        if gt_target is not None:
            gt_arr = self.normalize_fixed(self.load_array(gt_target))
            target_shape = gt_arr.shape
            panels.append(
                {
                    "title": f"Ground Truth\n({gt_arr.shape[0]}x{gt_arr.shape[1]})",
                    "data": gt_arr,
                    "available": True,
                }
            )
        else:
            panels.append(
                {
                    "title": "Ground Truth\n(N/A - Unpaired Test)",
                    "data": np.zeros(target_shape, dtype=np.float32),
                    "available": False,
                    "msg": "Ground Truth N/A\n(Unpaired Test Sample)",
                }
            )

        # Baseline NAFNet Panel
        if baseline_pred is not None:
            base_arr = self.normalize_fixed(self.load_array(baseline_pred))
            base_aligned = self.align_spatial_dimensions(base_arr, target_shape)
            panels.append(
                {
                    "title": f"Baseline NAFNet\n({base_arr.shape[0]}x{base_arr.shape[1]})",
                    "data": base_aligned,
                    "available": True,
                }
            )
        else:
            panels.append(
                {
                    "title": "Baseline NAFNet",
                    "data": np.zeros(target_shape, dtype=np.float32),
                    "available": False,
                    "msg": baseline_status_msg,
                }
            )

        # Improved Model Panel
        if improved_pred is not None:
            imp_arr = self.normalize_fixed(self.load_array(improved_pred))
            imp_aligned = self.align_spatial_dimensions(imp_arr, target_shape)
            panels.append(
                {
                    "title": f"Improved Model\n({imp_arr.shape[0]}x{imp_arr.shape[1]})",
                    "data": imp_aligned,
                    "available": True,
                }
            )
        else:
            panels.append(
                {
                    "title": "Improved Model",
                    "data": np.zeros(target_shape, dtype=np.float32),
                    "available": False,
                    "msg": improved_status_msg,
                }
            )

        # Optional Bicubic Reference Panel
        if bicubic_ref is not None:
            bic_arr = self.normalize_fixed(self.load_array(bicubic_ref))
            bic_aligned = self.align_spatial_dimensions(bic_arr, target_shape)
            panels.append(
                {
                    "title": f"Bicubic Reference\n({bic_arr.shape[0]}x{bic_arr.shape[1]})",
                    "data": bic_aligned,
                    "available": True,
                }
            )

        num_cols = len(panels)
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4.5), squeeze=False)
        axes_1d = axes[0]

        fig.suptitle(
            f"Qualitative Restoration Comparison: {sample_id} [Display Normalization: Fixed 0.0 - 1.0]",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        vmin, vmax = (gt_arr.min(), gt_arr.max()) if "gt_arr" in locals() else (0.0, 1.0)
        for col_idx, panel in enumerate(panels):
            ax = axes_1d[col_idx]
            ax.set_title(panel["title"], fontsize=11, pad=8)

            if panel["available"]:
                ax.imshow(panel["data"], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            else:
                ax.imshow(panel["data"], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", alpha=0.3)
                ax.text(
                    0.5,
                    0.5,
                    panel["msg"],
                    color="darkred" if "Unavailable" in panel["msg"] else "navy",
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"),
                )

            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filename = save_name if save_name else f"{sample_id}_comparison_grid.png"
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved qualitative comparison grid: {out_path}")
        return out_path

    def render_zoom_crop(
        self,
        raw_input: Union[str, Path, np.ndarray, torch.Tensor],
        crop_bbox: Tuple[int, int, int, int],
        gt_target: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        baseline_pred: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        improved_pred: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        bicubic_ref: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        sample_id: str = "sample",
        save_name: Optional[str] = None,
    ) -> Path:
        """Render zoomed crop insets for inspecting fine structures, edges, and artifacts.

        Args:
            raw_input: Raw noisy input array/path.
            crop_bbox: Crop bounding box (ymin, xmin, ymax, xmax) in 256x256 target coordinates.
            gt_target: Ground truth target array/path.
            baseline_pred: Baseline prediction array/path.
            improved_pred: Improved prediction array/path.
            bicubic_ref: Optional bicubic reference array/path.
            sample_id: Sample identifier string.
            save_name: Output filename string.

        Returns:
            Path: Path to saved zoom crop figure PNG.
        """
        ymin, xmin, ymax, xmax = crop_bbox
        target_shape = (256, 256)

        raw_arr = self.normalize_fixed(self.load_array(raw_input))
        raw_aligned = self.align_spatial_dimensions(raw_arr, target_shape)
        raw_crop = raw_aligned[ymin:ymax, xmin:xmax]

        panels: List[Dict[str, Any]] = [
            {
                "title": f"Raw Noisy Crop\n[{ymin}:{ymax}, {xmin}:{xmax}]",
                "data": raw_crop,
                "available": True,
            }
        ]

        if gt_target is not None:
            gt_arr = self.normalize_fixed(self.load_array(gt_target))
            gt_aligned = self.align_spatial_dimensions(gt_arr, target_shape)
            panels.append(
                {
                    "title": "Ground Truth Crop",
                    "data": gt_aligned[ymin:ymax, xmin:xmax],
                    "available": True,
                }
            )
        else:
            panels.append(
                {
                    "title": "Ground Truth Crop\n(N/A)",
                    "data": np.zeros((ymax - ymin, xmax - xmin), dtype=np.float32),
                    "available": False,
                    "msg": "Ground Truth N/A",
                }
            )

        if baseline_pred is not None:
            base_arr = self.normalize_fixed(self.load_array(baseline_pred))
            base_aligned = self.align_spatial_dimensions(base_arr, target_shape)
            panels.append(
                {
                    "title": "Baseline NAFNet Crop",
                    "data": base_aligned[ymin:ymax, xmin:xmax],
                    "available": True,
                }
            )
        else:
            panels.append(
                {
                    "title": "Baseline NAFNet Crop",
                    "data": np.zeros((ymax - ymin, xmax - xmin), dtype=np.float32),
                    "available": False,
                    "msg": "exp001 Baseline\n(Unavailable)",
                }
            )

        if improved_pred is not None:
            imp_arr = self.normalize_fixed(self.load_array(improved_pred))
            imp_aligned = self.align_spatial_dimensions(imp_arr, target_shape)
            panels.append(
                {
                    "title": "Improved Model Crop",
                    "data": imp_aligned[ymin:ymax, xmin:xmax],
                    "available": True,
                }
            )
        else:
            panels.append(
                {
                    "title": "Improved Model Crop",
                    "data": np.zeros((ymax - ymin, xmax - xmin), dtype=np.float32),
                    "available": False,
                    "msg": "Pending Issue #38\ntraining results",
                }
            )

        if bicubic_ref is not None:
            bic_arr = self.normalize_fixed(self.load_array(bicubic_ref))
            bic_aligned = self.align_spatial_dimensions(bic_arr, target_shape)
            panels.append(
                {
                    "title": "Bicubic Ref Crop",
                    "data": bic_aligned[ymin:ymax, xmin:xmax],
                    "available": True,
                }
            )

        num_cols = len(panels)
        fig, axes = plt.subplots(1, num_cols, figsize=(3.8 * num_cols, 4.2), squeeze=False)
        axes_1d = axes[0]

        fig.suptitle(
            f"Qualitative Crop Zoom Detail ({ymin}:{ymax}, {xmin}:{xmax}): {sample_id} [Fixed 0.0 - 1.0]",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        vmin, vmax = (gt_arr.min(), gt_arr.max()) if "gt_arr" in locals() else (0.0, 1.0)
        for col_idx, panel in enumerate(panels):
            ax = axes_1d[col_idx]
            ax.set_title(panel["title"], fontsize=10, pad=6)

            if panel["available"]:
                ax.imshow(panel["data"], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            else:
                ax.imshow(panel["data"], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", alpha=0.3)
                ax.text(
                    0.5,
                    0.5,
                    panel["msg"],
                    color="darkred" if "Unavailable" in panel["msg"] else "navy",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
                )

            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filename = save_name if save_name else f"{sample_id}_zoom_crop.png"
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved qualitative zoom crop grid: {out_path}")
        return out_path
