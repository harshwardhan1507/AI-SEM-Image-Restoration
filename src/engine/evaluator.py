"""Evaluation execution engine module for SEM image restoration.

This module provides ``Evaluator``, which computes full-dataset mean PSNR
and SSIM scores, generates 4-panel visual comparison grids (Input vs Prediction vs
Target vs Absolute Error Map), logs figures to TensorBoard, and saves prediction
visualizations to disk.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.metrics.psnr_ssim import calculate_psnr, calculate_ssim


class Evaluator:
    """Evaluation engine for SEM image restoration models.

    Computes dataset-wide PSNR and SSIM metrics over paired test/validation splits,
    generates 4-panel visual comparison grids, logs figures to TensorBoard if a writer
    is supplied, and saves prediction visualization files to disk.

    Args:
        model: PyTorch model (e.g., NAFNet nn.Module).
        data_loader: DataLoader yielding sample batch dictionaries with ``"input"``,
            optional ``"target"``, and ``"filename"`` keys.
        device: Target device string or torch.device (default ``"cpu"``).
        writer: Optional TensorBoard SummaryWriter instance owned by the caller.
        output_dir: Directory path where visualization PNG files will be saved.
            Defaults to ``"outputs/predictions"``.
        max_visualizations: Maximum number of sample visualization figures to generate
            per evaluation run. Defaults to 10.

    Raises:
        TypeError: If model is not an instance of nn.Module.
        ValueError: If data_loader is None or max_visualizations < 0.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: Union[str, torch.device] = "cpu",
        writer: Optional[SummaryWriter] = None,
        output_dir: Union[str, Path] = "outputs/predictions",
        max_visualizations: int = 10,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be a torch.nn.Module, got {type(model).__name__}."
            )
        if data_loader is None:
            raise ValueError("data_loader cannot be None.")
        if max_visualizations < 0:
            raise ValueError(
                f"max_visualizations must be non-negative, got {max_visualizations}."
            )

        self.model = model
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.writer = writer
        self.output_dir = Path(output_dir).resolve()
        self.max_visualizations = max_visualizations

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to configured device
        self.model = self.model.to(self.device)

    @staticmethod
    def _sanitize_filename(filename: Any, fallback_idx: int) -> str:
        """Sanitize metadata filename string to prevent path traversal issues.

        Args:
            filename: Sample filename metadata string or primitive.
            fallback_idx: Fallback integer index if filename is empty/invalid.

        Returns:
            str: Safe stem string for output filenames.
        """
        if not filename or not isinstance(filename, str):
            return f"sample_{fallback_idx:04d}"

        # Extract path stem / basename to eliminate directory traversal characters
        safe_name = Path(filename).name
        stem = Path(safe_name).stem
        if not stem:
            return f"sample_{fallback_idx:04d}"

        # Remove any remaining unsafe path characters
        cleaned = (
            stem.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("..", "_")
        )
        return cleaned if cleaned else f"sample_{fallback_idx:04d}"

    def _create_comparison_figure(
        self,
        input_tensor: torch.Tensor,
        prediction_tensor: torch.Tensor,
        target_tensor: Optional[torch.Tensor],
        sample_id: str,
        psnr_val: Optional[float] = None,
        ssim_val: Optional[float] = None,
    ) -> plt.Figure:
        """Generate a 4-panel visual comparison figure.

        Panels:
            1. Input (LR display upsampled)
            2. Prediction (Restored HR)
            3. Target (Ground Truth HR or N/A)
            4. Absolute Error Map (|Pred - Target| or N/A)

        The input LR tensor is resized ONLY for Matplotlib display formatting.
        Native prediction and target tensors used for metric evaluation are never mutated.

        Args:
            input_tensor: LR input tensor of shape (1, H_lr, W_lr) on CPU.
            prediction_tensor: Restored HR tensor of shape (1, H_hr, W_hr) on CPU.
            target_tensor: Ground truth HR tensor of shape (1, H_hr, W_hr) on CPU or None.
            sample_id: Sample identifier string for plot titles.
            psnr_val: Calculated sample PSNR or None.
            ssim_val: Calculated sample SSIM or None.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        h_hr, w_hr = prediction_tensor.shape[1], prediction_tensor.shape[2]

        # Upsample LR input tensor ONLY for display copy
        input_4d = input_tensor.unsqueeze(0)  # (1, 1, H_lr, W_lr)
        input_upsampled = F.interpolate(
            input_4d, size=(h_hr, w_hr), mode="bicubic", align_corners=False
        ).squeeze(
            0
        )  # (1, H_hr, W_hr)

        input_np = input_upsampled[0].numpy()
        pred_np = prediction_tensor[0].numpy()

        has_target = target_tensor is not None
        target_np = target_tensor[0].numpy() if has_target else None

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"Sample: {sample_id}"
            + (
                f" | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
                if psnr_val is not None and ssim_val is not None
                else ""
            ),
            fontsize=12,
            fontweight="bold",
        )

        # Panel 1: Input (LR Upsampled)
        axes[0].imshow(input_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("Input (LR 2x Display)")
        axes[0].axis("off")

        # Panel 2: Prediction (Restored HR)
        axes[1].imshow(pred_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Prediction (NAFNet HR)")
        axes[1].axis("off")

        # Panel 3: Target (Ground Truth HR or N/A)
        if has_target and target_np is not None:
            axes[2].imshow(target_np, cmap="gray", vmin=0.0, vmax=1.0)
            axes[2].set_title("Target (Ground Truth HR)")
        else:
            axes[2].text(
                0.5,
                0.5,
                "Target N/A",
                ha="center",
                va="center",
                fontsize=14,
                color="red",
            )
            axes[2].set_title("Target")
        axes[2].axis("off")

        # Panel 4: Absolute Error Map
        if has_target and target_np is not None:
            error_np = np.abs(pred_np - target_np)
            im3 = axes[3].imshow(error_np, cmap="magma")
            axes[3].set_title("Absolute Error (|Pred - Target|)")
            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        else:
            axes[3].text(
                0.5,
                0.5,
                "Error Map N/A",
                ha="center",
                va="center",
                fontsize=14,
                color="red",
            )
            axes[3].set_title("Absolute Error Map")
        axes[3].axis("off")

        fig.tight_layout()
        return fig

    def evaluate(
        self,
        epoch: Optional[int] = None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Execute model evaluation over the full dataset split.

        Computes mean PSNR and SSIM metrics, generates prediction figures up to
        ``max_visualizations``, saves PNG files to ``output_dir``, and logs figures to
        TensorBoard if a writer was supplied.

        Args:
            epoch: Optional current training epoch integer for TensorBoard logging.
            save_visualizations: If True, generates and saves figure files/plots.

        Returns:
            Dict containing evaluation metrics and execution metadata:
                - ``"mean_psnr"``: Dataset-wide mean PSNR float (or 0.0 if targetless).
                - ``"mean_ssim"``: Dataset-wide mean SSIM float (or 0.0 if targetless).
                - ``"num_samples"``: Total evaluated samples integer.
                - ``"num_visualizations"``: Total visualization files generated.
                - ``"output_dir"``: Absolute output path string.
        """
        # Preserve original training mode
        original_training_state = self.model.training
        self.model.eval()

        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0
        vis_count = 0
        global_idx = 0

        with torch.no_grad():
            for batch in self.data_loader:
                input_tensor = batch["input"].to(self.device)
                target_raw = batch.get("target", None)
                target_tensor = (
                    target_raw.to(self.device) if target_raw is not None else None
                )
                filenames = batch.get("filename", [])
                batch_size = input_tensor.size(0)

                # Forward inference
                prediction = self.model(input_tensor)

                # Metrics computation if targets are available
                has_target = target_tensor is not None
                if has_target and target_tensor is not None:
                    batch_psnr = calculate_psnr(
                        prediction, target_tensor, data_range=1.0
                    )
                    batch_ssim = calculate_ssim(
                        prediction, target_tensor, data_range=1.0
                    )

                    total_psnr += batch_psnr * batch_size
                    total_ssim += batch_ssim * batch_size

                total_samples += batch_size

                # Visualizations generation
                if save_visualizations and vis_count < self.max_visualizations:
                    for i in range(batch_size):
                        if vis_count >= self.max_visualizations:
                            break

                        sample_filename = (
                            filenames[i]
                            if (isinstance(filenames, list) and i < len(filenames))
                            else None
                        )
                        safe_stem = self._sanitize_filename(
                            sample_filename, global_idx + i
                        )

                        # Detach and move to CPU for visualization plotting
                        inp_cpu = input_tensor[i].detach().cpu()
                        pred_cpu = prediction[i].detach().cpu()
                        tgt_cpu = (
                            target_tensor[i].detach().cpu()
                            if (has_target and target_tensor is not None)
                            else None
                        )

                        sample_psnr = (
                            calculate_psnr(pred_cpu, tgt_cpu, data_range=1.0)
                            if tgt_cpu is not None
                            else None
                        )
                        sample_ssim = (
                            calculate_ssim(pred_cpu, tgt_cpu, data_range=1.0)
                            if tgt_cpu is not None
                            else None
                        )

                        fig = self._create_comparison_figure(
                            input_tensor=inp_cpu,
                            prediction_tensor=pred_cpu,
                            target_tensor=tgt_cpu,
                            sample_id=safe_stem,
                            psnr_val=sample_psnr,
                            ssim_val=sample_ssim,
                        )

                        # Save figure to disk
                        save_path = self.output_dir / f"{safe_stem}_comparison.png"
                        fig.savefig(save_path, dpi=150, bbox_inches="tight")

                        # Log figure to TensorBoard if writer is provided (caller owns writer)
                        if self.writer is not None:
                            self.writer.add_figure(
                                tag=f"Evaluation/Comparison/{safe_stem}",
                                figure=fig,
                                global_step=epoch if epoch is not None else 0,
                            )

                        # Memory safety: close figure immediately
                        plt.close(fig)
                        vis_count += 1

                global_idx += batch_size

        # Restore original model training mode
        self.model.train(original_training_state)

        mean_psnr = (
            total_psnr / total_samples if total_samples > 0 and has_target else 0.0
        )
        mean_ssim = (
            total_ssim / total_samples if total_samples > 0 and has_target else 0.0
        )

        return {
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "num_samples": total_samples,
            "num_visualizations": vis_count,
            "output_dir": str(self.output_dir),
        }
