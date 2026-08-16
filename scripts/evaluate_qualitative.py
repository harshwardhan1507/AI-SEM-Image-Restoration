"""CLI script for running qualitative restoration failure analysis on SEM micrographs.

This script loads SEM dataset samples, audits baseline and improved prediction/checkpoint
availability, generates publication-ready 4-column visual comparison grids and crop zoom
views with fixed [0.0, 1.0] intensity mapping, and produces qualitative analysis artifacts.

Example:
    $ python scripts/evaluate_qualitative.py --dataset-dir D:/Programming/python/semicondata --num-samples 6 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.datasets.scanner import DatasetPair, DatasetScanner
from src.engine.inference import slide_window_inference
from src.utils.logger import setup_logger
from src.utils.qualitative_evaluator import QualitativeEvaluator
from src.utils.seed import set_seed


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for qualitative evaluation CLI.

    Args:
        args: Optional list of command line argument strings.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Qualitative Restoration Failure Analysis on SEM Micrographs."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="D:/Programming/python/semicondata",
        help="Path to root SEM dataset directory (default: D:/Programming/python/semicondata).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate ('train' or 'test').",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=str,
        default="outputs/predictions/exp001",
        help="Directory containing verified exp001 baseline .npy predictions.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=None,
        help="Path to verified exp001 model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--improved-predictions",
        type=str,
        default="outputs/predictions/exp038",
        help="Directory containing verified Issue #38 improved .npy predictions.",
    )
    parser.add_argument(
        "--improved-checkpoint",
        type=str,
        default=None,
        help="Path to verified Issue #38 model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--include-bicubic-ref",
        action="store_true",
        help="Include a separate explicitly labeled 'Bicubic Reference' comparison panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/images/qualitative_analysis",
        help="Directory to save generated comparison grid PNGs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of samples to evaluate (default 6).",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Explicit list of sample IDs to evaluate (e.g. --sample-ids 000000 000001).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection (default 42).",
    )
    parser.add_argument(
        "--crop-bbox",
        nargs=4,
        type=int,
        default=[64, 64, 192, 192],
        help="Crop bounding box coordinates for zoom view: ymin xmin ymax xmax (default 64 64 192 192).",
    )
    return parser.parse_args(args)


def resolve_prediction_or_inference(
    sample_id: str,
    input_path: Path,
    predictions_dir: Optional[Path],
    checkpoint_path: Optional[Path],
    logger: Any,
    label: str,
) -> Tuple[Optional[np.ndarray], str]:
    """Resolve prediction array from verified predictions directory or checkpoint.

    Returns:
        Tuple[Optional[np.ndarray], str]: Array if available, and status message string.
    """
    # 1. Check pre-computed predictions directory
    if predictions_dir and predictions_dir.exists():
        pred_file = predictions_dir / f"{sample_id}.npy"
        if pred_file.exists():
            try:
                arr = np.load(pred_file)
                return arr, "available"
            except Exception as e:
                logger.warning(f"Failed to load prediction file {pred_file}: {e}")

    # 2. Check model checkpoint
    if checkpoint_path and checkpoint_path.exists():
        try:
            from scripts.evaluate import load_model_and_weights

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_model_and_weights(checkpoint_path=checkpoint_path, config_path=None, device=device)

            raw_arr = np.load(input_path)
            raw_arr = np.array(raw_arr, dtype=np.float32)
            tensor = torch.from_numpy(raw_arr)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)

            out_tensor = slide_window_inference(model=model, x=tensor, device=device)
            out_arr = out_tensor.cpu().numpy()
            if out_arr.ndim > 2 and out_arr.shape[0] == 1:
                out_arr = out_arr.squeeze(0)
            return out_arr, "available"
        except Exception as e:
            logger.warning(f"Failed checkpoint inference for {label} from {checkpoint_path}: {e}")

    if label == "exp001 Baseline":
        return None, "exp001 Baseline\n(Unavailable)"
    return None, "Pending Issue #38\ntraining results"


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for qualitative evaluation CLI."""
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    logger = setup_logger(name="QualitativeEvaluatorCLI", log_dir="logs")

    dataset_root = Path(args.dataset_dir).resolve()
    if not dataset_root.exists():
        logger.error(f"Dataset root directory not found: {dataset_root}")
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")

    scanner = DatasetScanner(dataset_root)
    pairs: List[DatasetPair] = scanner.scan_split(args.split)
    logger.info(f"Discovered {len(pairs)} samples in split '{args.split}'")

    # Select samples deterministically
    if args.sample_ids:
        pair_map = {p.sample_id: p for p in pairs}
        selected_pairs = [pair_map[sid] for sid in args.sample_ids if sid in pair_map]
        if not selected_pairs:
            logger.error(f"None of specified sample IDs {args.sample_ids} found in dataset split.")
            raise ValueError(f"Specified sample IDs not found in split.")
    else:
        rng = np.random.RandomState(args.seed)
        num_sel = min(args.num_samples, len(pairs))
        indices = rng.choice(len(pairs), size=num_sel, replace=False)
        indices.sort()
        selected_pairs = [pairs[i] for i in indices]

    logger.info(f"Selected {len(selected_pairs)} samples for qualitative evaluation (seed={args.seed})")

    baseline_dir = Path(args.baseline_predictions).resolve() if args.baseline_predictions else None
    baseline_ckpt = Path(args.baseline_checkpoint).resolve() if args.baseline_checkpoint else None
    improved_dir = Path(args.improved_predictions).resolve() if args.improved_predictions else None
    improved_ckpt = Path(args.improved_checkpoint).resolve() if args.improved_checkpoint else None

    evaluator = QualitativeEvaluator(output_dir=args.output_dir)
    crop_bbox = tuple(args.crop_bbox)

    summary_artifacts = []

    for pair in selected_pairs:
        logger.info(f"Processing sample: {pair.sample_id}")
        raw_path = pair.input_path
        gt_path = pair.target_path

        base_arr, base_msg = resolve_prediction_or_inference(
            sample_id=pair.sample_id,
            input_path=raw_path,
            predictions_dir=baseline_dir,
            checkpoint_path=baseline_ckpt,
            logger=logger,
            label="exp001 Baseline",
        )

        imp_arr, imp_msg = resolve_prediction_or_inference(
            sample_id=pair.sample_id,
            input_path=raw_path,
            predictions_dir=improved_dir,
            checkpoint_path=improved_ckpt,
            logger=logger,
            label="Improved Model",
        )

        bicubic_ref = None
        if args.include_bicubic_ref:
            raw_arr = QualitativeEvaluator.load_array(raw_path)
            bicubic_ref = QualitativeEvaluator.align_spatial_dimensions(raw_arr, (256, 256))

        grid_path = evaluator.render_comparison_grid(
            raw_input=raw_path,
            gt_target=gt_path,
            baseline_pred=base_arr,
            improved_pred=imp_arr,
            bicubic_ref=bicubic_ref,
            sample_id=pair.sample_id,
            baseline_status_msg=base_msg,
            improved_status_msg=imp_msg,
        )

        zoom_path = evaluator.render_zoom_crop(
            raw_input=raw_path,
            crop_bbox=crop_bbox,
            gt_target=gt_path,
            baseline_pred=base_arr,
            improved_pred=imp_arr,
            bicubic_ref=bicubic_ref,
            sample_id=pair.sample_id,
        )

        summary_artifacts.append(
            {
                "sample_id": pair.sample_id,
                "grid_path": grid_path,
                "zoom_path": zoom_path,
                "baseline_status": "Available" if base_arr is not None else "Unavailable",
                "improved_status": "Available" if imp_arr is not None else "Pending Issue #38",
            }
        )

    logger.info("Qualitative evaluation run completed successfully.")
    for art in summary_artifacts:
        logger.info(
            f"Sample {art['sample_id']} | Baseline: {art['baseline_status']} | Improved: {art['improved_status']} | Grid: {art['grid_path'].name}"
        )


if __name__ == "__main__":
    main()
