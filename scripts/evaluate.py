"""CLI script for running patch-tiling sliding-window inference on SEM micrographs.

This script loads a trained model checkpoint and executes sliding-window inference with
Gaussian spatial blending on single .npy files or entire input directories.

Example:
    Run prediction on a single array file:
        $ python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --input data/sample.npy --output predictions/sample_restored.npy

    Run prediction on an entire directory of micrographs:
        $ python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --input data/test_dir --output predictions/restored_dir --tile-size 512 --overlap 0.25
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


import numpy as np
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.engine.checkpoint import CheckpointManager
from src.engine.inference import slide_window_inference
from src.models.builder import build_model
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for prediction CLI.

    Args:
        args: Optional list of command line argument strings.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run sliding-window NAFNet inference on SEM micrographs."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .npy file or directory containing .npy files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output .npy file or output directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration YAML file.",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Input spatial tile size in pixels (default 512).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Fractional tile overlap ratio, 0.0 <= overlap < 1.0 (default 0.25).",
    )
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=1,
        help="Tile mini-batch size per model pass (default 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target execution device ('auto', 'cuda', or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default 42).",
    )
    return parser.parse_args(args)


def resolve_device(device_setting: str) -> str:
    """Resolve device setting to explicit 'cuda' or 'cpu' string."""
    if device_setting.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def infer_nafnet_params_from_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """Infer NAFNet constructor hyperparameters directly from a model state_dict."""
    if "intro.weight" not in state_dict:
        return {}

    width = int(state_dict["intro.weight"].shape[0])
    img_channel = int(state_dict["intro.weight"].shape[1])

    upscale = 1
    if "up_tail.0.weight" in state_dict:
        up_out = int(state_dict["up_tail.0.weight"].shape[0])
        upscale = int(math.isqrt(up_out // img_channel))

    enc_stages: set[int] = set()
    enc_blocks: Dict[int, set[int]] = {}
    for key in state_dict.keys():
        parts = key.split(".")
        if parts[0] == "encoders" and len(parts) > 2 and parts[1].isdigit() and parts[2].isdigit():
            s_idx = int(parts[1])
            b_idx = int(parts[2])
            enc_stages.add(s_idx)
            enc_blocks.setdefault(s_idx, set()).add(b_idx)

    enc_blk_nums = (
        [len(enc_blocks[i]) for i in sorted(enc_stages)]
        if enc_stages
        else [1, 1, 1]
    )

    dec_stages: set[int] = set()
    dec_blocks: Dict[int, set[int]] = {}
    for key in state_dict.keys():
        parts = key.split(".")
        if parts[0] == "decoders" and len(parts) > 2 and parts[1].isdigit() and parts[2].isdigit():
            s_idx = int(parts[1])
            b_idx = int(parts[2])
            dec_stages.add(s_idx)
            dec_blocks.setdefault(s_idx, set()).add(b_idx)

    dec_blk_nums = (
        [len(dec_blocks[i]) for i in sorted(dec_stages)]
        if dec_stages
        else enc_blk_nums
    )

    mid_blocks: set[int] = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if parts[0] == "middle_blks" and len(parts) > 1 and parts[1].isdigit():
            mid_blocks.add(int(parts[1]))

    middle_blk_num = len(mid_blocks) if mid_blocks else 1

    return {
        "img_channel": img_channel,
        "width": width,
        "middle_blk_num": middle_blk_num,
        "enc_blk_nums": enc_blk_nums,
        "dec_blk_nums": dec_blk_nums,
        "upscale": upscale,
    }


def load_model_and_weights(
    checkpoint_path: Path, config_path: Optional[Path], device: str
) -> torch.nn.Module:
    """Load model architecture and weights from checkpoint and optional config."""
    ckpt_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model: Optional[torch.nn.Module] = None

    if config_path and config_path.exists():
        cfg = load_config(config_path=config_path)
        model = build_model(cfg)
    elif isinstance(ckpt_dict, dict) and "config" in ckpt_dict:
        model = build_model(ckpt_dict["config"])

    state_dict = (
        ckpt_dict["model"]
        if isinstance(ckpt_dict, dict) and "model" in ckpt_dict
        else (
            ckpt_dict["model_state_dict"]
            if isinstance(ckpt_dict, dict) and "model_state_dict" in ckpt_dict
            else (
                ckpt_dict.get("state_dict", ckpt_dict)
                if isinstance(ckpt_dict, dict)
                else ckpt_dict
            )
        )
    )

    if model is None:
        inferred_params = infer_nafnet_params_from_state_dict(state_dict)
        if inferred_params:
            from src.models.nafnet import NAFNet

            model = NAFNet(**inferred_params)

    if model is None:
        default_cfg_path = Path("configs/train.yaml")
        if default_cfg_path.exists():
            try:
                cfg = load_config(config_path=default_cfg_path)
                candidate_model = build_model(cfg)
                candidate_model.load_state_dict(state_dict)
                model = candidate_model
            except Exception:
                model = None

    if model is None:
        from src.models.nafnet import NAFNet

        model = NAFNet()

    model.load_state_dict(state_dict)
    model.to(device)
    return model




def process_single_file(
    input_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    tile_size: int,
    overlap: float,
    tile_batch_size: int,
    device: str,
) -> None:
    """Load, process, and save a single micrograph array file."""
    arr_raw = np.load(input_path)
    arr = np.array(arr_raw, dtype=np.float32)

    # Removed: arr = np.clip(arr, 0.0, 1.0) to preserve >1.0 intensities
    tensor = torch.from_numpy(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    output_tensor = slide_window_inference(
        model=model,
        x=tensor,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        device=device,
        use_gaussian=True,
    )

    out_arr = output_tensor.cpu().numpy()
    if out_arr.shape[0] == 1:
        out_arr = out_arr.squeeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out_arr)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main execution entry point for evaluate.py."""
    if args is None:
        args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file or directory not found: {input_path}")

    output_path = Path(args.output).resolve()

    set_seed(args.seed)
    logger = setup_logger(name="SEM_NAFNet_Predict", log_dir="logs")
    device_str = resolve_device(args.device)
    logger.info(f"Resolved execution device: {device_str}")

    # Build model and load weights using load_model_and_weights
    config_path = Path(args.config).resolve() if args.config else None
    model = load_model_and_weights(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device_str,
    )
    logger.info(f"Successfully loaded model and weights from {checkpoint_path}")


    if input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        process_single_file(
            input_path=input_path,
            output_path=output_path,
            model=model,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_batch_size=args.tile_batch_size,
            device=device_str,
        )
        logger.info(f"Saved prediction to {output_path}")
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        files = sorted(
            [
                f
                for f in input_path.glob("*.npy")
                if not f.name.startswith("._") and "__MACOSX" not in str(f)
            ]
        )
        logger.info(f"Processing {len(files)} .npy files in directory: {input_path}")

        for file_path in files:
            target_file = output_path / file_path.name
            process_single_file(
                input_path=file_path,
                output_path=target_file,
                model=model,
                tile_size=args.tile_size,
                overlap=args.overlap,
                tile_batch_size=args.tile_batch_size,
                device=device_str,
            )
        logger.info(f"Batch prediction completed. Saved all outputs to {output_path}")
    else:
        raise ValueError(f"Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
