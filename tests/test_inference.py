"""Unit and integration test suite for sliding-window inference pipeline.

Verifies:
- Shape contract preservation for both same-resolution (upscale=1) and super-resolution (upscale=2).
- Complete spatial boundary coverage for non-divisible and arbitrary dimensions.
- Safe padding and unpadding for images smaller than tile size.
- 2D Gaussian spatial blending and weight normalization correctness (seamless blending verification).
- Absence of non-finite values (NaN / Inf).
- Deterministic CPU execution.
- CLI script execution interface (scripts/evaluate.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.predict import main as predict_main
from scripts.predict import parse_args
from src.engine.checkpoint import CheckpointManager
from src.engine.inference import (
    SlidingWindowInference,
    _calculate_tile_starts,
    _generate_gaussian_weights,
    slide_window_inference,
)
from src.models.nafnet import NAFNet


class DummyModel(nn.Module):
    """Dummy model returning constant output matching upscale contract."""

    def __init__(self, upscale: int = 1, fill_val: float = 0.5) -> None:
        super().__init__()
        self.upscale = upscale
        self.fill_val = fill_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out_h = H * self.upscale
        out_w = W * self.upscale
        return torch.full(
            (B, C, out_h, out_w), self.fill_val, dtype=x.dtype, device=x.device
        )


class IdentityModel(nn.Module):
    """Identity model for upscale=1 or upscale=2 testing exact intensity preservation."""

    def __init__(self, upscale: int = 1) -> None:
        super().__init__()
        self.upscale = upscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upscale == 1:
            return x
        return nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")


def test_tile_starts_calculation() -> None:
    """Verify tile start coordinate calculation for various dimensions."""
    # Length equal to tile size
    assert _calculate_tile_starts(length=512, tile_size=512, stride=384) == [0]

    # Length smaller than tile size
    assert _calculate_tile_starts(length=256, tile_size=512, stride=384) == [0]

    # Length exactly divisible by stride
    assert _calculate_tile_starts(length=1280, tile_size=512, stride=384) == [
        0,
        384,
        768,
    ]

    # Length not divisible by stride; must append last_start (length - tile_size)
    starts = _calculate_tile_starts(length=1000, tile_size=512, stride=384)
    assert starts == [0, 384, 488]
    assert starts[-1] == 1000 - 512


def test_gaussian_weight_matrix() -> None:
    """Verify Gaussian weighting matrix properties: max at center, non-zero positive weights."""
    weight = _generate_gaussian_weights(
        tile_h=64, tile_w=64, device=torch.device("cpu")
    )
    assert weight.shape == (1, 1, 64, 64)

    arr = weight.squeeze().numpy()
    center_val = arr[31, 31]
    corner_val = arr[0, 0]

    # Center weight should be maximum and corners non-zero positive
    assert center_val > corner_val
    assert corner_val >= 1e-3
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_parameter_validation() -> None:
    """Verify parameter bounds checking for SlidingWindowInference."""
    model = DummyModel()
    with pytest.raises(TypeError, match="torch.nn.Module"):
        SlidingWindowInference(model="not_a_model")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_size"):
        SlidingWindowInference(model=model, tile_size=0)

    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowInference(model=model, overlap=1.0)

    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowInference(model=model, overlap=-0.1)

    with pytest.raises(ValueError, match="tile_batch_size"):
        SlidingWindowInference(model=model, tile_batch_size=0)


def test_upscale_1_restoration_contract() -> None:
    """Verify shape and value contract for same-resolution (upscale=1) inference."""
    model = DummyModel(upscale=1, fill_val=0.7)
    x = torch.randn(1, 64, 64)
    out = slide_window_inference(model, x, tile_size=32, overlap=0.25)

    assert out.shape == (1, 64, 64)
    assert torch.allclose(out, torch.tensor(0.7), atol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_upscale_2_super_resolution_contract() -> None:
    """Verify shape contract for super-resolution (upscale=2) inference."""
    model = DummyModel(upscale=2, fill_val=0.4)
    x = torch.randn(1, 100, 100)
    out = slide_window_inference(model, x, tile_size=64, overlap=0.5)

    # 100x100 input -> 200x200 output for upscale=2
    assert out.shape == (1, 200, 200)
    assert torch.allclose(out, torch.tensor(0.4), atol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_image_smaller_than_tile() -> None:
    """Verify safe padding and unpadding when input image is smaller than tile size."""
    model = DummyModel(upscale=2, fill_val=0.5)
    x = torch.randn(1, 32, 32)
    # tile_size=64 is larger than input 32x32
    out = slide_window_inference(model, x, tile_size=64, overlap=0.25)

    assert out.shape == (1, 64, 64)
    assert not torch.isnan(out).any()


def test_sub_2px_small_image() -> None:
    """Verify fallback constant padding for 1x1 image."""
    model = DummyModel(upscale=1, fill_val=0.8)
    x = torch.full((1, 1, 1), 0.3)
    out = slide_window_inference(model, x, tile_size=16, overlap=0.25)

    assert out.shape == (1, 1, 1)
    assert not torch.isnan(out).any()


def test_non_divisible_dimensions() -> None:
    """Verify inference on non-divisible spatial dimensions (e.g. 137x213)."""
    model = DummyModel(upscale=2, fill_val=0.5)
    x = torch.randn(1, 137, 213)
    out = slide_window_inference(model, x, tile_size=64, overlap=0.25)

    assert out.shape == (1, 274, 426)
    assert not torch.isnan(out).any()


def test_seamless_gaussian_blending_normalization() -> None:
    """Verify that Gaussian weighted accumulation and normalization accurately reconstruct constant field without seams."""
    model = IdentityModel(upscale=1)
    # Create constant image field of 0.75
    x = torch.full((1, 200, 300), 0.75)
    out = slide_window_inference(model, x, tile_size=64, overlap=0.5, use_gaussian=True)

    assert out.shape == (1, 200, 300)
    # Output must match constant input field exactly across overlapping boundaries
    assert torch.allclose(out, x, atol=1e-5)
    assert not torch.isnan(out).any()


def test_tile_batch_size_invariance() -> None:
    """Verify that varying tile_batch_size yields identical outputs."""
    model = NAFNet(
        img_channel=1,
        width=16,
        enc_blk_nums=[1],
        dec_blk_nums=[1],
        middle_blk_num=1,
        upscale=2,
    )
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 40, 40)

    out_b1 = slide_window_inference(
        model, x, tile_size=32, overlap=0.25, tile_batch_size=1
    )
    out_b4 = slide_window_inference(
        model, x, tile_size=32, overlap=0.25, tile_batch_size=4
    )

    assert torch.allclose(out_b1, out_b4, atol=1e-5)


def test_deterministic_cpu_execution() -> None:
    """Verify execution determinism on CPU across multiple runs."""
    model = NAFNet(
        img_channel=1,
        width=16,
        enc_blk_nums=[1],
        dec_blk_nums=[1],
        middle_blk_num=1,
        upscale=2,
    )
    model.eval()

    x = torch.ones(1, 48, 48)
    run1 = slide_window_inference(model, x, tile_size=32, overlap=0.25, device="cpu")
    run2 = slide_window_inference(model, x, tile_size=32, overlap=0.25, device="cpu")

    assert torch.equal(run1, run2)


def test_predict_cli_single_file_and_directory(tmp_path: Path) -> None:
    """Integration test verifying scripts/evaluate.py for single files and directories."""
    # 1. Create a dummy checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=ckpt_dir)
    model = NAFNet(
        img_channel=1,
        width=16,
        enc_blk_nums=[1],
        dec_blk_nums=[1],
        middle_blk_num=1,
        upscale=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = manager.save(epoch=1, model=model, optimizer=optimizer, metric=25.0)

    # 2. Test single file inference
    single_input = tmp_path / "test_input.npy"
    single_output = tmp_path / "test_output.npy"
    np.save(single_input, np.ones((64, 64), dtype=np.float32) * 0.5)

    args_single = [
        "--checkpoint",
        str(ckpt_path),
        "--input",
        str(single_input),
        "--output",
        str(single_output),
        "--tile-size",
        "32",
        "--overlap",
        "0.25",
        "--device",
        "cpu",
    ]
    parsed = parse_args(args_single)
    predict_main(parsed)

    assert single_output.exists()
    arr_single = np.load(single_output)
    assert arr_single.shape == (128, 128)
    assert not np.isnan(arr_single).any()

    # 3. Test directory inference
    in_dir = tmp_path / "input_dir"
    out_dir = tmp_path / "output_dir"
    in_dir.mkdir(parents=True, exist_ok=True)

    np.save(in_dir / "001.npy", np.full((32, 32), 0.2, dtype=np.float32))
    np.save(in_dir / "002.npy", np.full((40, 40), 0.8, dtype=np.float32))

    args_dir = [
        "--checkpoint",
        str(ckpt_path),
        "--input",
        str(in_dir),
        "--output",
        str(out_dir),
        "--tile-size",
        "32",
        "--overlap",
        "0.25",
        "--device",
        "cpu",
    ]
    parsed_dir = parse_args(args_dir)
    predict_main(parsed_dir)

    assert (out_dir / "001.npy").exists()
    assert (out_dir / "002.npy").exists()
    assert np.load(out_dir / "001.npy").shape == (64, 64)
    assert np.load(out_dir / "002.npy").shape == (80, 80)
