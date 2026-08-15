"""Comprehensive integration and pipeline verification test suite.

Verifies:
- Dataset -> DataLoader -> NAFNet model forward pass tensor contracts
- Model prediction -> Metrics (PSNR / SSIM / LPIPS) evaluation compatibility
- Sliding-window inference with 2D Gaussian spatial blending and exact shape contracts
- Arbitrary rectangular full-frame micrographs without boundary loss (300x400 -> 600x800)
- End-to-end smoke test on synthetic .npy files executing entirely on CPU
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.collate import sem_collate
from src.datasets.sem_dataset import SEMDataset
from src.engine.inference import SlidingWindowInference, slide_window_inference
from src.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from src.models.nafnet import NAFNet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lightweight_model() -> NAFNet:
    """Create a fast, lightweight NAFNet model on CPU for integration testing."""
    model = NAFNet(
        img_channel=1,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        upscale=2,
        drop_out_rate=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def synthetic_pipeline_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic paired dataset with 4 samples for pipeline testing."""
    dataset_root = tmp_path / "pipeline_sem_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 5):
        sample_id = f"pipe_{i:03d}"
        # NoisyLR: 128x128 float32 in range [0, 1]
        noisy_arr = np.random.RandomState(i + 10).rand(128, 128).astype(np.float32)
        # GT: 256x256 float32 in range [0, 1]
        gt_arr = np.random.RandomState(i + 50).rand(256, 256).astype(np.float32)

        np.save(noisy_dir / f"{sample_id}.npy", noisy_arr)
        np.save(gt_dir / f"{sample_id}.npy", gt_arr)

    return dataset_root


# ---------------------------------------------------------------------------
# Test Group A: Dataset -> DataLoader -> Model Integration
# ---------------------------------------------------------------------------


class TestDatasetToModelIntegration:
    """Tests data flow from dataset through DataLoader into model forward pass."""

    def test_dataloader_to_model_forward(
        self, synthetic_pipeline_dataset_dir: Path, lightweight_model: NAFNet
    ) -> None:
        """DataLoader batches are correctly ingested and processed by NAFNet."""
        dataset = SEMDataset(synthetic_pipeline_dataset_dir, split="train")
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=sem_collate,
        )

        batch = next(iter(loader))
        assert "input" in batch
        assert "target" in batch
        assert batch["input"].shape == (2, 1, 128, 128)
        assert batch["target"].shape == (2, 1, 256, 256)

        with torch.inference_mode():
            output = lightweight_model(batch["input"])

        assert output.shape == (2, 1, 256, 256)
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all().item()


# ---------------------------------------------------------------------------
# Test Group B: Model Prediction -> Metrics Integration
# ---------------------------------------------------------------------------


class TestModelToMetricsIntegration:
    """Tests integration between model predictions and evaluation metrics."""

    def test_prediction_to_psnr_and_ssim(self, lightweight_model: NAFNet) -> None:
        """Model predictions and ground-truth targets evaluate cleanly in PSNR and SSIM."""
        x = torch.rand(2, 1, 128, 128)
        gt_target = torch.rand(2, 1, 256, 256)

        with torch.inference_mode():
            pred = lightweight_model(x)

        assert pred.shape == gt_target.shape

        # Metrics accept PyTorch 4D batched tensors directly
        psnr_val = calculate_psnr(pred, gt_target, data_range=1.0)
        ssim_val = calculate_ssim(pred, gt_target, data_range=1.0)

        assert isinstance(psnr_val, float)
        assert isinstance(ssim_val, float)
        assert torch.isfinite(torch.tensor(psnr_val)).item()
        assert -1.0 <= ssim_val <= 1.0


# ---------------------------------------------------------------------------
# Test Group C: Sliding-Window Inference Engine Integration
# ---------------------------------------------------------------------------


class TestSlidingWindowInferenceIntegration:
    """Tests sliding-window inference with Gaussian blending and tile accumulation."""

    def test_sliding_window_basic_contract(self, lightweight_model: NAFNet) -> None:
        """SlidingWindowInference restores (1, 1, 128, 128) to (1, 1, 256, 256) with 2x upscale."""
        infer_engine = SlidingWindowInference(
            model=lightweight_model,
            tile_size=64,
            overlap=0.25,
            tile_batch_size=2,
        )

        inp = torch.rand(1, 1, 128, 128)
        output = infer_engine.infer(inp, use_gaussian=True)

        assert output.shape == (1, 1, 256, 256)
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all().item()
        assert not torch.isnan(output).any().item()

    def test_sliding_window_3d_input(self, lightweight_model: NAFNet) -> None:
        """SlidingWindowInference accepts 3D tensor (1, H, W) and returns 3D tensor (1, 2H, 2W)."""
        infer_engine = SlidingWindowInference(
            model=lightweight_model,
            tile_size=64,
            overlap=0.25,
        )

        inp_3d = torch.rand(1, 64, 64)
        output_3d = infer_engine.infer(inp_3d, use_gaussian=True)

        assert output_3d.shape == (1, 128, 128)
        assert output_3d.ndim == 3

    def test_slide_window_inference_convenience_function(
        self, lightweight_model: NAFNet
    ) -> None:
        """Functional slide_window_inference wrapper executes seamlessly."""
        inp = torch.rand(1, 1, 64, 64)
        output = slide_window_inference(
            lightweight_model,
            inp,
            tile_size=32,
            overlap=0.25,
            use_gaussian=True,
        )

        assert output.shape == (1, 1, 128, 128)
        assert torch.isfinite(output).all().item()


# ---------------------------------------------------------------------------
# Test Group D: Rectangular Full-Frame Integration (300x400 -> 600x800)
# ---------------------------------------------------------------------------


class TestRectangularFullFrameIntegration:
    """Tests arbitrary rectangular micrograph restoration without spatial loss."""

    def test_rectangular_micrograph_full_frame(self, lightweight_model: NAFNet) -> None:
        """Sliding window processes rectangular 300x400 image to exact 600x800 output."""
        infer_engine = SlidingWindowInference(
            model=lightweight_model,
            tile_size=128,
            overlap=0.25,
            tile_batch_size=2,
        )

        # 300x400 rectangular input
        rect_input = torch.rand(1, 1, 300, 400)
        rect_output = infer_engine.infer(rect_input, use_gaussian=True)

        # 2x upscale contract guarantees exact dimensions
        assert rect_output.shape == (1, 1, 600, 800)
        assert torch.isfinite(rect_output).all().item()
        assert not torch.isnan(rect_output).any().item()
        assert not torch.isinf(rect_output).any().item()


# ---------------------------------------------------------------------------
# Test Group E: End-to-End CPU Smoke Test
# ---------------------------------------------------------------------------


class TestEndToEndSmokePipeline:
    """Complete end-to-end synthetic pipeline execution on CPU."""

    def test_complete_synthetic_pipeline_smoke(
        self, synthetic_pipeline_dataset_dir: Path, lightweight_model: NAFNet
    ) -> None:
        """Executes disk -> dataset -> DataLoader -> model -> metrics on CPU."""
        # 1. Dataset & DataLoader construction
        dataset = SEMDataset(
            synthetic_pipeline_dataset_dir,
            split="train",
            clip_range=(0.0, 1.0),
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=sem_collate,
        )

        # 2. Iterate through mini-batch
        batch = next(iter(loader))
        inputs = batch["input"]
        targets = batch["target"]
        filenames = batch["filename"]

        assert len(filenames) == 2
        assert inputs.shape == (2, 1, 128, 128)
        assert targets.shape == (2, 1, 256, 256)

        # 3. Model Forward Pass
        with torch.inference_mode():
            predictions = lightweight_model(inputs)

        assert predictions.shape == targets.shape

        # 4. Metric Evaluation
        psnr_score = calculate_psnr(predictions, targets, data_range=1.0)
        ssim_score = calculate_ssim(predictions, targets, data_range=1.0)

        assert isinstance(psnr_score, float)
        assert isinstance(ssim_score, float)
        assert psnr_score > 0.0
        assert -1.0 <= ssim_score <= 1.0
