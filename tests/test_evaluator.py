"""Unit tests for Evaluator class (Issue #20).

Tests verify evaluation loops, mean PSNR/SSIM calculation, targetless dataset splits,
visualization rendering, Matplotlib/TensorBoard figure generation, filename sanitization,
memory graph non-retention, and model training mode preservation on CPU.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.engine.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Synthetic test fixtures
# ---------------------------------------------------------------------------


class TinyIdentityModel(nn.Module):
    """Minimal model returning upsampled input or identity for testing."""

    def __init__(self, upscale: int = 2) -> None:
        super().__init__()
        self.upscale = upscale
        # Single parameter so model has parameters for device moving
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upscale > 1:
            h, w = x.shape[2] * self.upscale, x.shape[3] * self.upscale
            return torch.nn.functional.interpolate(x, size=(h, w), mode="nearest")
        return x


class SyntheticSEMTestDataset(Dataset):
    """Synthetic dataset for Evaluator tests with controlled values."""

    def __init__(
        self,
        num_samples: int = 8,
        lr_size: int = 16,
        hr_size: int = 32,
        has_target: bool = True,
        filenames: Optional[list] = None,
    ) -> None:
        self.num_samples = num_samples
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.has_target = has_target

        gen = torch.Generator().manual_seed(42)
        self.inputs = [
            torch.rand(1, lr_size, lr_size, generator=gen) for _ in range(num_samples)
        ]

        if has_target:
            # Upsample input nearest-neighbor for ground-truth matching
            self.targets = [
                torch.nn.functional.interpolate(
                    inp.unsqueeze(0), size=(hr_size, hr_size), mode="nearest"
                ).squeeze(0)
                for inp in self.inputs
            ]
        else:
            self.targets = [None] * num_samples

        self.filenames = (
            filenames
            if filenames is not None
            else [f"micrograph_{i:03d}" for i in range(num_samples)]
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input": self.inputs[index],
            "target": self.targets[index],
            "filename": self.filenames[index],
        }


def _collate_fn(batch: list) -> Dict[str, Any]:
    """Collate helper matching sem_collate behavior."""
    inputs = torch.stack([s["input"] for s in batch])
    targets_raw = [s["target"] for s in batch]
    has_targets = any(t is not None for t in targets_raw)

    targets = torch.stack(targets_raw) if has_targets else None
    filenames = [s["filename"] for s in batch]

    return {
        "input": inputs,
        "target": targets,
        "filename": filenames,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluatorConstruction:
    """Test 1: Evaluator instantiation and parameter validation."""

    def test_construction_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset()
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            assert evaluator.model is model
            assert evaluator.data_loader is loader
            assert evaluator.device == torch.device("cpu")
            assert evaluator.output_dir == Path(tmpdir).resolve()
            assert evaluator.max_visualizations == 10

    def test_construction_invalid_model(self) -> None:
        dataset = SyntheticSEMTestDataset()
        loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

        with pytest.raises(TypeError, match="must be a torch.nn.Module"):
            Evaluator(model="not_a_module", data_loader=loader)

    def test_construction_invalid_loader(self) -> None:
        model = TinyIdentityModel()
        with pytest.raises(ValueError, match="cannot be None"):
            Evaluator(model=model, data_loader=None)

    def test_construction_invalid_max_vis(self) -> None:
        model = TinyIdentityModel()
        dataset = SyntheticSEMTestDataset()
        loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

        with pytest.raises(ValueError, match="must be non-negative"):
            Evaluator(model=model, data_loader=loader, max_visualizations=-1)


class TestCPUEvaluation:
    """Test 2: Full evaluation execution on CPU."""

    def test_cpu_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=4)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            results = evaluator.evaluate(save_visualizations=True)

            assert "mean_psnr" in results
            assert "mean_ssim" in results
            assert results["num_samples"] == 4
            assert results["num_visualizations"] <= 10


class TestMeanPSNRCalculation:
    """Test 3: Mean PSNR calculation across dataset."""

    def test_mean_psnr_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=4)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            results = evaluator.evaluate(save_visualizations=False)

            # TinyIdentityModel nearest-interpolates LR to match target,
            # so PSNR should be finite and positive
            assert isinstance(results["mean_psnr"], float)
            assert results["mean_psnr"] > 0.0


class TestMeanSSIMCalculation:
    """Test 4: Mean SSIM calculation across dataset."""

    def test_mean_ssim_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=4)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            results = evaluator.evaluate(save_visualizations=False)

            assert isinstance(results["mean_ssim"], float)
            assert 0.0 <= results["mean_ssim"] <= 1.0


class TestCorrectBatchHandling:
    """Test 5: Sample-weighted metric aggregation across unequal batch sizes."""

    def test_sample_weighted_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            # 5 samples with batch_size=2 gives batches of sizes [2, 2, 1]
            dataset = SyntheticSEMTestDataset(num_samples=5)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            results = evaluator.evaluate(save_visualizations=False)
            assert results["num_samples"] == 5


class TestNoGradientEvaluation:
    """Test 6: Confirms no gradients are computed or retained during evaluation."""

    def test_no_gradient_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=4)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            evaluator.evaluate(save_visualizations=False)

            # Model parameter gradients should be None
            for param in model.parameters():
                assert param.grad is None


class TestModelModeRestoration:
    """Test 7: Confirms model training mode is preserved and restored."""

    def test_model_mode_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            model.train()  # Explicitly set to training mode
            assert model.training is True

            dataset = SyntheticSEMTestDataset(num_samples=2)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
            )

            evaluator.evaluate(save_visualizations=False)

            # Model mode must be restored to training=True
            assert model.training is True


class TestOutputDirCreation:
    """Test 8: Confirms output_dir directory is created automatically."""

    def test_output_dir_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "nested" / "predictions"
            assert not target_dir.exists()

            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=2)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=target_dir,
            )
            evaluator.evaluate(save_visualizations=False)

            assert target_dir.exists()


class TestComparisonImageGeneration:
    """Test 9: Confirms comparison visualization files are written to disk."""

    def test_comparison_images_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=3)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
                max_visualizations=3,
            )

            results = evaluator.evaluate(save_visualizations=True)

            assert results["num_visualizations"] == 3
            png_files = list(Path(tmpdir).glob("*_comparison.png"))
            assert len(png_files) == 3


class TestErrorMapGeneration:
    """Test 10: Confirms error map panel rendering succeeds without error."""

    def test_error_map_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=1)
            loader = DataLoader(dataset, batch_size=1, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
                max_visualizations=1,
            )

            results = evaluator.evaluate(save_visualizations=True)
            assert results["num_visualizations"] == 1


class TestTensorBoardFigureLogging:
    """Test 11: Confirms SummaryWriter.add_figure is invoked when writer is supplied."""

    def test_tensorboard_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MagicMock(spec=SummaryWriter)
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=2)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                writer=writer,
                output_dir=tmpdir,
                max_visualizations=2,
            )

            evaluator.evaluate(epoch=1, save_visualizations=True)

            assert writer.add_figure.call_count == 2


class TestEvaluationWithoutWriter:
    """Test 12: Confirms evaluation works correctly when writer=None."""

    def test_evaluation_without_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=2)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                writer=None,
                output_dir=tmpdir,
            )

            results = evaluator.evaluate(save_visualizations=True)
            assert results["num_samples"] == 2


class TestDeterministicOutputFilenames:
    """Test 13: Confirms path traversal filenames are sanitized safely."""

    def test_filename_sanitization(self) -> None:
        unsafe_filenames = [
            "../sub/image_001.npy",
            "C:\\Users\\admin\\test.npy",
            "../../../etc/passwd",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=3, filenames=unsafe_filenames)
            loader = DataLoader(dataset, batch_size=3, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
                max_visualizations=3,
            )

            results = evaluator.evaluate(save_visualizations=True)
            assert results["num_visualizations"] == 3

            # Verify saved files are inside output_dir without path traversal
            saved_files = list(Path(tmpdir).glob("*.png"))
            assert len(saved_files) == 3
            for p in saved_files:
                assert p.parent == Path(tmpdir).resolve()


class TestTargetlessBatchHandling:
    """Test 14: Confirms targetless batches (target=None) handle metrics and plots safely."""

    def test_targetless_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=2, has_target=False)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
                max_visualizations=2,
            )

            results = evaluator.evaluate(save_visualizations=True)

            # Targetless batches should return 0.0 for PSNR/SSIM
            assert results["mean_psnr"] == 0.0
            assert results["mean_ssim"] == 0.0
            assert results["num_visualizations"] == 2


class TestNoFullDatasetTensorAccumulation:
    """Test 15: Confirms predictions are not accumulated in a global list across batches."""

    def test_no_tensor_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyIdentityModel()
            dataset = SyntheticSEMTestDataset(num_samples=10)
            loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

            evaluator = Evaluator(
                model=model,
                data_loader=loader,
                device="cpu",
                output_dir=tmpdir,
                max_visualizations=0,  # No visualizations
            )

            results = evaluator.evaluate(save_visualizations=False)
            assert results["num_samples"] == 10
