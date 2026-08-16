"""Pytest test suite for qualitative restoration failure analysis module (src/utils/qualitative_evaluator.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.scanner import DatasetScanner
from src.utils.qualitative_evaluator import QualitativeEvaluator


def test_fixed_intensity_normalization() -> None:
    """Test that normalize_fixed preserves raw unclipped intensity without artificial clipping."""
    raw_arr = np.array([[-0.5, 0.5], [1.5, 0.8]], dtype=np.float32)
    norm_arr = QualitativeEvaluator.normalize_fixed(raw_arr)

    assert np.array_equal(norm_arr, raw_arr)
    assert norm_arr[0, 0] == -0.5
    assert norm_arr[0, 1] == 0.5
    assert norm_arr[1, 0] == 1.5
    assert norm_arr[1, 1] == 0.8


def test_load_array_handling(tmp_path: Path) -> None:
    """Test array loading from file path, NumPy array, and PyTorch Tensor."""
    arr = np.ones((128, 128), dtype=np.float32) * 0.5
    file_path = tmp_path / "test_sample.npy"
    np.save(file_path, arr)

    loaded_from_file = QualitativeEvaluator.load_array(file_path)
    assert isinstance(loaded_from_file, np.ndarray)
    assert loaded_from_file.shape == (128, 128)

    tensor = torch.ones((1, 1, 128, 128), dtype=torch.float32) * 0.5
    loaded_from_tensor = QualitativeEvaluator.load_array(tensor)
    assert isinstance(loaded_from_tensor, np.ndarray)
    assert loaded_from_tensor.shape == (128, 128)

    loaded_from_numpy = QualitativeEvaluator.load_array(arr)
    assert isinstance(loaded_from_numpy, np.ndarray)
    assert loaded_from_numpy.shape == (128, 128)


def test_spatial_dimension_alignment() -> None:
    """Test spatial dimension upsampling for visual grid alignment."""
    raw_arr = np.ones((128, 128), dtype=np.float32)
    aligned_arr = QualitativeEvaluator.align_spatial_dimensions(raw_arr, (256, 256))

    assert aligned_arr.shape == (256, 256)
    assert aligned_arr.dtype == np.float32


def test_grid_rendering_and_output_file_creation(tmp_path: Path) -> None:
    """Test full comparison grid and zoom crop rendering file outputs."""
    evaluator = QualitativeEvaluator(output_dir=tmp_path / "qual_results", dpi=100)

    raw_arr = np.random.rand(128, 128).astype(np.float32)
    gt_arr = np.random.rand(256, 256).astype(np.float32)
    base_arr = np.random.rand(256, 256).astype(np.float32)

    grid_path = evaluator.render_comparison_grid(
        raw_input=raw_arr,
        gt_target=gt_arr,
        baseline_pred=base_arr,
        improved_pred=None,
        sample_id="test_001",
    )

    assert grid_path.exists()
    assert grid_path.stat().st_size > 0

    zoom_path = evaluator.render_zoom_crop(
        raw_input=raw_arr,
        crop_bbox=(32, 32, 96, 96),
        gt_target=gt_arr,
        baseline_pred=base_arr,
        improved_pred=None,
        sample_id="test_001",
    )

    assert zoom_path.exists()
    assert zoom_path.stat().st_size > 0


def test_missing_baseline_and_improved_model_handling(tmp_path: Path) -> None:
    """Test rendering when baseline and improved predictions are missing/unavailable."""
    evaluator = QualitativeEvaluator(output_dir=tmp_path / "qual_results", dpi=100)

    raw_arr = np.zeros((128, 128), dtype=np.float32)
    gt_arr = np.ones((256, 256), dtype=np.float32)

    grid_path = evaluator.render_comparison_grid(
        raw_input=raw_arr,
        gt_target=gt_arr,
        baseline_pred=None,
        improved_pred=None,
        sample_id="test_missing_models",
        baseline_status_msg="exp001 Baseline\n(Unavailable)",
        improved_status_msg="Pending Issue #38\ntraining results",
    )

    assert grid_path.exists()
    assert grid_path.stat().st_size > 0


def test_unpaired_test_split_missing_gt_handling(tmp_path: Path) -> None:
    """Test rendering when GT target is None (unpaired test split)."""
    evaluator = QualitativeEvaluator(output_dir=tmp_path / "qual_results", dpi=100)

    raw_arr = np.zeros((128, 128), dtype=np.float32)

    grid_path = evaluator.render_comparison_grid(
        raw_input=raw_arr,
        gt_target=None,
        baseline_pred=None,
        improved_pred=None,
        sample_id="test_unpaired",
    )

    assert grid_path.exists()
    assert grid_path.stat().st_size > 0


def test_deterministic_sample_selection(tmp_path: Path) -> None:
    """Test deterministic sample selection using random seeds."""
    dataset_root = tmp_path / "sem_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    for i in range(10):
        name = f"sample_{i:03d}.npy"
        np.save(noisy_dir / name, np.zeros((128, 128), dtype=np.float32))
        np.save(gt_dir / name, np.ones((256, 256), dtype=np.float32))

    scanner = DatasetScanner(dataset_root)
    pairs = scanner.scan_split("train")

    rng1 = np.random.RandomState(42)
    sel1 = rng1.choice(len(pairs), size=4, replace=False)

    rng2 = np.random.RandomState(42)
    sel2 = rng2.choice(len(pairs), size=4, replace=False)

    assert np.array_equal(sel1, sel2)


def test_graceful_missing_file_error_handling() -> None:
    """Test FileNotFoundError when loading non-existent array file."""
    non_existent = Path("non_existent_file_12345.npy")
    with pytest.raises(FileNotFoundError):
        QualitativeEvaluator.load_array(non_existent)
