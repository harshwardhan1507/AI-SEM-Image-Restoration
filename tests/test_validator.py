"""Pytest unit test suite for dataset validator (src/datasets/validator.py)."""

from pathlib import Path

import numpy as np
import pytest

from src.datasets.scanner import DatasetPair
from src.datasets.validator import (
    DatasetValidationError,
    DatasetValidator,
    InvalidDtypeError,
    InvalidShapeError,
)


def test_validate_pair_success(tmp_path: Path) -> None:
    """Test validate_pair succeeds for valid GT (256, 256) float32 and NoisyLR (128, 128) float32 arrays."""
    noisy_file = tmp_path / "noisy_001.npy"
    gt_file = tmp_path / "gt_001.npy"

    np.save(noisy_file, np.zeros((128, 128), dtype=np.float32))
    np.save(gt_file, np.ones((256, 256), dtype=np.float32))

    pair = DatasetPair(sample_id="001", input_path=noisy_file, target_path=gt_file)
    validator = DatasetValidator()

    assert validator.validate_pair(pair) is True


def test_validate_invalid_dtype(tmp_path: Path) -> None:
    """Test InvalidDtypeError raised when array data type is not float32."""
    noisy_file = tmp_path / "noisy_001.npy"
    np.save(noisy_file, np.zeros((128, 128), dtype=np.float64))

    pair = DatasetPair(sample_id="001", input_path=noisy_file, target_path=None)
    validator = DatasetValidator()

    with pytest.raises(InvalidDtypeError, match="Invalid data type"):
        validator.validate_pair(pair)


def test_validate_invalid_shape(tmp_path: Path) -> None:
    """Test InvalidShapeError raised when spatial array shape does not match expected dimensions."""
    noisy_file = tmp_path / "noisy_001.npy"
    np.save(noisy_file, np.zeros((64, 64), dtype=np.float32))

    pair = DatasetPair(sample_id="001", input_path=noisy_file, target_path=None)
    validator = DatasetValidator()

    with pytest.raises(InvalidShapeError, match="Invalid spatial dimensions"):
        validator.validate_pair(pair)


def test_validate_duplicate_sample_ids(tmp_path: Path) -> None:
    """Test DatasetValidationError raised when dataset index contains duplicate sample IDs."""
    noisy_file = tmp_path / "noisy_001.npy"
    np.save(noisy_file, np.zeros((128, 128), dtype=np.float32))

    pair1 = DatasetPair(sample_id="dup_001", input_path=noisy_file, target_path=None)
    pair2 = DatasetPair(sample_id="dup_001", input_path=noisy_file, target_path=None)

    validator = DatasetValidator()
    with pytest.raises(DatasetValidationError, match="Duplicate sample ID"):
        validator.validate_dataset_index([pair1, pair2])
