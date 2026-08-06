"""Pytest unit test suite for dataset scanner (src/datasets/scanner.py)."""

from pathlib import Path

import pytest

from src.datasets.scanner import DatasetPair, DatasetScanner


def test_scanner_invalid_root() -> None:
    """Test FileNotFoundError raised when root directory does not exist."""
    with pytest.raises(FileNotFoundError):
        DatasetScanner("non_existent_dataset_root_999")


def test_scanner_filters_macosx_and_hidden(tmp_path: Path) -> None:
    """Test hidden files starting with '._' or inside '__MACOSX' are ignored."""
    dataset_root = tmp_path / "dataset"
    train_gt = dataset_root / "train" / "GT"
    train_noisy = dataset_root / "train" / "NoisyLR"
    train_gt.mkdir(parents=True)
    train_noisy.mkdir(parents=True)

    # Valid files
    (train_gt / "img_001.npy").touch()
    (train_noisy / "img_001.npy").touch()

    # Hidden / macOS files
    (train_gt / "._img_001.npy").touch()
    (train_noisy / ".DS_Store").touch()

    macosx_dir = train_gt / "__MACOSX"
    macosx_dir.mkdir()
    (macosx_dir / "img_001.npy").touch()

    scanner = DatasetScanner(dataset_root)
    pairs = scanner.scan_split("train")

    assert len(pairs) == 1
    assert pairs[0].sample_id == "img_001"
    assert pairs[0].input_path.name == "img_001.npy"
    assert pairs[0].target_path is not None
    assert pairs[0].target_path.name == "img_001.npy"


def test_scanner_scan_train_mismatch(tmp_path: Path) -> None:
    """Test ValueError raised when GT and NoisyLR file counts do not match."""
    dataset_root = tmp_path / "dataset"
    train_gt = dataset_root / "train" / "GT"
    train_noisy = dataset_root / "train" / "NoisyLR"
    train_gt.mkdir(parents=True)
    train_noisy.mkdir(parents=True)

    (train_gt / "img_001.npy").touch()
    (train_gt / "img_002.npy").touch()
    (train_noisy / "img_001.npy").touch()

    scanner = DatasetScanner(dataset_root)
    with pytest.raises(ValueError, match="Mismatched dataset counts"):
        scanner.scan_split("train")


def test_scanner_scan_test_split(tmp_path: Path) -> None:
    """Test scanning test split returns DatasetPair instances with target_path=None."""
    dataset_root = tmp_path / "dataset"
    test_noisy = dataset_root / "test" / "NoisyLR"
    test_noisy.mkdir(parents=True)

    (test_noisy / "test_001.npy").touch()
    (test_noisy / "test_002.npy").touch()

    scanner = DatasetScanner(dataset_root)
    pairs = scanner.scan_split("test")

    assert len(pairs) == 2
    assert isinstance(pairs[0], DatasetPair)
    assert pairs[0].target_path is None
    assert pairs[1].target_path is None
