"""Pytest unit test suite for SEMDataset (src/datasets/sem_dataset.py)."""

from pathlib import Path

import numpy as np
import torch

from src.datasets.sem_dataset import SEMDataset


def test_sem_dataset_len_and_shapes(tmp_path: Path) -> None:
    """Test SEMDataset len and sample dictionary shapes."""
    dataset_root = tmp_path / "sem_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    np.save(noisy_dir / "sample_001.npy", np.zeros((128, 128), dtype=np.float32))
    np.save(gt_dir / "sample_001.npy", np.ones((256, 256), dtype=np.float32))

    dataset = SEMDataset(dataset_root, split="train")
    assert len(dataset) == 1

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "input" in sample
    assert "target" in sample
    assert "filename" in sample

    assert sample["filename"] == "sample_001"
    assert isinstance(sample["input"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)

    assert sample["input"].shape == (1, 128, 128)
    assert sample["target"].shape == (1, 256, 256)
    assert sample["input"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32


def test_sem_dataset_intensity_clipping(tmp_path: Path) -> None:
    """Test out-of-bounds array intensities are clipped to [0.0, 1.0]."""
    dataset_root = tmp_path / "sem_data"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    noisy_arr = np.zeros((128, 128), dtype=np.float32)
    noisy_arr[0, 0] = -0.5
    noisy_arr[0, 1] = 1.5

    np.save(noisy_dir / "clip_001.npy", noisy_arr)
    np.save(gt_dir / "clip_001.npy", np.zeros((256, 256), dtype=np.float32))

    dataset = SEMDataset(dataset_root, split="train", clip_range=(0.0, 1.0))
    sample = dataset[0]

    input_tensor = sample["input"]
    assert torch.min(input_tensor).item() >= 0.0
    assert torch.max(input_tensor).item() <= 1.0


def test_sem_dataset_test_split(tmp_path: Path) -> None:
    """Test SEMDataset on test split returns None for target tensor."""
    dataset_root = tmp_path / "sem_data"
    test_noisy = dataset_root / "test" / "NoisyLR"
    test_noisy.mkdir(parents=True)

    np.save(test_noisy / "test_001.npy", np.zeros((128, 128), dtype=np.float32))

    dataset = SEMDataset(dataset_root, split="test")
    sample = dataset[0]

    assert sample["filename"] == "test_001"
    assert sample["input"].shape == (1, 128, 128)
    assert sample["target"] is None
