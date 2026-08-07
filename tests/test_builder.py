"""Unit tests for PyTorch DataLoader builder, collate function, and worker initialization."""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.builder import (
    build_dataloader,
    build_dataloaders,
    seed_worker,
    validate_dataloader_params,
)
from src.datasets.collate import sem_collate
from src.datasets.sem_dataset import SEMDataset
from src.utils.config import Config


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary mock dataset directory structure with valid npy files."""
    dataset_dir = tmp_path / "mock_dataset"

    # Train directory structure: GT (256, 256) and NoisyLR (128, 128)
    train_gt = dataset_dir / "train" / "GT"
    train_noisy = dataset_dir / "train" / "NoisyLR"
    train_gt.mkdir(parents=True, exist_ok=True)
    train_noisy.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        sample_name = f"sample_{i:03d}.npy"
        np.save(train_gt / sample_name, np.random.rand(256, 256).astype(np.float32))
        np.save(train_noisy / sample_name, np.random.rand(128, 128).astype(np.float32))

    # Test directory structure: NoisyLR (128, 128)
    test_noisy = dataset_dir / "test" / "NoisyLR"
    test_noisy.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        sample_name = f"test_sample_{i:03d}.npy"
        np.save(test_noisy / sample_name, np.random.rand(128, 128).astype(np.float32))

    return dataset_dir


class DummyDataset(Dataset):
    """Simple dummy dataset for testing edge cases."""

    def __init__(self, length: int = 10) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input": torch.full((1, 64, 64), float(idx), dtype=torch.float32),
            "target": torch.full((1, 64, 64), float(idx * 2), dtype=torch.float32),
            "filename": f"dummy_{idx}",
        }


def test_collate_correctness() -> None:
    """Test sem_collate with complete pairs, missing targets, and extra keys."""
    sample1 = {
        "input": torch.ones(1, 32, 32),
        "target": torch.ones(1, 64, 64),
        "filename": "img_001",
    }
    sample2 = {
        "input": torch.zeros(1, 32, 32),
        "target": torch.zeros(1, 64, 64),
        "filename": "img_002",
    }

    batch = sem_collate([sample1, sample2])
    assert isinstance(batch["input"], torch.Tensor)
    assert batch["input"].shape == (2, 1, 32, 32)
    assert batch["input"].dtype == torch.float32

    assert isinstance(batch["target"], torch.Tensor)
    assert batch["target"].shape == (2, 1, 64, 64)

    assert isinstance(batch["filename"], list)
    assert batch["filename"] == ["img_001", "img_002"]


def test_collate_none_target() -> None:
    """Test sem_collate when targets are None (e.g. test set)."""
    sample1 = {
        "input": torch.ones(1, 32, 32),
        "target": None,
        "filename": "img_001",
    }
    sample2 = {
        "input": torch.zeros(1, 32, 32),
        "target": None,
        "filename": "img_002",
    }

    batch = sem_collate([sample1, sample2])
    assert batch["input"].shape == (2, 1, 32, 32)
    assert batch["target"] is None
    assert batch["filename"] == ["img_001", "img_002"]


def test_collate_empty_batch_raises() -> None:
    """Test sem_collate raises ValueError on empty batch input."""
    with pytest.raises(ValueError, match="Cannot collate an empty batch"):
        sem_collate([])


def test_build_dataloader_single_worker(mock_dataset_dir: Path) -> None:
    """Test build_dataloader with single worker (num_workers=0)."""
    dataset = SEMDataset(root_dir=mock_dataset_dir, split="train")
    loader = build_dataloader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2

    batch = next(iter(loader))
    assert batch["input"].shape == (2, 1, 128, 128)
    assert batch["target"].shape == (2, 1, 256, 256)
    assert batch["input"].dtype == torch.float32
    assert len(batch["filename"]) == 2


def test_build_dataloader_multi_worker(mock_dataset_dir: Path) -> None:
    """Test build_dataloader with multi-worker execution (num_workers=2)."""
    dataset = SEMDataset(root_dir=mock_dataset_dir, split="train")
    loader = build_dataloader(
        dataset=dataset,
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        shuffle=False,
    )
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 2

    batch = next(iter(loader))
    assert batch["input"].shape == (2, 1, 128, 128)


def test_build_dataloaders_all_splits(mock_dataset_dir: Path) -> None:
    """Test build_dataloaders creating train, val, and test loaders from Config."""
    config = Config(
        {
            "data": {
                "dataset_dir": str(mock_dataset_dir),
                "num_workers": 0,
                "train_batch_size": 2,
                "test_batch_size": 1,
                "splits": ["train", "test"],
            },
            "system": {"seed": 42},
        }
    )

    loaders = build_dataloaders(config)
    assert "train" in loaders
    assert "test" in loaders

    train_loader = loaders["train"]
    test_loader = loaders["test"]

    train_batch = next(iter(train_loader))
    assert train_batch["input"].shape[0] == 2
    assert train_batch["target"] is not None

    test_batch = next(iter(test_loader))
    assert test_batch["input"].shape[0] == 1
    assert test_batch["target"] is None


def test_deterministic_worker_init() -> None:
    """Test seed_worker function for deterministic worker seeding."""
    torch.manual_seed(1234)
    seed_worker(0)
    val1 = np.random.rand()

    torch.manual_seed(1234)
    seed_worker(0)
    val2 = np.random.rand()

    assert val1 == val2


def test_validation_empty_dataset_raises() -> None:
    """Test validation raises ValueError when dataset length is zero."""
    dataset = DummyDataset(length=0)
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataloader_params(dataset=dataset, batch_size=2, num_workers=0)


def test_validation_invalid_config_raises() -> None:
    """Test validation raises ValueError on negative batch_size or num_workers."""
    dataset = DummyDataset(length=5)

    with pytest.raises(ValueError, match="Invalid batch_size"):
        validate_dataloader_params(dataset=dataset, batch_size=0, num_workers=0)

    with pytest.raises(ValueError, match="Invalid num_workers"):
        validate_dataloader_params(dataset=dataset, batch_size=2, num_workers=-1)


def test_metadata_preservation(mock_dataset_dir: Path) -> None:
    """Test metadata and string filenames are preserved correctly in batching."""
    dataset = SEMDataset(root_dir=mock_dataset_dir, split="train")
    loader = build_dataloader(dataset=dataset, batch_size=5, num_workers=0)
    batch = next(iter(loader))

    assert "filename" in batch
    filenames = batch["filename"]
    assert isinstance(filenames, list)
    assert len(filenames) == 5
    assert all(isinstance(fn, str) for fn in filenames)
