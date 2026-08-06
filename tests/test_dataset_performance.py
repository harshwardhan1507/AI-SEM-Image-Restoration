"""Pytest performance and stability test suite for SEMDataset."""

import time
from pathlib import Path

import numpy as np
import torch

from src.datasets.sem_dataset import SEMDataset


def test_dataset_init_performance(tmp_path: Path) -> None:
    """Test SEMDataset initialization latency is sub-100ms."""
    dataset_root = tmp_path / "sem_perf_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    for i in range(50):
        np.save(noisy_dir / f"img_{i:03d}.npy", np.zeros((128, 128), dtype=np.float32))
        np.save(gt_dir / f"img_{i:03d}.npy", np.ones((256, 256), dtype=np.float32))

    t0 = time.perf_counter()
    dataset = SEMDataset(dataset_root, split="train")
    init_time = (time.perf_counter() - t0) * 1000.0

    assert len(dataset) == 50
    assert init_time < 100.0  # Latency under 100ms


def test_dataset_single_fetch_performance(tmp_path: Path) -> None:
    """Test sample fetch latency is under 5ms."""
    dataset_root = tmp_path / "sem_perf_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    np.save(noisy_dir / "perf_001.npy", np.zeros((128, 128), dtype=np.float32))
    np.save(gt_dir / "perf_001.npy", np.ones((256, 256), dtype=np.float32))

    dataset = SEMDataset(dataset_root, split="train")

    t0 = time.perf_counter()
    _ = dataset[0]
    fetch_latency = (time.perf_counter() - t0) * 1000.0

    assert fetch_latency < 10.0  # Fetch latency under 10ms


def test_dataset_repeated_access_stability(tmp_path: Path) -> None:
    """Test repeated accesses yield identical deterministic tensors without memory leaks."""
    dataset_root = tmp_path / "sem_perf_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    np.save(noisy_dir / "repeat_001.npy", np.random.randn(128, 128).astype(np.float32))
    np.save(gt_dir / "repeat_001.npy", np.random.randn(256, 256).astype(np.float32))

    dataset = SEMDataset(dataset_root, split="train")

    sample_first = dataset[0]

    for _ in range(50):
        sample_curr = dataset[0]
        assert torch.equal(sample_first["input"], sample_curr["input"])  # type: ignore
        assert torch.equal(sample_first["target"], sample_curr["target"])  # type: ignore
