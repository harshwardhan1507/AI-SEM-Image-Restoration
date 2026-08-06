"""Pytest unit test suite for paired data augmentations (src/datasets/transforms.py)."""

from pathlib import Path

import numpy as np
import torch

from src.datasets.sem_dataset import SEMDataset
from src.datasets.transforms import get_transforms


def test_paired_transforms_eval_mode() -> None:
    """Test eval mode returns identity pass-through without modifications."""
    transform = get_transforms(is_train=False)
    input_arr = np.zeros((128, 128), dtype=np.float32)
    target_arr = np.ones((256, 256), dtype=np.float32)

    trans_in, trans_tgt = transform(input_arr, target_arr)
    assert np.array_equal(trans_in, input_arr)
    assert np.array_equal(trans_tgt, target_arr)


def test_paired_transforms_train_mode_tensor() -> None:
    """Test train mode maintains tensor channel shapes (1, H, W)."""
    transform = get_transforms(is_train=True)
    input_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)
    target_tensor = torch.ones((1, 256, 256), dtype=torch.float32)

    trans_in, trans_tgt = transform(input_tensor, target_tensor)

    assert isinstance(trans_in, torch.Tensor)
    assert isinstance(trans_tgt, torch.Tensor)
    assert trans_in.shape == (1, 128, 128)
    assert trans_tgt.shape == (1, 256, 256)


def test_sem_dataset_with_transforms(tmp_path: Path) -> None:
    """Test SEMDataset integration with paired spatial transforms."""
    dataset_root = tmp_path / "sem_data"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    np.save(noisy_dir / "aug_001.npy", np.zeros((128, 128), dtype=np.float32))
    np.save(gt_dir / "aug_001.npy", np.ones((256, 256), dtype=np.float32))

    dataset = SEMDataset(
        dataset_root,
        split="train",
        transform=get_transforms(is_train=True),
    )
    sample = dataset[0]

    assert sample["input"].shape == (1, 128, 128)
    assert sample["target"].shape == (1, 256, 256)
