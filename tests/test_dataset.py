"""Comprehensive unit and integration tests for dataset loading and transforms.

Verifies:
- Dataset construction and sample pairing
- .npy file loading, dtype handling, and error validation (corrupt, missing, invalid shape/dtype)
- Shape contracts for paired (NoisyLR 128x128 -> GT 256x256) and test splits
- Dynamic value clipping to [0.0, 1.0] and custom ranges
- Spatial transformations (HorizontalFlip, VerticalFlip, RandomRotate90) with asymmetric patterns
- Synchronized pair transformation consistency
- Spatial divisibility and padding contracts
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.sem_dataset import SEMDataset
from src.datasets.transforms import PairedTransforms
from src.datasets.validator import (
    DatasetValidationError,
    InvalidDtypeError,
    InvalidShapeError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_paired_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic paired dataset with 3 train pairs."""
    dataset_root = tmp_path / "sem_dataset"
    gt_dir = dataset_root / "train" / "GT"
    noisy_dir = dataset_root / "train" / "NoisyLR"
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 4):
        sample_id = f"sample_{i:03d}"
        noisy_arr = (np.random.RandomState(i).rand(128, 128) * 1.5 - 0.25).astype(
            np.float32
        )
        gt_arr = np.random.RandomState(i + 100).rand(256, 256).astype(np.float32)

        np.save(noisy_dir / f"{sample_id}.npy", noisy_arr)
        np.save(gt_dir / f"{sample_id}.npy", gt_arr)

    return dataset_root


@pytest.fixture
def synthetic_test_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic test dataset with 2 unpaired samples."""
    dataset_root = tmp_path / "sem_test_dataset"
    noisy_dir = dataset_root / "test" / "NoisyLR"
    noisy_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 3):
        sample_id = f"test_{i:03d}"
        noisy_arr = np.random.RandomState(i + 200).rand(128, 128).astype(np.float32)
        np.save(noisy_dir / f"{sample_id}.npy", noisy_arr)

    return dataset_root


# ---------------------------------------------------------------------------
# Test Group A: Dataset Construction & Sample Pairing
# ---------------------------------------------------------------------------


class TestDatasetConstruction:
    """Tests for dataset initialization and split scanning."""

    def test_construction_with_synthetic_dir(
        self, synthetic_paired_dataset_dir: Path
    ) -> None:
        """Dataset instantiates successfully with valid directory and reports correct length."""
        dataset = SEMDataset(synthetic_paired_dataset_dir, split="train")
        assert len(dataset) == 3
        assert dataset.split == "train"

    def test_paired_samples_matched_correctly(
        self, synthetic_paired_dataset_dir: Path
    ) -> None:
        """Paired NoisyLR and GT samples are matched by sample ID."""
        dataset = SEMDataset(synthetic_paired_dataset_dir, split="train")
        for i in range(len(dataset)):
            sample = dataset[i]
            expected_id = f"sample_{i + 1:03d}"
            assert sample["filename"] == expected_id
            assert isinstance(sample["input"], torch.Tensor)
            assert isinstance(sample["target"], torch.Tensor)

    def test_test_split_unpaired(self, synthetic_test_dataset_dir: Path) -> None:
        """Test split loads unpaired samples with target=None."""
        dataset = SEMDataset(synthetic_test_dataset_dir, split="test")
        assert len(dataset) == 2
        sample = dataset[0]
        assert sample["filename"] == "test_001"
        assert isinstance(sample["input"], torch.Tensor)
        assert sample["target"] is None

    def test_empty_split_raises_error(self, tmp_path: Path) -> None:
        """Scanning an empty split directory raises FileNotFoundError or DatasetValidationError."""
        empty_root = tmp_path / "empty_dataset"
        empty_root.mkdir(parents=True, exist_ok=True)
        with pytest.raises((FileNotFoundError, DatasetValidationError)):
            SEMDataset(empty_root, split="train")


# ---------------------------------------------------------------------------
# Test Group B: File Loading & Error Handling
# ---------------------------------------------------------------------------


class TestFileLoadingAndValidation:
    """Tests for .npy file loading, data types, and corruption handling."""

    def test_npy_dtype_preservation(self, synthetic_paired_dataset_dir: Path) -> None:
        """Loaded arrays are converted to torch.float32 tensors."""
        dataset = SEMDataset(synthetic_paired_dataset_dir, split="train")
        sample = dataset[0]
        inp = sample["input"]
        tgt = sample["target"]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)
        assert inp.dtype == torch.float32
        assert tgt.dtype == torch.float32

    def test_corrupted_npy_file_handling(self, tmp_path: Path) -> None:
        """Corrupted .npy file header raises DatasetValidationError during validation."""
        bad_root = tmp_path / "corrupt_data"
        gt_dir = bad_root / "train" / "GT"
        noisy_dir = bad_root / "train" / "NoisyLR"
        gt_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid binary data
        with open(noisy_dir / "bad_sample.npy", "wb") as f:
            f.write(b"NOT_A_VALID_NUMPY_FILE_CORRUPTED_DATA")
        np.save(gt_dir / "bad_sample.npy", np.zeros((256, 256), dtype=np.float32))

        with pytest.raises(DatasetValidationError, match="Failed to load array header"):
            SEMDataset(bad_root, split="train", validate=True)

    def test_invalid_dtype_handling(self, tmp_path: Path) -> None:
        """Array with non-float32 dtype raises InvalidDtypeError."""
        bad_root = tmp_path / "bad_dtype_data"
        gt_dir = bad_root / "train" / "GT"
        noisy_dir = bad_root / "train" / "NoisyLR"
        gt_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            noisy_dir / "dtype_sample.npy", np.zeros((128, 128), dtype=np.int32)
        )  # int32 instead of float32
        np.save(gt_dir / "dtype_sample.npy", np.zeros((256, 256), dtype=np.float32))

        with pytest.raises(InvalidDtypeError, match="Invalid data type"):
            SEMDataset(bad_root, split="train", validate=True)

    def test_invalid_shape_handling(self, tmp_path: Path) -> None:
        """Array with incorrect spatial dimensions raises InvalidShapeError."""
        bad_root = tmp_path / "bad_shape_data"
        gt_dir = bad_root / "train" / "GT"
        noisy_dir = bad_root / "train" / "NoisyLR"
        gt_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            noisy_dir / "shape_sample.npy", np.zeros((64, 64), dtype=np.float32)
        )  # 64x64 instead of 128x128
        np.save(gt_dir / "shape_sample.npy", np.zeros((256, 256), dtype=np.float32))

        with pytest.raises(InvalidShapeError, match="Invalid spatial dimensions"):
            SEMDataset(bad_root, split="train", validate=True)


# ---------------------------------------------------------------------------
# Test Group C: Shape Contract
# ---------------------------------------------------------------------------


class TestShapeContract:
    """Tests for tensor shapes and layout returned by dataset."""

    def test_primary_shape_contract(self, synthetic_paired_dataset_dir: Path) -> None:
        """Valid dataset item returns (1, 128, 128) input and (1, 256, 256) target."""
        dataset = SEMDataset(synthetic_paired_dataset_dir, split="train")
        sample = dataset[0]

        input_tensor = sample["input"]
        target_tensor = sample["target"]

        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)

        assert input_tensor.ndim == 3
        assert input_tensor.shape == (1, 128, 128)

        assert target_tensor.ndim == 3
        assert target_tensor.shape == (1, 256, 256)


# ---------------------------------------------------------------------------
# Test Group D: Value Handling & Clipping
# ---------------------------------------------------------------------------


class TestValueHandling:
    """Tests for pixel intensity range and dynamic clipping behavior."""

    def test_default_clipping_bounds(self, tmp_path: Path) -> None:
        """Synthetic out-of-bound values (-0.5, +1.8) are clipped strictly to [0.0, 1.0]."""
        dataset_root = tmp_path / "clip_data"
        gt_dir = dataset_root / "train" / "GT"
        noisy_dir = dataset_root / "train" / "NoisyLR"
        gt_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        raw_noisy = np.array([[-0.5, 0.5], [1.8, 0.2]], dtype=np.float32)
        raw_noisy = np.pad(raw_noisy, ((0, 126), (0, 126)), mode="constant")

        raw_gt = np.zeros((256, 256), dtype=np.float32)

        np.save(noisy_dir / "sample_clip.npy", raw_noisy)
        np.save(gt_dir / "sample_clip.npy", raw_gt)

        dataset = SEMDataset(dataset_root, split="train", clip_range=(0.0, 1.0))
        sample = dataset[0]

        inp = sample["input"]
        assert isinstance(inp, torch.Tensor)
        assert torch.min(inp).item() >= 0.0
        assert torch.max(inp).item() <= 1.0
        assert inp[0, 0, 0].item() == 0.0  # Clipped from -0.5
        assert inp[0, 0, 1].item() == 0.5  # Preserved
        assert inp[0, 1, 0].item() == 1.0  # Clipped from 1.8

    def test_none_clipping_preserves_raw_values(self, tmp_path: Path) -> None:
        """When clip_range is None, raw negative and high values are preserved."""
        dataset_root = tmp_path / "no_clip_data"
        gt_dir = dataset_root / "train" / "GT"
        noisy_dir = dataset_root / "train" / "NoisyLR"
        gt_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        raw_noisy = np.zeros((128, 128), dtype=np.float32)
        raw_noisy[0, 0] = -0.35
        raw_noisy[0, 1] = 1.65
        np.save(noisy_dir / "raw_sample.npy", raw_noisy)
        np.save(gt_dir / "raw_sample.npy", np.zeros((256, 256), dtype=np.float32))

        dataset = SEMDataset(dataset_root, split="train", clip_range=None)
        sample = dataset[0]

        inp = sample["input"]
        assert isinstance(inp, torch.Tensor)
        assert pytest.approx(inp[0, 0, 0].item(), abs=1e-5) == -0.35
        assert pytest.approx(inp[0, 0, 1].item(), abs=1e-5) == 1.65


# ---------------------------------------------------------------------------
# Test Group E & F: Spatial Transformations & Pair Consistency
# ---------------------------------------------------------------------------


class TestSpatialTransformations:
    """Tests for spatial augmentations and synchronized paired transforms."""

    def test_horizontal_flip_asymmetric(self) -> None:
        """Horizontal flip exactly inverts column order of asymmetric 2D tensor."""
        import albumentations as A

        hflip = A.HorizontalFlip(p=1.0)
        # Construct strictly asymmetric array
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        expected_flipped = np.fliplr(arr)

        res = hflip(image=arr)
        np.testing.assert_array_equal(res["image"], expected_flipped)

    def test_vertical_flip_asymmetric(self) -> None:
        """Vertical flip exactly inverts row order of asymmetric 2D tensor."""
        import albumentations as A

        vflip = A.VerticalFlip(p=1.0)
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        expected_flipped = np.flipud(arr)

        res = vflip(image=arr)
        np.testing.assert_array_equal(res["image"], expected_flipped)

    def test_random_rotate90_asymmetric(self) -> None:
        """Rotate90 produces orthogonal 90-degree rotated array."""
        import albumentations as A

        rot = A.RandomRotate90(p=1.0)
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        res = rot(image=arr)

        # Result must be one of rot90 k in [0, 1, 2, 3]
        valid_rotations = [np.rot90(arr, k=k) for k in range(4)]
        assert any(np.array_equal(res["image"], valid) for valid in valid_rotations)

    def test_paired_transforms_consistency(self) -> None:
        """PairedTransforms applies identical spatial operations to input and target."""
        # Create asymmetric input (128x128) and target (256x256) with distinctive quadrant values
        inp = torch.zeros((1, 128, 128), dtype=torch.float32)
        inp[:, :64, :64] = 1.0  # Top-left quadrant marked

        tgt = torch.zeros((1, 256, 256), dtype=torch.float32)
        tgt[:, :128, :128] = 1.0  # Top-left quadrant marked

        transforms = PairedTransforms(is_train=True)

        for seed in range(5):
            np.random.seed(seed)
            t_inp, t_tgt = transforms(inp.clone(), tgt.clone())

            assert isinstance(t_inp, torch.Tensor)
            assert isinstance(t_tgt, torch.Tensor)

            # Find the quadrant containing the 1.0 block in both
            inp_tl = (t_inp[:, :64, :64] == 1.0).all().item()
            inp_tr = (t_inp[:, :64, 64:] == 1.0).all().item()
            inp_bl = (t_inp[:, 64:, :64] == 1.0).all().item()
            inp_br = (t_inp[:, 64:, 64:] == 1.0).all().item()

            tgt_tl = (t_tgt[:, :128, :128] == 1.0).all().item()
            tgt_tr = (t_tgt[:, :128, 128:] == 1.0).all().item()
            tgt_bl = (t_tgt[:, 128:, :128] == 1.0).all().item()
            tgt_br = (t_tgt[:, 128:, 128:] == 1.0).all().item()

            # The marked quadrant must match across both input and target
            assert (
                (inp_tl and tgt_tl)
                or (inp_tr and tgt_tr)
                or (inp_bl and tgt_bl)
                or (inp_br and tgt_br)
            )

    def test_paired_transforms_eval_mode_identity(self) -> None:
        """When is_train=False, PairedTransforms returns tensors unmodified."""
        inp = torch.rand(1, 128, 128)
        tgt = torch.rand(1, 256, 256)

        eval_transforms = PairedTransforms(is_train=False)
        t_inp, t_tgt = eval_transforms(inp, tgt)

        assert isinstance(t_inp, torch.Tensor)
        assert isinstance(t_tgt, torch.Tensor)
        assert torch.equal(inp, t_inp)
        assert torch.equal(tgt, t_tgt)


# ---------------------------------------------------------------------------
# Test Group G: Divisibility and Padding Contract
# ---------------------------------------------------------------------------


class TestDivisibilityAndPadding:
    """Tests for spatial dimension divisibility and padding contracts."""

    def test_divisible_by_8_dimensions(self) -> None:
        """Standard input dimensions (128x128) are already divisible by 8 (2^3 downsample levels)."""
        h, w = 128, 128
        assert h % 8 == 0
        assert w % 8 == 0

    def test_padding_contract_in_model_forward(self) -> None:
        """NAFNet model pads non-divisible dimensions and unpads output back to exact scaled size."""
        from src.models.nafnet import NAFNet

        model = NAFNet(
            img_channel=1,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
            upscale=2,
        )
        model.eval()

        # Non-divisible input dimensions: (1, 1, 100, 130) -> upscale=2 -> (1, 1, 200, 260)
        x_non_div = torch.randn(1, 1, 100, 130)
        with torch.no_grad():
            out = model(x_non_div)

        assert out.shape == (1, 1, 200, 260)
