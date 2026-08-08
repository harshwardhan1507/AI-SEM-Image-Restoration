"""Unit tests for PSNR and SSIM metrics module."""

import numpy as np
import pytest
import skimage.metrics
import torch

from src.metrics import calculate_psnr, calculate_ssim


def test_identical_images() -> None:
    """Test 1: Identical images yield PSNR=inf and SSIM=1.0."""
    img_np = np.random.RandomState(42).rand(64, 64).astype(np.float32)
    img_tensor = torch.from_numpy(img_np)

    # NumPy 2D
    assert calculate_psnr(img_np, img_np, data_range=1.0) == float("inf")
    assert calculate_ssim(img_np, img_np, data_range=1.0) == pytest.approx(1.0)

    # PyTorch 3D (1, H, W)
    img_3d = img_tensor.unsqueeze(0)
    assert calculate_psnr(img_3d, img_3d, data_range=1.0) == float("inf")
    assert calculate_ssim(img_3d, img_3d, data_range=1.0) == pytest.approx(1.0)


def test_known_images_reference() -> None:
    """Test 2: Compare outputs against skimage reference functions within 1e-4 tolerance."""
    rng = np.random.RandomState(123)
    target = rng.rand(64, 64).astype(np.float32)
    prediction = target + rng.normal(0, 0.05, (64, 64)).astype(np.float32)
    prediction = np.clip(prediction, 0.0, 1.0)

    ref_psnr = skimage.metrics.peak_signal_noise_ratio(
        target, prediction, data_range=1.0
    )
    ref_ssim = skimage.metrics.structural_similarity(target, prediction, data_range=1.0)

    calc_psnr = calculate_psnr(prediction, target, data_range=1.0)
    calc_ssim = calculate_ssim(prediction, target, data_range=1.0)

    assert calc_psnr == pytest.approx(ref_psnr, abs=1e-4)
    assert calc_ssim == pytest.approx(ref_ssim, abs=1e-4)


def test_numpy_2d() -> None:
    """Test 3: 2D (H, W) NumPy inputs produce scalar metrics."""
    rng = np.random.RandomState(456)
    target = rng.rand(128, 128).astype(np.float32)
    pred = rng.rand(128, 128).astype(np.float32)

    psnr = calculate_psnr(pred, target, data_range=1.0)
    ssim = calculate_ssim(pred, target, data_range=1.0)

    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
    assert 0.0 <= ssim <= 1.0


def test_pytorch_3d() -> None:
    """Test 4: 3D (1, H, W) PyTorch Tensor inputs work cleanly."""
    rng = np.random.RandomState(789)
    target = torch.from_numpy(rng.rand(1, 128, 128).astype(np.float32))
    pred = torch.from_numpy(rng.rand(1, 128, 128).astype(np.float32))

    psnr = calculate_psnr(pred, target, data_range=1.0)
    ssim = calculate_ssim(pred, target, data_range=1.0)

    assert isinstance(psnr, float)
    assert isinstance(ssim, float)


def test_batch_semantics() -> None:
    """Test 5: Batch metric equals the average of individual sample metrics."""
    rng = np.random.RandomState(101)
    batch_size = 4
    pred_batch = rng.rand(batch_size, 1, 64, 64).astype(np.float32)
    target_batch = rng.rand(batch_size, 1, 64, 64).astype(np.float32)

    batch_psnr = calculate_psnr(pred_batch, target_batch, data_range=1.0)
    batch_ssim = calculate_ssim(pred_batch, target_batch, data_range=1.0)

    sample_psnrs = [
        calculate_psnr(pred_batch[i], target_batch[i], data_range=1.0)
        for i in range(batch_size)
    ]
    sample_ssims = [
        calculate_ssim(pred_batch[i], target_batch[i], data_range=1.0)
        for i in range(batch_size)
    ]

    expected_psnr = float(np.mean(sample_psnrs))
    expected_ssim = float(np.mean(sample_ssims))

    assert batch_psnr == pytest.approx(expected_psnr, abs=1e-5)
    assert batch_ssim == pytest.approx(expected_ssim, abs=1e-5)


def test_shape_mismatch() -> None:
    """Test 6: Mismatched prediction and target shapes raise ValueError."""
    pred = np.zeros((64, 64), dtype=np.float32)
    target = np.zeros((64, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_psnr(pred, target)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_ssim(pred, target)


def test_invalid_data_range() -> None:
    """Test 7: data_range <= 0 raises ValueError."""
    img = np.zeros((64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="data_range must be positive"):
        calculate_psnr(img, img, data_range=0.0)

    with pytest.raises(ValueError, match="data_range must be positive"):
        calculate_ssim(img, img, data_range=-1.0)


def test_nan_inf_rejection() -> None:
    """Test 8: NaN or Inf values in input raise ValueError."""
    pred_nan = np.zeros((64, 64), dtype=np.float32)
    pred_nan[0, 0] = np.nan
    target = np.zeros((64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite values"):
        calculate_psnr(pred_nan, target)

    with pytest.raises(ValueError, match="non-finite values"):
        calculate_ssim(pred_nan, target)

    pred_inf = np.zeros((64, 64), dtype=np.float32)
    pred_inf[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite values"):
        calculate_psnr(pred_inf, target)


def test_half_precision_tensors() -> None:
    """Test 9: torch.float16 and torch.bfloat16 tensors are supported."""
    rng = np.random.RandomState(202)
    pred_np = rng.rand(1, 64, 64).astype(np.float32)
    target_np = rng.rand(1, 64, 64).astype(np.float32)

    pred_f16 = torch.from_numpy(pred_np).to(torch.float16)
    target_f16 = torch.from_numpy(target_np).to(torch.float16)

    psnr_f16 = calculate_psnr(pred_f16, target_f16, data_range=1.0)
    ssim_f16 = calculate_ssim(pred_f16, target_f16, data_range=1.0)

    ref_psnr = calculate_psnr(pred_np, target_np, data_range=1.0)
    ref_ssim = calculate_ssim(pred_np, target_np, data_range=1.0)

    assert psnr_f16 == pytest.approx(ref_psnr, abs=1e-2)
    assert ssim_f16 == pytest.approx(ref_ssim, abs=1e-2)

    pred_bf16 = torch.from_numpy(pred_np).to(torch.bfloat16)
    target_bf16 = torch.from_numpy(target_np).to(torch.bfloat16)

    psnr_bf16 = calculate_psnr(pred_bf16, target_bf16, data_range=1.0)
    ssim_bf16 = calculate_ssim(pred_bf16, target_bf16, data_range=1.0)

    assert isinstance(psnr_bf16, float)
    assert isinstance(ssim_bf16, float)


def test_no_gradient_mutation() -> None:
    """Test 10: Input tensor with requires_grad=True is unchanged, retains flag, and populates no gradients."""
    pred_tensor = torch.rand(1, 64, 64, requires_grad=True)
    target_tensor = torch.rand(1, 64, 64, requires_grad=True)

    pred_clone = pred_tensor.clone().detach()

    psnr_val = calculate_psnr(pred_tensor, target_tensor, data_range=1.0)
    ssim_val = calculate_ssim(pred_tensor, target_tensor, data_range=1.0)

    assert pred_tensor.requires_grad is True
    assert target_tensor.requires_grad is True
    assert pred_tensor.grad is None
    assert target_tensor.grad is None
    assert torch.equal(pred_tensor.detach(), pred_clone)
    assert isinstance(psnr_val, float)
    assert isinstance(ssim_val, float)
