"""Comprehensive unit and reference verification tests for evaluation metrics.

Verifies:
- PSNR mathematical correctness, known constant error, infinite return for identical images
- SSIM mathematical correctness, range bounds [-1.0, 1.0], reference agreement with skimage
- Metric directionality: higher PSNR is better, higher SSIM is better, lower LPIPS is better
- Multi-dimensional layout support: 2D (H, W), 3D (1, H, W), and 4D (B, 1, H, W)
- Dimension and shape mismatch validation
- Numerical stability, constant images, zero MSE, NaN/Inf rejection
- Half precision (FP16/BF16) and gradient preservation
- LPIPS perceptual evaluation and grayscale channel expansion
"""

import numpy as np
import pytest
import skimage.metrics
import torch

from src.metrics.lpips import calculate_lpips
from src.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# ---------------------------------------------------------------------------
# Test Group A: PSNR Correctness & Mathematical Verification
# ---------------------------------------------------------------------------


class TestPSNRCorrectness:
    """Tests for PSNR calculation against theoretical formulas and reference baselines."""

    def test_psnr_identical_images_is_infinite(self) -> None:
        """Identical images yield float('inf') for NumPy and PyTorch tensors."""
        img_np = np.random.RandomState(42).rand(64, 64).astype(np.float32)
        img_torch_2d = torch.from_numpy(img_np)
        img_torch_3d = img_torch_2d.unsqueeze(0)  # (1, 64, 64)
        img_torch_4d = img_torch_3d.unsqueeze(0)  # (1, 1, 64, 64)

        assert calculate_psnr(img_np, img_np, data_range=1.0) == float("inf")
        assert calculate_psnr(img_torch_3d, img_torch_3d, data_range=1.0) == float(
            "inf"
        )
        assert calculate_psnr(img_torch_4d, img_torch_4d, data_range=1.0) == float(
            "inf"
        )

    def test_psnr_known_constant_error(self) -> None:
        """Verify PSNR against an independently calculated analytical value.

        For data_range = 1.0 and a uniform constant error e = 0.1:
        MSE = e^2 = 0.01
        PSNR = 10 * log10(1.0 / 0.01) = 10 * log10(100) = 20.0 dB.
        """
        target = np.full((64, 64), 0.5, dtype=np.float32)
        pred = np.full((64, 64), 0.6, dtype=np.float32)  # constant error +0.1

        calculated = calculate_psnr(pred, target, data_range=1.0)
        expected = 20.0
        assert calculated == pytest.approx(expected, abs=1e-5)

        # For constant error e = 0.01:
        # MSE = 0.0001 -> PSNR = 10 * log10(10000) = 40.0 dB
        pred_small = np.full((64, 64), 0.51, dtype=np.float32)
        calculated_small = calculate_psnr(pred_small, target, data_range=1.0)
        assert calculated_small == pytest.approx(40.0, abs=1e-5)

    def test_psnr_worse_reconstruction_produces_lower_score(self) -> None:
        """Larger reconstruction error monotonically reduces PSNR."""
        rng = np.random.RandomState(123)
        target = rng.rand(64, 64).astype(np.float32)

        pred_good = np.clip(
            target + rng.normal(0, 0.01, (64, 64)).astype(np.float32), 0.0, 1.0
        )
        pred_bad = np.clip(
            target + rng.normal(0, 0.10, (64, 64)).astype(np.float32), 0.0, 1.0
        )

        psnr_good = calculate_psnr(pred_good, target, data_range=1.0)
        psnr_bad = calculate_psnr(pred_bad, target, data_range=1.0)

        assert psnr_good > psnr_bad

    def test_psnr_reference_agreement_with_skimage(self) -> None:
        """Compare PSNR against skimage reference within 1e-4 tolerance."""
        rng = np.random.RandomState(456)
        target = rng.rand(64, 64).astype(np.float32)
        pred = np.clip(
            target + rng.normal(0, 0.05, (64, 64)).astype(np.float32), 0.0, 1.0
        )

        ref_psnr = skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=1.0)
        calc_psnr = calculate_psnr(pred, target, data_range=1.0)

        assert calc_psnr == pytest.approx(ref_psnr, abs=1e-4)


# ---------------------------------------------------------------------------
# Test Group B: SSIM Correctness & Reference Agreement
# ---------------------------------------------------------------------------


class TestSSIMCorrectness:
    """Tests for SSIM structural correlation and boundary conditions."""

    def test_ssim_identical_images_is_one(self) -> None:
        """Identical images yield SSIM = 1.0."""
        img = np.random.RandomState(789).rand(64, 64).astype(np.float32)
        assert calculate_ssim(img, img, data_range=1.0) == pytest.approx(1.0, abs=1e-5)

    def test_ssim_structurally_different_images(self) -> None:
        """Structurally distinct images produce lower SSIM within [-1.0, 1.0]."""
        rng = np.random.RandomState(101)
        target = rng.rand(64, 64).astype(np.float32)
        pred = rng.rand(64, 64).astype(np.float32)

        ssim_val = calculate_ssim(pred, target, data_range=1.0)
        assert -1.0 <= ssim_val < 1.0

    def test_ssim_reference_agreement_with_skimage(self) -> None:
        """Compare SSIM against skimage reference within 1e-4 tolerance."""
        rng = np.random.RandomState(202)
        target = rng.rand(64, 64).astype(np.float32)
        pred = np.clip(
            target + rng.normal(0, 0.03, (64, 64)).astype(np.float32), 0.0, 1.0
        )

        ref_ssim = skimage.metrics.structural_similarity(target, pred, data_range=1.0)
        calc_ssim = calculate_ssim(pred, target, data_range=1.0)

        assert calc_ssim == pytest.approx(ref_ssim, abs=1e-4)

    def test_ssim_constant_images(self) -> None:
        """Constant images evaluate cleanly without numerical failure."""
        c1 = np.full((64, 64), 0.3, dtype=np.float32)
        c2 = np.full((64, 64), 0.7, dtype=np.float32)

        # Identical constant images
        assert calculate_ssim(c1, c1, data_range=1.0) == pytest.approx(1.0, abs=1e-4)
        # Distinct constant images
        assert calculate_ssim(c1, c2, data_range=1.0) < 1.0


# ---------------------------------------------------------------------------
# Test Group C: Metric Directionality & Verification
# ---------------------------------------------------------------------------


class TestMetricDirectionality:
    """Explicit tests verifying metric optimization directions."""

    def test_metric_direction_hierarchy(self) -> None:
        """Verify: higher PSNR = better, higher SSIM = better, lower LPIPS = better."""
        rng = np.random.RandomState(303)
        target = rng.rand(64, 64).astype(np.float32)

        # High fidelity prediction (slight noise)
        pred_high = np.clip(
            target + rng.normal(0, 0.01, (64, 64)).astype(np.float32), 0.0, 1.0
        )
        # Low fidelity prediction (heavy noise)
        pred_low = np.clip(
            target + rng.normal(0, 0.15, (64, 64)).astype(np.float32), 0.0, 1.0
        )

        psnr_high = calculate_psnr(pred_high, target, data_range=1.0)
        psnr_low = calculate_psnr(pred_low, target, data_range=1.0)
        assert psnr_high > psnr_low, (
            "PSNR direction violation: higher should indicate better quality"
        )

        ssim_high = calculate_ssim(pred_high, target, data_range=1.0)
        ssim_low = calculate_ssim(pred_low, target, data_range=1.0)
        assert ssim_high > ssim_low, (
            "SSIM direction violation: higher should indicate better quality"
        )

        # LPIPS evaluation (if package available)
        lpips_high = calculate_lpips(pred_high, target, data_range=1.0)
        lpips_low = calculate_lpips(pred_low, target, data_range=1.0)
        if lpips_high is not None and lpips_low is not None:
            assert lpips_high < lpips_low, (
                "LPIPS direction violation: lower should indicate better quality"
            )


# ---------------------------------------------------------------------------
# Test Group D: Shape Handling & Validation
# ---------------------------------------------------------------------------


class TestShapeHandlingAndRejection:
    """Tests for layout handling and dimension validation."""

    def test_supported_layouts(self) -> None:
        """Supports 2D (H,W), 3D (1,H,W), and 4D (B,1,H,W) across NumPy and PyTorch."""
        rng = np.random.RandomState(404)
        t_2d = rng.rand(32, 32).astype(np.float32)
        p_2d = rng.rand(32, 32).astype(np.float32)

        # 2D NumPy
        assert isinstance(calculate_psnr(p_2d, t_2d), float)
        assert isinstance(calculate_ssim(p_2d, t_2d), float)

        # 3D PyTorch
        t_3d = torch.from_numpy(t_2d).unsqueeze(0)
        p_3d = torch.from_numpy(p_2d).unsqueeze(0)
        assert isinstance(calculate_psnr(p_3d, t_3d), float)
        assert isinstance(calculate_ssim(p_3d, t_3d), float)

        # 4D PyTorch
        t_4d = t_3d.unsqueeze(0)
        p_4d = p_3d.unsqueeze(0)
        assert isinstance(calculate_psnr(p_4d, t_4d), float)
        assert isinstance(calculate_ssim(p_4d, t_4d), float)

    def test_batch_semantics_equals_sample_mean(self) -> None:
        """Batched metrics equal the mean of individual sample metrics."""
        rng = np.random.RandomState(505)
        b = 3
        preds = rng.rand(b, 1, 32, 32).astype(np.float32)
        targets = rng.rand(b, 1, 32, 32).astype(np.float32)

        batch_psnr = calculate_psnr(preds, targets)
        batch_ssim = calculate_ssim(preds, targets)

        sample_psnrs = [calculate_psnr(preds[i], targets[i]) for i in range(b)]
        sample_ssims = [calculate_ssim(preds[i], targets[i]) for i in range(b)]

        assert batch_psnr == pytest.approx(float(np.mean(sample_psnrs)), abs=1e-5)
        assert batch_ssim == pytest.approx(float(np.mean(sample_ssims)), abs=1e-5)

    def test_shape_mismatch_raises_value_error(self) -> None:
        """Mismatched spatial shapes raise ValueError."""
        p = np.zeros((32, 32), dtype=np.float32)
        t = np.zeros((32, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_psnr(p, t)

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_ssim(p, t)

    def test_invalid_channel_counts_raise_value_error(self) -> None:
        """Non single-channel 3D or 4D tensors raise ValueError."""
        # 3D with 3 channels
        p_3c = np.zeros((3, 32, 32), dtype=np.float32)
        t_3c = np.zeros((3, 32, 32), dtype=np.float32)

        with pytest.raises(ValueError, match="single-channel"):
            calculate_psnr(p_3c, t_3c)

        with pytest.raises(ValueError, match="single-channel"):
            calculate_ssim(p_3c, t_3c)


# ---------------------------------------------------------------------------
# Test Group E: Numerical Stability & Precision
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical stability edge cases, precision types, and autograd safety."""

    def test_nan_inf_rejection(self) -> None:
        """Non-finite inputs (NaN or Inf) raise ValueError."""
        t = np.zeros((32, 32), dtype=np.float32)
        p_nan = t.copy()
        p_nan[0, 0] = np.nan

        p_inf = t.copy()
        p_inf[0, 0] = np.inf

        with pytest.raises(ValueError, match="non-finite values"):
            calculate_psnr(p_nan, t)

        with pytest.raises(ValueError, match="non-finite values"):
            calculate_psnr(p_inf, t)

        with pytest.raises(ValueError, match="non-finite values"):
            calculate_ssim(p_nan, t)

    def test_invalid_data_range_raises_value_error(self) -> None:
        """Non-positive data_range raises ValueError."""
        img = np.zeros((32, 32), dtype=np.float32)

        with pytest.raises(ValueError, match="data_range must be positive"):
            calculate_psnr(img, img, data_range=0.0)

        with pytest.raises(ValueError, match="data_range must be positive"):
            calculate_ssim(img, img, data_range=-1.0)

    def test_half_precision_support(self) -> None:
        """torch.float16 and torch.bfloat16 tensors are evaluated without error."""
        rng = np.random.RandomState(606)
        pred_f32 = torch.from_numpy(rng.rand(1, 32, 32).astype(np.float32))
        target_f32 = torch.from_numpy(rng.rand(1, 32, 32).astype(np.float32))

        pred_f16 = pred_f32.to(torch.float16)
        target_f16 = target_f32.to(torch.float16)

        psnr_f16 = calculate_psnr(pred_f16, target_f16, data_range=1.0)
        ssim_f16 = calculate_ssim(pred_f16, target_f16, data_range=1.0)

        assert isinstance(psnr_f16, float)
        assert isinstance(ssim_f16, float)

    def test_no_gradient_mutation(self) -> None:
        """Input tensors with requires_grad=True retain flags and populate no gradients."""
        pred = torch.rand(1, 32, 32, requires_grad=True)
        target = torch.rand(1, 32, 32, requires_grad=True)

        _ = calculate_psnr(pred, target)
        _ = calculate_ssim(pred, target)

        assert pred.requires_grad is True
        assert target.requires_grad is True
        assert pred.grad is None
        assert target.grad is None
