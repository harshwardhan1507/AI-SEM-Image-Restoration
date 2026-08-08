"""Pytest unit test suite for differentiable restoration losses (src/losses/)."""

import math

import pytest
import torch

from src.losses import CharbonnierLoss, PSNRLoss, build_loss, build_loss_function
from src.utils.config import ConfigDict

# --- Charbonnier Loss Tests ---


def test_charbonnier_construction() -> None:
    """Test CharbonnierLoss initialization defaults and invalid parameters."""
    criterion = CharbonnierLoss(eps=1e-3, reduction="mean")
    assert criterion.eps == 1e-3
    assert criterion.reduction == "mean"

    with pytest.raises(ValueError, match="eps must be a positive float"):
        CharbonnierLoss(eps=0.0)

    with pytest.raises(ValueError, match="Unsupported reduction"):
        CharbonnierLoss(reduction="invalid_reduction")


def test_charbonnier_known_values() -> None:
    """Test CharbonnierLoss output against hand-calculated mathematical values."""
    eps = 1e-3
    criterion = CharbonnierLoss(eps=eps, reduction="mean")

    pred = torch.tensor([[0.5, 0.8], [0.2, 0.9]], dtype=torch.float32)
    target = torch.tensor([[0.5, 0.6], [0.2, 0.4]], dtype=torch.float32)

    diff = pred - target  # [[0.0, 0.2], [0.0, 0.5]]
    elem_loss = torch.sqrt(diff * diff + eps * eps)
    expected_mean = torch.mean(elem_loss).item()

    output = criterion(pred, target)
    assert output.ndim == 0  # scalar
    assert output.item() == pytest.approx(expected_mean, rel=1e-5)


def test_charbonnier_zero_residual() -> None:
    """Test CharbonnierLoss when prediction equals target."""
    eps = 1e-3
    criterion = CharbonnierLoss(eps=eps, reduction="mean")

    img = torch.rand(2, 1, 64, 64)
    loss = criterion(img, img)
    # sqrt(0^2 + eps^2) = eps
    assert loss.item() == pytest.approx(eps, rel=1e-5)


def test_charbonnier_reductions() -> None:
    """Test CharbonnierLoss mean, sum, and none reduction semantics."""
    pred = torch.rand(4, 1, 32, 32)
    target = torch.rand(4, 1, 32, 32)

    loss_none = CharbonnierLoss(reduction="none")(pred, target)
    loss_mean = CharbonnierLoss(reduction="mean")(pred, target)
    loss_sum = CharbonnierLoss(reduction="sum")(pred, target)

    assert loss_none.shape == (4, 1, 32, 32)
    assert loss_mean.item() == pytest.approx(torch.mean(loss_none).item(), rel=1e-5)
    assert loss_sum.item() == pytest.approx(torch.sum(loss_none).item(), rel=1e-5)


def test_charbonnier_autograd_and_non_mutation() -> None:
    """Test CharbonnierLoss backward pass, gradient finiteness, and non-mutation."""
    criterion = CharbonnierLoss()
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64, requires_grad=False)

    pred_clone = pred.clone().detach()

    loss = criterion(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.requires_grad is True
    assert torch.equal(pred.detach(), pred_clone)


def test_charbonnier_shape_mismatch() -> None:
    """Test CharbonnierLoss raises ValueError on shape mismatch."""
    criterion = CharbonnierLoss()
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 32, 32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        criterion(pred, target)


# --- PSNR Loss Tests ---


def test_psnr_loss_construction() -> None:
    """Test PSNRLoss initialization defaults and invalid parameters."""
    criterion = PSNRLoss(data_range=1.0, eps=1e-8, reduction="mean")
    assert criterion.data_range == 1.0
    assert criterion.eps == 1e-8
    assert criterion.reduction == "mean"

    with pytest.raises(ValueError, match="data_range must be a positive float"):
        PSNRLoss(data_range=0.0)

    with pytest.raises(ValueError, match="eps must be a positive float"):
        PSNRLoss(eps=0.0)

    with pytest.raises(ValueError, match="Unsupported reduction"):
        PSNRLoss(reduction="invalid")


def test_psnr_loss_known_values() -> None:
    """Test PSNRLoss output against independently calculated formula."""
    data_range = 1.0
    eps = 1e-8
    criterion = PSNRLoss(data_range=data_range, eps=eps, reduction="mean")

    pred = torch.ones(1, 1, 4, 4, dtype=torch.float32) * 0.8
    target = torch.ones(1, 1, 4, 4, dtype=torch.float32) * 0.6

    mse = (0.8 - 0.6) ** 2  # 0.04
    expected_psnr_loss = (10.0 / math.log(10.0)) * math.log(
        (mse + eps) / (data_range**2)
    )

    loss = criterion(pred, target)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(expected_psnr_loss, rel=1e-4)


def test_psnr_loss_batch_semantics() -> None:
    """Test PSNRLoss per-sample batch semantics and reduction modes."""
    batch_size = 4
    pred = torch.rand(batch_size, 1, 32, 32)
    target = torch.rand(batch_size, 1, 32, 32)

    criterion_none = PSNRLoss(reduction="none")
    criterion_mean = PSNRLoss(reduction="mean")
    criterion_sum = PSNRLoss(reduction="sum")

    loss_none = criterion_none(pred, target)
    loss_mean = criterion_mean(pred, target)
    loss_sum = criterion_sum(pred, target)

    # reduction='none' should return per-sample loss tensor of shape (B,)
    assert loss_none.shape == (batch_size,)

    # Per-sample loss check
    sample_losses = [
        PSNRLoss(reduction="mean")(pred[i : i + 1], target[i : i + 1]).item()
        for i in range(batch_size)
    ]
    for i in range(batch_size):
        assert loss_none[i].item() == pytest.approx(sample_losses[i], rel=1e-4)

    assert loss_mean.item() == pytest.approx(torch.mean(loss_none).item(), rel=1e-5)
    assert loss_sum.item() == pytest.approx(torch.sum(loss_none).item(), rel=1e-5)


def test_psnr_loss_zero_mse_stability() -> None:
    """Test PSNRLoss behavior when prediction equals target (zero MSE)."""
    data_range = 1.0
    eps = 1e-8
    criterion = PSNRLoss(data_range=data_range, eps=eps, reduction="mean")

    img = torch.rand(2, 1, 64, 64, requires_grad=True)
    loss = criterion(img, img)

    # With MSE=0, loss = (10/ln(10)) * ln(1e-8) = -80.0 dB
    expected_loss = (10.0 / math.log(10.0)) * math.log(eps)
    assert loss.item() == pytest.approx(expected_loss, rel=1e-4)

    # Verify backpropagation produces finite gradients even with zero MSE
    loss.backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all().item()


def test_psnr_loss_autograd_finiteness() -> None:
    """Test PSNRLoss gradient computation and finiteness."""
    criterion = PSNRLoss()
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)

    loss = criterion(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


def test_psnr_loss_shape_mismatch() -> None:
    """Test PSNRLoss raises ValueError on shape mismatch."""
    criterion = PSNRLoss()
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 32, 32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        criterion(pred, target)


# --- Loss Builder Tests ---


def test_build_loss_charbonnier() -> None:
    """Test building CharbonnierLoss from dict, ConfigDict, or string."""
    loss1 = build_loss("charbonnier")
    assert isinstance(loss1, CharbonnierLoss)
    assert loss1.eps == 1e-3

    cfg_dict = {"loss": {"name": "CharbonnierLoss", "eps": 1e-4, "reduction": "sum"}}
    loss2 = build_loss(cfg_dict)
    assert isinstance(loss2, CharbonnierLoss)
    assert loss2.eps == 1e-4
    assert loss2.reduction == "sum"

    cfg_obj = ConfigDict({"name": "charbonnier", "eps": 1e-2})
    loss3 = build_loss_function(cfg_obj)
    assert isinstance(loss3, CharbonnierLoss)
    assert loss3.eps == 1e-2


def test_build_loss_psnr() -> None:
    """Test building PSNRLoss from dictionary or ConfigDict."""
    cfg_dict = {
        "loss": {
            "name": "PSNRLoss",
            "data_range": 1.0,
            "eps": 1e-6,
            "reduction": "mean",
        }
    }
    loss = build_loss(cfg_dict)
    assert isinstance(loss, PSNRLoss)
    assert loss.data_range == 1.0
    assert loss.eps == 1e-6


def test_build_loss_unsupported() -> None:
    """Test build_loss raises ValueError for unsupported loss types."""
    with pytest.raises(ValueError, match="Unsupported loss type"):
        build_loss({"name": "UnsupportedLossType"})

    with pytest.raises(ValueError, match="Loss type must be a string"):
        build_loss({"name": 12345})
