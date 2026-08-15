"""Comprehensive unit tests for NAFNet architecture and model builder.

Verifies the complete NAFNet architecture (Issue #11) including:
- Construction via builder
- Forward pass shape contracts (B,1,128,128) -> (B,1,256,256)
- Multi-resolution inputs
- Batch size variations
- Gradient flow
- Parameter count validation
- Determinism
- torch.compile compatibility
- AMP autocast compatibility
- CPU execution
- Memory leak checks
"""

import gc
from typing import Any

import pytest
import torch
import torch.nn as nn

from src.models.builder import build_model
from src.models.nafnet import NAFNet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model() -> NAFNet:
    """Create a tiny NAFNet model for fast testing."""
    return NAFNet(
        img_channel=1,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        upscale=2,
        drop_out_rate=0.0,
    )


@pytest.fixture
def small_model() -> NAFNet:
    """Create a small NAFNet model matching NAFNet-Small SR config."""
    return NAFNet(
        img_channel=1,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        upscale=2,
        drop_out_rate=0.0,
    )


# ---------------------------------------------------------------------------
# Test: Builder Construction
# ---------------------------------------------------------------------------


class TestBuilderConstruction:
    """Tests for the build_model factory function."""

    def test_builder_returns_nafnet(self) -> None:
        """build_model returns a NAFNet instance."""
        cfg: dict[str, Any] = {
            "model": {
                "img_channel": 1,
                "width": 16,
                "enc_blk_nums": [1, 1],
                "dec_blk_nums": [1, 1],
                "middle_blk_num": 1,
                "upscale": 2,
            }
        }
        model = build_model(cfg)
        assert isinstance(model, NAFNet)

    def test_builder_with_flat_config(self) -> None:
        """build_model works with flat dictionary (no 'model' key)."""
        cfg: dict[str, Any] = {
            "img_channel": 1,
            "width": 16,
            "enc_blk_nums": [1, 1],
            "dec_blk_nums": [1, 1],
            "middle_blk_num": 1,
            "upscale": 2,
        }
        model = build_model(cfg)
        assert isinstance(model, NAFNet)

    def test_builder_uses_defaults(self) -> None:
        """build_model uses default parameters when not specified."""
        cfg: dict[str, Any] = {"model": {}}
        model = build_model(cfg)
        assert isinstance(model, NAFNet)
        assert model.img_channel == 1
        assert model.width == 32
        assert model.upscale == 2

    def test_builder_invalid_width(self) -> None:
        """build_model raises ValueError for non-positive width."""
        cfg: dict[str, Any] = {"model": {"width": 0}}
        with pytest.raises(ValueError, match="width"):
            build_model(cfg)

    def test_builder_invalid_width_type(self) -> None:
        """build_model raises ValueError for non-integer width."""
        cfg: dict[str, Any] = {"model": {"width": 3.5}}
        with pytest.raises(ValueError, match="width"):
            build_model(cfg)

    def test_builder_mismatched_stages(self) -> None:
        """build_model raises ValueError when enc/dec stage counts differ."""
        cfg: dict[str, Any] = {
            "model": {"enc_blk_nums": [1, 1, 1], "dec_blk_nums": [1, 1]}
        }
        with pytest.raises(ValueError, match="same length"):
            build_model(cfg)

    def test_builder_invalid_enc_blk_type(self) -> None:
        """build_model raises ValueError for non-list enc_blk_nums."""
        cfg: dict[str, Any] = {"model": {"enc_blk_nums": "invalid"}}
        with pytest.raises(ValueError, match="enc_blk_nums"):
            build_model(cfg)

    def test_builder_negative_drop_out_rate(self) -> None:
        """build_model raises ValueError for negative drop_out_rate."""
        cfg: dict[str, Any] = {"model": {"drop_out_rate": -0.1}}
        with pytest.raises(ValueError, match="drop_out_rate"):
            build_model(cfg)


# ---------------------------------------------------------------------------
# Test: Direct NAFNet Construction
# ---------------------------------------------------------------------------


class TestNAFNetConstruction:
    """Tests for direct NAFNet constructor validation."""

    def test_valid_construction(self) -> None:
        """NAFNet constructs successfully with valid parameters."""
        model = NAFNet(
            img_channel=1,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            upscale=2,
        )
        assert isinstance(model, nn.Module)

    def test_negative_img_channel(self) -> None:
        """NAFNet raises ValueError for non-positive img_channel."""
        with pytest.raises(ValueError, match="img_channel"):
            NAFNet(img_channel=0)

    def test_negative_width(self) -> None:
        """NAFNet raises ValueError for non-positive width."""
        with pytest.raises(ValueError, match="width"):
            NAFNet(width=-1)

    def test_negative_middle_blk_num(self) -> None:
        """NAFNet raises ValueError for non-positive middle_blk_num."""
        with pytest.raises(ValueError, match="middle_blk_num"):
            NAFNet(middle_blk_num=0)

    def test_empty_enc_blk_nums(self) -> None:
        """NAFNet raises ValueError for empty enc_blk_nums."""
        with pytest.raises(ValueError, match="enc_blk_nums"):
            NAFNet(enc_blk_nums=[])

    def test_mismatched_stage_lengths(self) -> None:
        """NAFNet raises ValueError for mismatched enc/dec lengths."""
        with pytest.raises(ValueError, match="same length"):
            NAFNet(enc_blk_nums=[1, 1], dec_blk_nums=[1])

    def test_zero_block_in_enc(self) -> None:
        """NAFNet raises ValueError for zero block count in encoder."""
        with pytest.raises(ValueError, match="enc_blk_nums"):
            NAFNet(enc_blk_nums=[1, 0, 1], dec_blk_nums=[1, 1, 1])

    def test_negative_upscale(self) -> None:
        """NAFNet raises ValueError for non-positive upscale."""
        with pytest.raises(ValueError, match="upscale"):
            NAFNet(upscale=0)

    def test_invalid_drop_out_rate(self) -> None:
        """NAFNet raises ValueError for drop_out_rate > 1.0."""
        with pytest.raises(ValueError, match="drop_out_rate"):
            NAFNet(drop_out_rate=1.5)


# ---------------------------------------------------------------------------
# Test: Forward Pass Shape Contracts
# ---------------------------------------------------------------------------


class TestForwardShapeContracts:
    """Tests for tensor shape contracts through the forward pass."""

    def test_primary_contract_128_to_256(self, tiny_model: NAFNet) -> None:
        """Primary shape contract: (B,1,128,128) -> (B,1,256,256)."""
        x = torch.randn(1, 1, 128, 128)
        y = tiny_model(x)
        assert y.shape == (1, 1, 256, 256)

    def test_input_256(self, tiny_model: NAFNet) -> None:
        """Shape contract: (B,1,256,256) -> (B,1,512,512)."""
        x = torch.randn(1, 1, 256, 256)
        y = tiny_model(x)
        assert y.shape == (1, 1, 512, 512)

    def test_input_64(self, tiny_model: NAFNet) -> None:
        """Shape contract: (B,1,64,64) -> (B,1,128,128)."""
        x = torch.randn(1, 1, 64, 64)
        y = tiny_model(x)
        assert y.shape == (1, 1, 128, 128)

    def test_non_square_input(self, tiny_model: NAFNet) -> None:
        """Shape contract for non-square input: (B,1,128,64) -> (B,1,256,128)."""
        x = torch.randn(1, 1, 128, 64)
        y = tiny_model(x)
        assert y.shape == (1, 1, 256, 128)

    def test_non_power_of_two(self, tiny_model: NAFNet) -> None:
        """Shape contract for dimensions not divisible by padder_size.

        Input is auto-padded internally and output is cropped back.
        """
        x = torch.randn(1, 1, 100, 100)
        y = tiny_model(x)
        assert y.shape == (1, 1, 200, 200)

    def test_wrong_channel_count(self, tiny_model: NAFNet) -> None:
        """Forward raises ValueError for incorrect input channel count."""
        x = torch.randn(1, 3, 128, 128)  # Model expects 1 channel
        with pytest.raises(ValueError, match="channels"):
            tiny_model(x)

    def test_wrong_dimensions(self, tiny_model: NAFNet) -> None:
        """Forward raises ValueError for non-4D input tensor."""
        x = torch.randn(1, 128, 128)  # 3D tensor
        with pytest.raises(ValueError, match="4D"):
            tiny_model(x)

    def test_upscale_1(self) -> None:
        """Shape contract for upscale=1 (same-resolution restoration)."""
        model = NAFNet(
            img_channel=1,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            upscale=1,
        )
        x = torch.randn(1, 1, 128, 128)
        y = model(x)
        assert y.shape == (1, 1, 128, 128)

    def test_rectangular_micrograph_300_400(self, tiny_model: NAFNet) -> None:
        """Shape contract for arbitrary rectangular input: (1, 1, 300, 400) -> (1, 1, 600, 800)."""
        x = torch.randn(1, 1, 300, 400)
        y = tiny_model(x)
        assert y.shape == (1, 1, 600, 800)
        assert torch.isfinite(y).all()
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_output_sanity_finite(self, tiny_model: NAFNet) -> None:
        """Model output contains only finite numerical values (zero NaNs and zero Infs)."""
        x = torch.rand(2, 1, 128, 128)
        y = tiny_model(x)
        assert y.shape == (2, 1, 256, 256)
        assert torch.isfinite(y).all().item()
        assert not torch.isnan(y).any().item()
        assert not torch.isinf(y).any().item()

    def test_3_channel_input(self) -> None:
        """Shape contract for 3-channel (RGB) input."""
        model = NAFNet(
            img_channel=3,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            upscale=2,
        )
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert y.shape == (1, 3, 128, 128)


# ---------------------------------------------------------------------------
# Test: Batch Sizes
# ---------------------------------------------------------------------------


class TestBatchSizes:
    """Tests for various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_forward(self, tiny_model: NAFNet, batch_size: int) -> None:
        """Forward pass works with batch sizes 1, 2, and 4."""
        x = torch.randn(batch_size, 1, 64, 64)
        y = tiny_model(x)
        assert y.shape == (batch_size, 1, 128, 128)


# ---------------------------------------------------------------------------
# Test: Gradient Flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests for gradient flow through all parameters."""

    def test_all_parameters_receive_gradients(self, tiny_model: NAFNet) -> None:
        """Backward pass propagates gradients to all parameters.

        Note: Due to NAFNet's zero-init design (beta=0, gamma=0), parameters
        inside NAFBlock sub-branches (e.g., LayerNorm2d weight/bias, conv
        weights upstream of a zero-scaled branch) may correctly receive
        zero gradients at initialization. This test verifies that all
        parameters are part of the computation graph (grad is not None).
        """
        x = torch.randn(1, 1, 64, 64)
        y = tiny_model(x)
        loss = y.sum()
        loss.backward()

        for name, param in tiny_model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has None gradient after backward pass"
            )

    def test_critical_parameters_have_nonzero_gradients(
        self, tiny_model: NAFNet
    ) -> None:
        """Critical structural parameters (head, tail, beta, gamma) have non-zero gradients."""
        x = torch.randn(1, 1, 64, 64)
        y = tiny_model(x)
        loss = y.sum()
        loss.backward()

        # Head and tail convolutions must always have non-zero gradients
        for name, param in tiny_model.named_parameters():
            if "intro" in name or "up_tail" in name:
                assert param.grad is not None and param.grad.abs().sum() > 0, (
                    f"Critical parameter '{name}' has zero gradient"
                )

        # beta and gamma scaling parameters must have non-zero gradients
        # (they are the gateway for branch learning to begin)
        for name, param in tiny_model.named_parameters():
            if name.endswith("beta") or name.endswith("gamma"):
                assert param.grad is not None and param.grad.abs().sum() > 0, (
                    f"Scaling parameter '{name}' has zero gradient"
                )

    def test_no_detached_parameters(self, tiny_model: NAFNet) -> None:
        """All parameters require gradients (none accidentally detached)."""
        for name, param in tiny_model.named_parameters():
            assert param.requires_grad, f"Parameter '{name}' does not require gradients"


# ---------------------------------------------------------------------------
# Test: Parameter Count
# ---------------------------------------------------------------------------


class TestParameterCount:
    """Tests for parameter count validation against analytical formula."""

    def test_nafblock_params_formula(self) -> None:
        """Verify NAFBlock parameter count matches formula P(C) = 7C² + 33C."""
        from src.models.nafblock import NAFBlock

        for c in [16, 32, 64, 128]:
            block = NAFBlock(c)
            actual = sum(p.numel() for p in block.parameters())
            expected = 7 * c * c + 33 * c
            assert actual == expected, (
                f"NAFBlock({c}): expected {expected}, got {actual}"
            )

    def test_total_parameter_count_tiny(self) -> None:
        """Verify total parameter count for tiny configuration is reasonable."""
        model = NAFNet(
            img_channel=1,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
            upscale=2,
        )
        total_params = sum(p.numel() for p in model.parameters())
        # Tiny model should have a reasonable number of parameters
        assert total_params > 0
        assert total_params < 1_000_000  # Should be well under 1M

    def test_parameter_count_scales_with_width(self) -> None:
        """Doubling width should approximately quadruple parameters (due to C² scaling)."""
        model_16 = NAFNet(
            img_channel=1,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            upscale=2,
        )
        model_32 = NAFNet(
            img_channel=1,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            upscale=2,
        )
        params_16 = sum(p.numel() for p in model_16.parameters())
        params_32 = sum(p.numel() for p in model_32.parameters())
        # Ratio should be roughly 4x (due to C² in NAFBlock formula)
        ratio = params_32 / params_16
        assert 3.0 < ratio < 5.0, f"Width scaling ratio: {ratio:.2f} (expected ~4.0)"

    def test_preferred_width48_parameter_count(self) -> None:
        """Verify the preferred deployment model (Width 48) parameter count (~2.52M)."""
        cfg = {
            "model": {
                "img_channel": 1,
                "width": 48,
                "middle_blk_num": 1,
                "enc_blk_nums": [1, 1, 1],
                "dec_blk_nums": [1, 1, 1],
                "upscale": 2,
            }
        }
        model_48 = build_model(cfg)
        params_48 = sum(p.numel() for p in model_48.parameters())
        assert params_48 == 2_521_444, (
            f"Expected 2,521,444 params for Width 48, got {params_48}"
        )


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for deterministic output given identical inputs."""

    def test_deterministic_output(self, tiny_model: NAFNet) -> None:
        """Identical inputs produce identical outputs in eval mode."""
        tiny_model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y1 = tiny_model(x.clone())
            y2 = tiny_model(x.clone())
        assert torch.allclose(y1, y2, atol=1e-6), (
            f"Outputs differ: max diff = {(y1 - y2).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test: torch.compile
# ---------------------------------------------------------------------------


class TestTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available in this PyTorch version",
    )
    def test_compile_forward(self, tiny_model: NAFNet) -> None:
        """Model can be compiled with torch.compile and produces correct shapes.

        Note: This test may be skipped on Windows if the C++ compiler (cl.exe)
        is not available, as torch.compile requires the Inductor backend
        which depends on a C++ toolchain.
        """
        try:
            compiled_model = torch.compile(tiny_model)
            x = torch.randn(1, 1, 64, 64)
            y = compiled_model(x)
            assert y.shape == (1, 1, 128, 128)
        except Exception as e:
            if "Compiler" in str(e) or "InvalidCxxCompiler" in str(e):
                pytest.skip(f"torch.compile requires a C++ compiler not available: {e}")
            raise


# ---------------------------------------------------------------------------
# Test: AMP Autocast
# ---------------------------------------------------------------------------


class TestAMPAutocast:
    """Tests for Automatic Mixed Precision compatibility."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for AMP test"
    )
    def test_amp_forward_cuda(self, tiny_model: NAFNet) -> None:
        """Forward pass works under CUDA AMP autocast."""
        device = torch.device("cuda")
        model = tiny_model.to(device)
        x = torch.randn(1, 1, 64, 64, device=device)
        with torch.cuda.amp.autocast():
            y = model(x)
        assert y.shape == (1, 1, 128, 128)

    def test_amp_forward_cpu(self, tiny_model: NAFNet) -> None:
        """Forward pass works under CPU AMP autocast."""
        x = torch.randn(1, 1, 64, 64)
        with torch.amp.autocast("cpu"):
            y = tiny_model(x)
        assert y.shape == (1, 1, 128, 128)


# ---------------------------------------------------------------------------
# Test: CPU Forward
# ---------------------------------------------------------------------------


class TestCPUForward:
    """Tests for CPU device execution."""

    def test_cpu_forward(self, tiny_model: NAFNet) -> None:
        """Model runs on CPU and produces correct output shapes."""
        assert next(tiny_model.parameters()).device.type == "cpu"
        x = torch.randn(1, 1, 64, 64)
        y = tiny_model(x)
        assert y.shape == (1, 1, 128, 128)
        assert y.device.type == "cpu"


# ---------------------------------------------------------------------------
# Test: Memory Leaks
# ---------------------------------------------------------------------------


class TestMemoryLeaks:
    """Tests for tensor memory leak detection."""

    def test_no_tensor_leaks(self, tiny_model: NAFNet) -> None:
        """Multiple forward/backward iterations do not leak tensors."""
        gc.collect()
        initial_tensors = len(gc.get_objects())

        for _ in range(5):
            x = torch.randn(1, 1, 32, 32)
            y = tiny_model(x)
            loss = y.sum()
            loss.backward()
            tiny_model.zero_grad()

        gc.collect()
        final_tensors = len(gc.get_objects())

        # Allow some tolerance for Python object creation overhead
        tensor_growth = final_tensors - initial_tensors
        assert tensor_growth < 500, (
            f"Potential memory leak: object count grew by {tensor_growth}"
        )
