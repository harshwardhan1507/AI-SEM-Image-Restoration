"""Unit tests for NAFNet foundational computational primitives (blocks.py)."""

import pytest
import torch

from src.models import LayerNorm2d, SimpleGate, SimplifiedChannelAttention


class TestLayerNorm2d:
    """Test suite for LayerNorm2d module."""

    def test_init_validation(self) -> None:
        """Verify constructor parameter validation."""
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            LayerNorm2d(0)

        with pytest.raises(ValueError, match="channels must be a positive integer"):
            LayerNorm2d(-8)

        with pytest.raises(ValueError, match="eps must be positive"):
            LayerNorm2d(16, eps=0.0)

        with pytest.raises(ValueError, match="eps must be positive"):
            LayerNorm2d(16, eps=-1e-5)

    def test_parameter_shapes_and_initialization(self) -> None:
        """Verify parameter shapes, initial values, and parameter count."""
        channels = 32
        norm = LayerNorm2d(channels=channels, eps=1e-6)

        assert norm.weight.shape == (1, channels, 1, 1)
        assert norm.bias.shape == (1, channels, 1, 1)

        assert torch.allclose(norm.weight, torch.ones(1, channels, 1, 1))
        assert torch.allclose(norm.bias, torch.zeros(1, channels, 1, 1))

        param_count = sum(p.numel() for p in norm.parameters())
        assert param_count == 2 * channels

    def test_shape_preservation(self) -> None:
        """Verify shape contract (B, C, H, W) -> (B, C, H, W) across various tensor sizes."""
        test_shapes = [
            (1, 16, 64, 64),
            (4, 32, 128, 128),
            (2, 64, 32, 32),
            (8, 3, 16, 16),
        ]
        for shape in test_shapes:
            x = torch.randn(*shape)
            norm = LayerNorm2d(channels=shape[1])
            y = norm(x)
            assert y.shape == shape, f"Expected shape {shape}, got {y.shape}"

    def test_input_shape_and_channel_validation(self) -> None:
        """Verify error handling on incorrect input dimensions or channel mismatches."""
        norm = LayerNorm2d(channels=16)

        # 3D tensor instead of 4D
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            norm(torch.randn(16, 32, 32))

        # Channel mismatch (32 channels instead of 16)
        with pytest.raises(ValueError, match="Expected tensor with 16 channels"):
            norm(torch.randn(2, 32, 64, 64))

    def test_mathematical_correctness(self) -> None:
        """Verify mean is zero and variance is unit per spatial position."""
        channels = 64
        norm = LayerNorm2d(channels=channels, eps=1e-6)

        # Input with random non-zero mean and variance
        x = torch.randn(4, channels, 16, 16) * 5.0 + 3.0
        y = norm(x)

        # Calculate channel-wise mean and variance per pixel
        mean = y.mean(dim=1)
        var = (y - y.mean(dim=1, keepdim=True)).pow(2).mean(dim=1)

        # Mean should be close to 0 and var close to 1
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_gradient_propagation(self) -> None:
        """Verify backward pass and gradient flow through weight and bias."""
        channels = 16
        norm = LayerNorm2d(channels=channels)
        x = torch.randn(2, channels, 16, 16, requires_grad=True)

        y = norm(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None

        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(norm.weight.grad).any()
        assert not torch.isnan(norm.bias.grad).any()

    def test_determinism_and_eval_mode(self) -> None:
        """Verify train() and eval() modes yield identical deterministic outputs."""
        norm = LayerNorm2d(channels=16)
        x = torch.randn(2, 16, 32, 32)

        norm.train()
        y_train = norm(x)

        norm.eval()
        y_eval = norm(x)

        assert torch.equal(y_train, y_eval)

    def test_torchscript_compatibility(self) -> None:
        """Verify TorchScript scripting and tracing compatibility."""
        norm = LayerNorm2d(channels=16)
        x = torch.randn(2, 16, 32, 32)

        # Test JIT script
        scripted_norm = torch.jit.script(norm)
        y_script = scripted_norm(x)
        assert torch.allclose(norm(x), y_script, atol=1e-6)

        # Test JIT trace
        traced_norm = torch.jit.trace(norm, x)
        y_trace = traced_norm(x)
        assert torch.allclose(norm(x), y_trace, atol=1e-6)


class TestSimpleGate:
    """Test suite for SimpleGate module."""

    def test_zero_parameters(self) -> None:
        """Verify SimpleGate has zero learnable parameters."""
        gate = SimpleGate()
        param_count = sum(p.numel() for p in gate.parameters())
        assert param_count == 0
        assert len(list(gate.parameters())) == 0

    def test_shape_reduction(self) -> None:
        """Verify shape contract (B, 2C, H, W) -> (B, C, H, W)."""
        gate = SimpleGate()
        test_cases = [
            ((1, 64, 32, 32), (1, 32, 32, 32)),
            ((2, 128, 64, 64), (2, 64, 64, 64)),
            ((4, 16, 16, 16), (4, 8, 16, 16)),
        ]
        for input_shape, expected_shape in test_cases:
            x = torch.randn(*input_shape)
            y = gate(x)
            assert y.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {y.shape}"
            )

    def test_input_validation(self) -> None:
        """Verify exception handling for non-4D tensors or odd channel counts."""
        gate = SimpleGate()

        # 3D tensor
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            gate(torch.randn(32, 16, 16))

        # Odd channel count (15)
        with pytest.raises(
            ValueError, match="SimpleGate requires input channel dimension to be even"
        ):
            gate(torch.randn(2, 15, 32, 32))

    def test_mathematical_correctness(self) -> None:
        """Verify element-wise multiplication of split channels."""
        gate = SimpleGate()

        # Case 1: Ones input (B, 2C, H, W) -> (B, C, H, W) of ones
        x_ones = torch.ones(2, 64, 16, 16)
        y_ones = gate(x_ones)
        assert torch.allclose(y_ones, torch.ones(2, 32, 16, 16))

        # Case 2: Specific values X1 = 2.0, X2 = 3.0
        x1 = torch.full((1, 8, 4, 4), 2.0)
        x2 = torch.full((1, 8, 4, 4), 3.0)
        x_concat = torch.cat([x1, x2], dim=1)  # shape (1, 16, 4, 4)
        y_concat = gate(x_concat)
        assert torch.allclose(y_concat, torch.full((1, 8, 4, 4), 6.0))

    def test_gradient_propagation(self) -> None:
        """Verify gradient flow through SimpleGate back to input tensor."""
        gate = SimpleGate()
        x = torch.randn(2, 16, 16, 16, requires_grad=True)

        y = gate(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.shape == x.shape

    def test_determinism_and_eval_mode(self) -> None:
        """Verify train() and eval() modes yield identical outputs."""
        gate = SimpleGate()
        x = torch.randn(2, 32, 16, 16)

        gate.train()
        y_train = gate(x)

        gate.eval()
        y_eval = gate(x)

        assert torch.equal(y_train, y_eval)

    def test_torchscript_compatibility(self) -> None:
        """Verify TorchScript scripting and tracing compatibility."""
        gate = SimpleGate()
        x = torch.randn(2, 32, 16, 16)

        # Test JIT script
        scripted_gate = torch.jit.script(gate)
        y_script = scripted_gate(x)
        assert torch.allclose(gate(x), y_script)

        # Test JIT trace
        traced_gate = torch.jit.trace(gate, x)
        y_trace = traced_gate(x)
        assert torch.allclose(gate(x), y_trace)


class TestSimplifiedChannelAttention:
    """Test suite for SimplifiedChannelAttention (SCA) module."""

    def test_init_validation(self) -> None:
        """Verify constructor channel validation."""
        with pytest.raises(ValueError, match="channels must be a positive integer"):
            SimplifiedChannelAttention(0)

        with pytest.raises(ValueError, match="channels must be a positive integer"):
            SimplifiedChannelAttention(-16)

    def test_parameter_count(self) -> None:
        """Verify parameter count matches 1x1 conv (C * C weight + C bias)."""
        channels = 32
        sca = SimplifiedChannelAttention(channels=channels)

        expected_params = (
            channels * channels + channels
        )  # weight: (C, C, 1, 1), bias: (C,)
        param_count = sum(p.numel() for p in sca.parameters())
        assert param_count == expected_params

    def test_shape_preservation(self) -> None:
        """Verify shape contract (B, C, H, W) -> (B, C, H, W) across various shapes."""
        test_shapes = [
            (1, 16, 64, 64),
            (4, 32, 128, 128),
            (2, 64, 32, 32),
            (8, 3, 16, 16),
        ]
        for shape in test_shapes:
            x = torch.randn(*shape)
            sca = SimplifiedChannelAttention(channels=shape[1])
            y = sca(x)
            assert y.shape == shape, f"Expected shape {shape}, got {y.shape}"

    def test_input_validation(self) -> None:
        """Verify exception handling for non-4D tensors or channel mismatches."""
        sca = SimplifiedChannelAttention(channels=16)

        # 3D tensor
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            sca(torch.randn(16, 32, 32))

        # Channel mismatch (32 channels instead of 16)
        with pytest.raises(ValueError, match="Expected tensor with 16 channels"):
            sca(torch.randn(2, 32, 64, 64))

    def test_gradient_propagation(self) -> None:
        """Verify backward pass and gradient flow through input, conv weights, and conv bias."""
        channels = 16
        sca = SimplifiedChannelAttention(channels=channels)
        x = torch.randn(2, channels, 16, 16, requires_grad=True)

        y = sca(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert sca.conv.weight.grad is not None
        assert sca.conv.bias.grad is not None

        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(sca.conv.weight.grad).any()
        assert not torch.isnan(sca.conv.bias.grad).any()

    def test_determinism_and_eval_mode(self) -> None:
        """Verify train() and eval() modes yield identical deterministic outputs."""
        sca = SimplifiedChannelAttention(channels=16)
        x = torch.randn(2, 16, 32, 32)

        sca.train()
        y_train = sca(x)

        sca.eval()
        y_eval = sca(x)

        assert torch.equal(y_train, y_eval)

    def test_torchscript_compatibility(self) -> None:
        """Verify TorchScript scripting and tracing compatibility."""
        sca = SimplifiedChannelAttention(channels=16)
        x = torch.randn(2, 16, 32, 32)

        # Test JIT script
        scripted_sca = torch.jit.script(sca)
        y_script = scripted_sca(x)
        assert torch.allclose(sca(x), y_script, atol=1e-6)

        # Test JIT trace
        traced_sca = torch.jit.trace(sca, x)
        y_trace = traced_sca(x)
        assert torch.allclose(sca(x), y_trace, atol=1e-6)
