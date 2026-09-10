import pytest
import torch
import torch.nn as nn

from shrinkai.adapters.projector import AttentionHeadSelector, FeatureProjector

# ==========================================
# FEATURE PROJECTOR
# ==========================================


def test_projector_initialization_conv_linear():
    """Verifies that the projector correctly builds Conv2d and Linear layers based on config."""
    config = {
        "conv_layer": {
            "in_channels": 16,
            "out_channels": 32,
            "type": "conv",
            "use_norm": False,
        },
        "linear_layer": {
            "in_channels": 64,
            "out_channels": 128,
            "type": "linear",
            "use_norm": False,
        },
    }

    projector = FeatureProjector(config)

    assert isinstance(projector.projectors["conv_layer"][0], nn.Conv2d)
    assert projector.projectors["conv_layer"][0].in_channels == 16
    assert projector.projectors["conv_layer"][0].out_channels == 32

    assert isinstance(projector.projectors["linear_layer"][0], nn.Linear)
    assert projector.projectors["linear_layer"][0].in_features == 64
    assert projector.projectors["linear_layer"][0].out_features == 128


def test_projector_initialization_with_norm():
    """Verifies that normalization layers are properly appended when requested."""
    config = {
        "conv_norm": {
            "in_channels": 16,
            "out_channels": 32,
            "type": "conv",
            "use_norm": True,
        },
        "lin_norm": {
            "in_channels": 64,
            "out_channels": 128,
            "type": "linear",
            "use_norm": True,
        },
    }

    projector = FeatureProjector(config)

    # Conv should be followed by BatchNorm2d
    assert isinstance(projector.projectors["conv_norm"][1], nn.BatchNorm2d)
    assert projector.projectors["conv_norm"][1].num_features == 32

    # Linear should be followed by LayerNorm
    assert isinstance(projector.projectors["lin_norm"][1], nn.LayerNorm)


def test_projector_invalid_type_raises_error():
    """Verifies that an unsupported projection type raises a ValueError."""
    config = {"bad_layer": {"in_channels": 10, "out_channels": 20, "type": "rnn"}}

    with pytest.raises(ValueError, match="Unsupported projection type 'rnn'"):
        FeatureProjector(config)


def test_projector_safe_keys_mapping():
    """Verifies that dot notation (e.g., 'layer.1') is safely converted for ModuleDict."""
    config = {"layer.1.conv": {"in_channels": 16, "out_channels": 32, "type": "conv"}}
    projector = FeatureProjector(config)

    assert "layer_1_conv" in projector.projectors
    assert projector.mapping_keys["layer.1.conv"] == "layer_1_conv"


def test_projector_forward_shapes_and_passthrough():
    """Verifies that mapped features are reshaped and unmapped features are ignored."""
    config = {
        "layer_conv": {"in_channels": 16, "out_channels": 32, "type": "conv"},
        "layer_lin": {"in_channels": 64, "out_channels": 128, "type": "linear"},
    }
    projector = FeatureProjector(config)

    student_features = {
        "layer_conv": torch.randn(2, 16, 8, 8),
        "layer_lin": torch.randn(2, 64),
        "unmapped_layer": torch.randn(2, 5),
    }

    out = projector(student_features)

    assert out["layer_conv"].shape == (2, 32, 8, 8)
    assert out["layer_lin"].shape == (2, 128)
    assert "unmapped_layer" in out
    assert torch.equal(out["unmapped_layer"], student_features["unmapped_layer"])


# ==========================================
# ATTENTION SELECTOR
# ==========================================


def test_attention_head_selector_valid_shapes():
    """Verifies that the selector correctly slices 4D attention tensors."""
    teacher_attention = {
        "layer_1": torch.rand(2, 12, 16, 16),
        "layer_2": torch.rand(2, 12, 16, 16),
    }

    selector = AttentionHeadSelector(heads_to_keep=[0, 6])
    projected = selector(teacher_attention)

    for key, tensor in projected.items():
        assert tensor.shape == (2, 2, 16, 16)
        assert torch.equal(tensor[:, 0, :, :], teacher_attention[key][:, 0, :, :])
        assert torch.equal(tensor[:, 1, :, :], teacher_attention[key][:, 6, :, :])


def test_attention_head_selector_invalid_dim():
    """Verifies that the selector raises a ValueError for non-4D tensors."""
    invalid_attention = {"layer_1": torch.rand(2, 16, 768)}

    selector = AttentionHeadSelector(heads_to_keep=[0, 6])
    with pytest.raises(ValueError, match="Expected 4D attention tensor"):
        selector(invalid_attention)
