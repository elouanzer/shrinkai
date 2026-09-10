import pytest
import torch
import torch.nn as nn

from shrinkai.adapters.extractor import FeatureExtractor

# ==========================================
# MOCK
# ==========================================


class DummyTupleLayer(nn.Module):
    """A mock layer that returns a tuple to test the tuple-handling hook logic."""

    def forward(self, x):
        return x, x * 2, "metadata"


class SimpleNet(nn.Module):
    """A mock neural network to test feature extraction."""

    def __init__(self):
        super().__init__()
        self.block_1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.block_2 = nn.Linear(20, 5)
        self.tuple_layer = DummyTupleLayer()

    def forward(self, x):
        x = self.block_1(x)
        x = self.relu(x)
        x = self.block_2(x)
        x, _, _ = self.tuple_layer(x)
        return x


# ==========================================
# FEATURE EXTRACTOR
# ==========================================


def test_extractor_initialization_with_list():
    """Verifies that passing a list of layer names creates an identity mapping."""
    model = SimpleNet()
    extractor = FeatureExtractor(model, target_layers=["block_1", "relu"])

    assert extractor.target_layers == {"block_1": "block_1", "relu": "relu"}
    assert len(extractor._hooks) == 2


def test_extractor_initialization_with_dict():
    """Verifies that passing a dict properly maps layer names to aliases."""
    model = SimpleNet()
    extractor = FeatureExtractor(model, target_layers={"block_1": "stage_1", "block_2": "stage_2"})

    assert extractor.target_layers == {"block_1": "stage_1", "block_2": "stage_2"}


def test_extractor_invalid_layer_raises_error():
    """Verifies that targeting a non-existent layer raises a ValueError."""
    model = SimpleNet()
    with pytest.raises(ValueError, match="not found in the model"):
        FeatureExtractor(model, target_layers=["fake_layer"])


def test_extractor_forward_pass_captures_features():
    """Verifies that the hook system captures intermediate features during forward pass."""
    model = SimpleNet()
    extractor = FeatureExtractor(model, target_layers={"block_1": "stage_1", "block_2": "stage_2"})

    x = torch.randn(2, 10)
    logits, features = extractor(x)

    assert logits.shape == (2, 5)

    assert "stage_1" in features
    assert "stage_2" in features

    assert features["stage_1"].shape == (2, 20)
    assert features["stage_2"].shape == (2, 5)


def test_extractor_handles_tuple_outputs():
    """Verifies that the hook safely extracts the main tensor if a layer outputs a tuple."""
    model = SimpleNet()
    extractor = FeatureExtractor(model, target_layers=["tuple_layer"])

    x = torch.randn(2, 10)
    _, features = extractor(x)

    assert "tuple_layer" in features
    assert isinstance(features["tuple_layer"], torch.Tensor)
    assert features["tuple_layer"].shape == (2, 5)


def test_extractor_remove_hooks():
    """Verifies that calling remove_hooks() detaches all hooks successfully."""
    model = SimpleNet()
    extractor = FeatureExtractor(model, target_layers=["block_1", "block_2"])
    assert len(extractor._hooks) == 2

    extractor.remove_hooks()
    assert len(extractor._hooks) == 0

    x = torch.randn(2, 10)
    _, features = extractor(x)
    assert len(features) == 0
