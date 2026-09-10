import pytest
import torch
import torch.nn as nn

# Update imports with your exact project paths if necessary
from shrinkai.distillation.losses.base import BaseDistillationLoss
from shrinkai.distillation.losses.wrappers import (
    CombinedLoss,
    HybridLoss,
    ProjectedFeatureLoss,
)

# ==========================================
# Mock
# ==========================================


class DummyLoss(BaseDistillationLoss):
    """A dummy loss that returns a fixed value to verify weight distributions."""

    def __init__(self, val: float):
        super().__init__()
        self.val = val

    def forward(self, s_out, t_out, labels=None):
        return torch.tensor(self.val, dtype=torch.float32)


class MockProjector(nn.Module):
    """A mock projector that simply multiplies tensor values by 2."""

    def forward(self, features_dict):
        return {k: v * 2 for k, v in features_dict.items()}


class MockFeatureLoss(BaseDistillationLoss):
    """A mock feature loss that sums the differences to verify pooling behavior."""

    def forward(self, student_outputs, teacher_outputs, labels=None):
        total = 0.0
        for k in student_outputs.keys():
            total += torch.sum(student_outputs[k] - teacher_outputs[k])
        return total


# ==========================================
# Combined
# ==========================================


def test_combined_loss_initialization_format():
    """Verifies that the class raises an error if the input format is not a list of tuples."""
    with pytest.raises(ValueError, match="must be a list of tuples"):
        # Passing separate lists instead of a list of tuples
        CombinedLoss([DummyLoss(1.0), 1.0])


def test_combined_loss_forward():
    """Verifies that the weighted sum computation is mathematically correct."""
    loss_fn = CombinedLoss([(DummyLoss(2.0), 0.5), (DummyLoss(3.0), 2.0)])
    # Expected: 0.5 * 2.0 + 2.0 * 3.0 = 1.0 + 6.0 = 7.0
    out = loss_fn(student_outputs=None, teacher_outputs=None)
    assert torch.allclose(out, torch.tensor(7.0))


def test_combined_loss_ignores_zero_weight():
    """Verifies that a weight of 0 completely ignores the associated loss."""
    loss_fn = CombinedLoss([(DummyLoss(100.0), 0.0), (DummyLoss(5.0), 1.0)])
    out = loss_fn(None, None)
    assert out.item() == 5.0


# ==========================================
# Hybrid
# ==========================================


def test_hybrid_loss_additive():
    """Verifies the additive mode behavior: L = L_prim + weight * L_feat"""
    loss_fn = HybridLoss(
        logit_loss=DummyLoss(2.0),
        feature_loss=DummyLoss(3.0),
        feature_weight=2.0,
        convex_weighting=False,
    )
    # Expected: 1.0 * 2.0 + 2.0 * 3.0 = 8.0
    out = loss_fn(None, None)
    assert out.item() == 8.0


def test_hybrid_loss_convex():
    """Verifies the convex mode behavior: L = (1-w) * L_prim + w * L_feat"""
    loss_fn = HybridLoss(
        logit_loss=DummyLoss(10.0),
        feature_loss=DummyLoss(4.0),
        feature_weight=0.25,
        convex_weighting=True,
    )
    # Expected: 0.75 * 10.0 + 0.25 * 4.0 = 7.5 + 1.0 = 8.5
    out = loss_fn(None, None)
    assert out.item() == 8.5


def test_hybrid_loss_convex_bounds():
    """Verifies that the convex weight is strictly bounded between 0.0 and 1.0."""
    fw = 1.5
    error = f"For convex weighting, feature_weight must be between 0.0and 1.0, got {fw}"
    with pytest.raises(ValueError, match=error):
        HybridLoss(
            logit_loss=DummyLoss(1.0),
            feature_loss=DummyLoss(1.0),
            feature_weight=fw,
            convex_weighting=True,
        )


# ==========================================
# Projected Feature Loss
# ==========================================


def test_projected_feature_loss_requires_tuples():
    """Verifies the safeguard rejecting raw tensors (requires FeatureExtractor tuples)."""
    loss_fn = ProjectedFeatureLoss(MockProjector(), MockFeatureLoss())

    with pytest.raises(ValueError, match="expects outputs to be a tuple"):
        # Simulating raw tensor inputs instead of (logits, features_dict)
        loss_fn(torch.tensor(1), torch.tensor(1))


def test_projected_feature_loss_auto_pooling_cnn():
    """Verifies the projector application and spatial alignment (Adaptive Pooling 2D)."""
    loss_fn = ProjectedFeatureLoss(MockProjector(), MockFeatureLoss())

    # Student: 16x16 spatial map initialized with 1.0s
    s_features = {"layer1": torch.ones(1, 16, 16, 16)}

    # Teacher: 8x8 spatial map (smaller) initialized with 1.0s
    t_features = {"layer1": torch.ones(1, 16, 8, 8)}

    # Simulate FeatureExtractor outputs -> Tuple (logits, dict)
    s_out = (None, s_features)
    t_out = (None, t_features)

    # Internal operations:
    # 1. Projector: multiplies student features by 2 (-> 2.0s)
    # 2. Pooling: downsamples 16x16 to 8x8 (values remain 2.0s due to avg pooling)
    # 3. Loss computation: sum(student - teacher) -> sum(2.0 - 1.0) over 1*16*8*8 tensor
    # Expected result: 1 * 16 * 64 = 1024
    out = loss_fn(s_out, t_out)

    assert out.item() == 1024.0


def test_projected_feature_loss_projects_student_by_default():
    """Verifies that by default (project_teacher=False), the student is projected."""
    loss_fn = ProjectedFeatureLoss(MockProjector(), MockFeatureLoss(), project_teacher=False)

    s_features = {"layer1": torch.ones(1, 10)}
    t_features = {"layer1": torch.ones(1, 10)}
    out = loss_fn((None, s_features), (None, t_features))

    assert out.item() == 10.0


def test_projected_feature_loss_projects_teacher_when_requested():
    """Verifies that when project_teacher=True, the teacher is projected (useful for attention)."""
    loss_fn = ProjectedFeatureLoss(MockProjector(), MockFeatureLoss(), project_teacher=True)

    s_features = {"layer1": torch.ones(1, 10)}
    t_features = {"layer1": torch.ones(1, 10)}
    out = loss_fn((None, s_features), (None, t_features))

    assert out.item() == -10.0
