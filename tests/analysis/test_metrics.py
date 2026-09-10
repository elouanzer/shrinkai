import pytest
import torch

from shrinkai.analysis.metrics import CKAMetric, RSAMetric, SpatialAttentionMetric

# ==========================================
# CKA
# ==========================================


def test_cka_identical_features():
    """Verifies that identical representations yield a CKA score of 1.0."""
    metric = CKAMetric()
    features = torch.randn(32, 128)  # Batch=32, Dim=128

    score = metric.compute(features, features)
    # Using pytest.approx because floating-point math might give 0.999999
    assert score == pytest.approx(1.0, rel=1e-4)


def test_cka_different_dimensions():
    """Verifies that CKA works with different channel dimensions."""
    metric = CKAMetric()
    # Student has 64 channels, Teacher has 256 channels
    s_features = torch.randn(32, 64)
    t_features = torch.randn(32, 256)

    score = metric.compute(s_features, t_features)
    assert 0.0 <= score <= 1.0  # CKA is strictly bounded between 0 and 1


def test_cka_4d_tensors():
    """Verifies that CKA automatically flattens 4D CNN tensors."""
    metric = CKAMetric()
    features = torch.randn(16, 64, 8, 8)

    score = metric.compute(features, features)
    assert score == pytest.approx(1.0, rel=1e-4)


# ==========================================
# RSA
# ==========================================


def test_rsa_identical_features():
    """Verifies that identical topologies yield an RSA score of 1.0."""
    metric = RSAMetric()
    features = torch.randn(32, 128)

    score = metric.compute(features, features)
    assert score == pytest.approx(1.0, rel=1e-4)


# ==========================================
# SPATIAL ATTENTION
# ==========================================


def test_spatial_attention_identical():
    """Verifies that identical CNN feature maps yield an Attention score of 1.0."""
    metric = SpatialAttentionMetric()
    features = torch.randn(16, 64, 16, 16)  # 4D tensor is required

    score = metric.compute(features, features)
    assert score == pytest.approx(1.0, rel=1e-4)


def test_spatial_attention_auto_pooling():
    """Verifies that the metric handles different spatial resolutions (e.g., 16x16 vs 8x8)."""
    metric = SpatialAttentionMetric()
    s_features = torch.randn(16, 64, 16, 16)
    t_features = torch.randn(16, 128, 8, 8)

    # Should run without crashing thanks to internal adaptive_avg_pool2d
    score = metric.compute(s_features, t_features)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_spatial_attention_ignores_1d_2d():
    """Verifies that 1D/2D tensors (e.g., NLP or Linear layers) return 0.0 silently."""
    metric = SpatialAttentionMetric()
    features_2d = torch.randn(32, 128)

    score = metric.compute(features_2d, features_2d)
    assert score == 0.0
