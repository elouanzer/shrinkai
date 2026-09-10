import pytest
import torch

from shrinkai.distillation.losses import (
    AttentionMapLoss,
    FeatureLoss,
    GramMatrixLoss,
)

# ==========================================
# Fixtures
# ==========================================


@pytest.fixture
def sample_features():
    """Provides standard dummy student/teacher feature tensors [Batch, Channels, H, W]."""
    torch.manual_seed(42)
    batch_size, channels, h, w = 4, 16, 8, 8
    student_feat = torch.randn(batch_size, channels, h, w, requires_grad=True)
    teacher_feat = torch.randn(batch_size, channels, h, w)
    return student_feat, teacher_feat


@pytest.fixture
def sample_attention_maps():
    """Provides standard dummy attention map tensors [Batch, Heads, Seq_Len, Seq_Len]."""
    torch.manual_seed(42)
    batch_size, heads, seq_len = 4, 8, 32
    # Simulating raw attention scores (before softmax)
    student_attn = torch.randn(batch_size, heads, seq_len, seq_len, requires_grad=True)
    teacher_attn = torch.randn(batch_size, heads, seq_len, seq_len)
    return student_attn, teacher_attn


@pytest.fixture
def sample_feature_dicts(sample_features):
    """Provides dictionaries of feature tensors to test dict iteration."""
    s_feat, t_feat = sample_features
    # Create a second set of features for the dict
    s_feat2 = torch.randn_like(s_feat, requires_grad=True)
    t_feat2 = torch.randn_like(t_feat)

    student_dict = {"layer_1": s_feat, "layer_2": s_feat2}
    teacher_dict = {"layer_1": t_feat, "layer_2": t_feat2}
    return student_dict, teacher_dict


# ==========================================
# Tests for FeatureLoss
# ==========================================


def test_feature_loss_initialization():
    """Tests parameter validation during FeatureLoss initialization."""
    loss_mse = FeatureLoss(loss_type="mse", normalize=True)
    assert loss_mse.loss_type == "mse"
    assert loss_mse.normalize is True

    with pytest.raises(ValueError, match="Unsupported loss_type"):
        FeatureLoss(loss_type="kl")


@pytest.mark.parametrize("loss_type", ["mse", "l1", "cosine"])
def test_feature_loss_forward_types(sample_features, loss_type):
    """Ensures FeatureLoss computes correctly for all supported distance metrics."""
    s_feat, t_feat = sample_features
    loss_fn = FeatureLoss(loss_type=loss_type, normalize=False)

    loss = loss_fn(s_feat, t_feat)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_feature_loss_forward_dict(sample_feature_dicts):
    """Ensures FeatureLoss correctly averages losses across dictionary keys."""
    s_dict, t_dict = sample_feature_dicts
    loss_fn = FeatureLoss(loss_type="mse")

    loss = loss_fn(s_dict, t_dict)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_feature_loss_shape_mismatch():
    """Ensures an exception is raised when student and teacher features have different shapes."""
    loss_fn = FeatureLoss()
    s_feat = torch.randn(4, 16, 8, 8)
    t_feat = torch.randn(4, 32, 8, 8)  # Different channel size

    with pytest.raises(ValueError, match="Feature shape mismatch"):
        loss_fn(s_feat, t_feat)


def test_feature_loss_dict_key_mismatch(sample_feature_dicts):
    """Ensures an exception is raised when dict keys don't match."""
    s_dict, t_dict = sample_feature_dicts
    loss_fn = FeatureLoss()

    # Modify teacher keys
    bad_t_dict = {"layer_1": t_dict["layer_1"], "layer_wrong": t_dict["layer_2"]}

    with pytest.raises(ValueError, match="Dictionary key mismatch"):
        loss_fn(s_dict, bad_t_dict)


def test_feature_loss_gradients(sample_features):
    """Verifies that gradients are computed properly for the student model."""
    s_feat, t_feat = sample_features
    loss_fn = FeatureLoss(loss_type="mse")

    loss = loss_fn(s_feat, t_feat)
    loss.backward()

    assert s_feat.grad is not None
    assert not torch.isnan(s_feat.grad).any()


# ==========================================
# Tests for AttentionMapLoss
# ==========================================


def test_attention_loss_initialization():
    """Tests parameter validation for AttentionMapLoss."""
    loss_kl = AttentionMapLoss(loss_type="kl")
    assert loss_kl.loss_type == "kl"

    with pytest.raises(ValueError, match="Unsupported loss_type"):
        AttentionMapLoss(loss_type="cosine")


@pytest.mark.parametrize("loss_type", ["mse", "kl"])
def test_attention_loss_forward_types(sample_attention_maps, loss_type):
    """Ensures AttentionMapLoss computes correctly for both MSE and KL."""
    s_attn, t_attn = sample_attention_maps
    loss_fn = AttentionMapLoss(loss_type=loss_type)

    loss = loss_fn(s_attn, t_attn)
    loss.backward()

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert s_attn.grad is not None


def test_attention_loss_shape_mismatch():
    """Ensures an exception is raised when number of attention heads differ."""
    loss_fn = AttentionMapLoss()
    s_attn = torch.randn(2, 4, 32, 32)  # 4 heads
    t_attn = torch.randn(2, 8, 32, 32)  # 8 heads

    with pytest.raises(ValueError, match="Attention map shape mismatch"):
        loss_fn(s_attn, t_attn)


# ==========================================
# Tests for GramMatrixLoss
# ==========================================


def test_gram_loss_initialization():
    """Tests parameter validation for GramMatrixLoss."""
    loss_cosine = GramMatrixLoss(loss_type="cosine")
    assert loss_cosine.loss_type == "cosine"

    with pytest.raises(ValueError, match="Unsupported loss_type"):
        GramMatrixLoss(loss_type="kl")


@pytest.mark.parametrize("loss_type", ["mse", "l1", "cosine"])
def test_gram_loss_forward_types(sample_features, loss_type):
    """Ensures GramMatrixLoss computes correctly for all supported distance metrics."""
    s_feat, t_feat = sample_features
    loss_fn = GramMatrixLoss(loss_type=loss_type)

    loss = loss_fn(s_feat, t_feat)
    loss.backward()

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert s_feat.grad is not None


def test_gram_loss_spatial_invariance():
    """CRITICAL TEST: Gram matrix loss should work even if spatial dimensions differ."""
    loss_fn = GramMatrixLoss()
    # Student has 16x16 resolution
    s_feat = torch.randn(2, 8, 16, 16, requires_grad=True)
    # Teacher has 32x32 resolution
    t_feat = torch.randn(2, 8, 32, 32)

    # This should NOT raise an error because the channel dimension (8) matches.
    # The resulting Gram matrices will both be [2, 8, 8].
    loss = loss_fn(s_feat, t_feat)
    loss.backward()

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_gram_loss_channel_mismatch():
    """Ensures an exception is raised when the channel dimensions differ."""
    loss_fn = GramMatrixLoss()
    s_feat = torch.randn(2, 8, 16, 16)
    t_feat = torch.randn(2, 16, 16, 16)  # Different channel dimension

    with pytest.raises(ValueError, match="Channel dimension mismatch"):
        loss_fn(s_feat, t_feat)


def test_gram_loss_identical_features():
    """When student and teacher features are identical, MSE Gram Loss should be ~0."""
    feat = torch.randn(4, 16, 8, 8)
    loss_fn = GramMatrixLoss(loss_type="mse")

    loss = loss_fn(feat, feat)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
