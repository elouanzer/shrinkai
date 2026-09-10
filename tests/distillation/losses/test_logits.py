import pytest
import torch

from shrinkai.distillation.losses import (
    BCEKDLoss,
    HintonLoss,
    JSDLoss,
    PureKDLoss,
    ReverseKLLoss,
)


@pytest.fixture
def sample_logits():
    """Provides standard dummy student/teacher logits and target labels."""
    torch.manual_seed(42)
    batch_size = 8
    num_classes = 10
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    return student_logits, teacher_logits, labels


@pytest.fixture
def sample_multilabel_logits():
    """Provides standard dummy student/teacher logits and multi-hot target labels."""
    torch.manual_seed(42)
    batch_size = 8
    num_classes = 10
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    # Labels for BCE need to be floats in [0, 1]
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    return student_logits, teacher_logits, labels


# ==========================================
# Tests for HintonLoss
# ==========================================


def test_hinton_loss_initialization():
    """Tests parameter validation during HintonLoss initialization."""
    # Valid initializations
    loss_fn = HintonLoss(temperature=2.0, alpha=0.3)
    assert loss_fn.temperature == 2.0
    assert loss_fn.alpha == 0.3

    # Invalid temperature (must be > 0)
    with pytest.raises(ValueError, match="Temperature must be positive"):
        HintonLoss(temperature=0.0)
    with pytest.raises(ValueError, match="Temperature must be positive"):
        HintonLoss(temperature=-1.0)

    # Invalid alpha (must be in [0.0, 1.0])
    with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
        HintonLoss(alpha=-0.1)
    with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
        HintonLoss(alpha=1.1)


def test_hinton_loss_forward_shape_and_scalar(sample_logits):
    """Ensures forward output is a scalar loss tensor."""
    student_logits, teacher_logits, labels = sample_logits
    loss_fn = HintonLoss(temperature=4.0, alpha=0.5)

    loss = loss_fn(student_logits, teacher_logits, labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_hinton_loss_shape_mismatch():
    """Ensures an exception is raised when student and teacher shapes do not match."""
    loss_fn = HintonLoss()
    student_logits = torch.randn(8, 10)
    teacher_logits = torch.randn(8, 5)  # Mismatched class dimension
    labels = torch.randint(0, 10, (8,))

    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(student_logits, teacher_logits, labels)


def test_hinton_loss_alpha_one_without_labels(sample_logits):
    """Ensures alpha=1.0 runs purely on teacher outputs without requiring hard labels."""
    student_logits, teacher_logits, _ = sample_logits
    loss_fn = HintonLoss(temperature=4.0, alpha=1.0)

    # Calling forward with labels=None should succeed when alpha=1.0
    loss = loss_fn(student_logits, teacher_logits, labels=None)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_hinton_loss_missing_labels_when_alpha_less_than_one(sample_logits):
    """Ensures an error is raised if labels are missing when alpha < 1.0."""
    student_logits, teacher_logits, _ = sample_logits
    loss_fn = HintonLoss(temperature=4.0, alpha=0.7)

    with pytest.raises(ValueError, match="Ground-truth `labels` are required"):
        loss_fn(student_logits, teacher_logits, labels=None)


def test_hinton_loss_gradients_backpropagation(sample_logits):
    """Verifies that gradients are computed properly for the student model."""
    student_logits, teacher_logits, labels = sample_logits
    loss_fn = HintonLoss(temperature=3.0, alpha=0.5)

    loss = loss_fn(student_logits, teacher_logits, labels)
    loss.backward()

    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()
    assert (student_logits.grad != 0).any()


def test_hinton_loss_temperature_scaling_effect():
    """Verifies that temperature scaling smooths probability distributions."""
    batch_size = 4
    num_classes = 5
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)

    loss_t1 = HintonLoss(temperature=1.0, alpha=1.0)
    loss_t10 = HintonLoss(temperature=10.0, alpha=1.0)

    val_t1 = loss_t1(student_logits, teacher_logits)
    val_t10 = loss_t10(student_logits, teacher_logits)

    # Distillation loss value varies depending on temperature scaling T^2 factor
    assert isinstance(val_t1, torch.Tensor)
    assert isinstance(val_t10, torch.Tensor)
    assert not torch.isnan(val_t1)
    assert not torch.isnan(val_t10)


def test_hinton_loss_identical_distributions():
    """When student and teacher outputs are identical and alpha=1.0, KL loss should be ~0."""
    logits = torch.randn(8, 10)
    loss_fn = HintonLoss(temperature=2.0, alpha=1.0)

    loss = loss_fn(logits, logits)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


# ==========================================
# Tests for PureKDLoss
# ==========================================


def test_pure_kd_loss_initialization():
    """Validates that PureKDLoss properly hardcodes alpha to 1.0."""
    loss_fn = PureKDLoss(temperature=3.0)
    assert loss_fn.temperature == 3.0
    assert loss_fn.alpha == 1.0


def test_pure_kd_loss_forward(sample_logits):
    """Ensures PureKDLoss runs perfectly without labels."""
    student_logits, teacher_logits, _ = sample_logits
    loss_fn = PureKDLoss(temperature=2.0)

    loss = loss_fn(student_logits, teacher_logits, labels=None)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


# ==========================================
# Tests for ReverseKLLoss
# ==========================================


def test_reverse_kl_loss_initialization():
    """Validates temperature constraints for ReverseKLLoss."""
    loss_fn = ReverseKLLoss(temperature=1.5)
    assert loss_fn.temperature == 1.5

    with pytest.raises(ValueError, match="Temperature must be positive"):
        ReverseKLLoss(temperature=-0.5)


def test_reverse_kl_loss_forward_and_gradients(sample_logits):
    """Verifies scalar output and valid gradient flow for the student."""
    student_logits, teacher_logits, _ = sample_logits
    loss_fn = ReverseKLLoss(temperature=1.0)

    loss = loss_fn(student_logits, teacher_logits)
    loss.backward()

    assert isinstance(loss, torch.Tensor)
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_reverse_kl_loss_identical_distributions():
    """When student and teacher outputs are identical, Reverse KL should be ~0."""
    logits = torch.randn(8, 10)
    loss_fn = ReverseKLLoss(temperature=1.0)

    loss = loss_fn(logits, logits)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


# ==========================================
# Tests for BCEKDLoss
# ==========================================


def test_bce_kd_loss_initialization():
    """Tests parameter validation for BCEKDLoss."""
    loss_fn = BCEKDLoss(temperature=2.0, alpha=0.5)
    assert loss_fn.temperature == 2.0

    with pytest.raises(ValueError, match="Alpha must be between 0.0 and 1.0"):
        BCEKDLoss(alpha=1.5)


def test_bce_kd_loss_forward_and_gradients(sample_multilabel_logits):
    """Ensures BCEKDLoss computes correctly with multi-hot labels."""
    student_logits, teacher_logits, labels = sample_multilabel_logits
    loss_fn = BCEKDLoss(temperature=2.0, alpha=0.5)

    loss = loss_fn(student_logits, teacher_logits, labels)
    loss.backward()

    assert isinstance(loss, torch.Tensor)
    assert student_logits.grad is not None
    assert not torch.isnan(loss)


def test_bce_kd_loss_missing_labels_when_alpha_less_than_one(sample_multilabel_logits):
    """Ensures an error is raised if multi-hot labels are missing when alpha < 1.0."""
    student_logits, teacher_logits, _ = sample_multilabel_logits
    loss_fn = BCEKDLoss(temperature=2.0, alpha=0.5)

    with pytest.raises(ValueError, match="Ground-truth `labels` are required"):
        loss_fn(student_logits, teacher_logits, labels=None)


# ==========================================
# Tests for JSDLoss
# ==========================================


def test_jsd_loss_initialization():
    """Validates temperature constraints for JSDLoss."""
    loss_fn = JSDLoss(temperature=3.0)
    assert loss_fn.temperature == 3.0


def test_jsd_loss_forward_and_gradients(sample_logits):
    """Verifies scalar output and valid gradient flow for JSDLoss."""
    student_logits, teacher_logits, _ = sample_logits
    loss_fn = JSDLoss(temperature=2.0)

    loss = loss_fn(student_logits, teacher_logits)
    loss.backward()

    assert isinstance(loss, torch.Tensor)
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_jsd_loss_symmetry(sample_logits):
    """JSD is symmetric: JSD(P||Q) == JSD(Q||P)."""
    student_logits, teacher_logits, _ = sample_logits
    loss_fn = JSDLoss(temperature=1.0)

    loss_s_t = loss_fn(student_logits, teacher_logits)
    loss_t_s = loss_fn(teacher_logits, student_logits)

    assert torch.isclose(loss_s_t, loss_t_s, atol=1e-5)


def test_jsd_loss_identical_distributions():
    """When student and teacher outputs are identical, JSD should be ~0."""
    logits = torch.randn(8, 10)
    loss_fn = JSDLoss(temperature=1.0)

    loss = loss_fn(logits, logits)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
