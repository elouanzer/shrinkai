import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from shrinkai.distillation.engine import DistillationEngine
from shrinkai.distillation.losses.base import BaseDistillationLoss

# ==========================================
# 1. MOCK CLASSES
# ==========================================


class DummyModel(nn.Module):
    """A simple linear model for testing the engine."""

    def __init__(self, return_tuple=False):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.return_tuple = return_tuple

    def forward(self, x):
        logits = self.fc(x)
        if self.tuple_mode():
            return logits, {"layer1": x}
        return logits

    def tuple_mode(self):
        return getattr(self, "return_tuple", False)


class DummyLoss(BaseDistillationLoss):
    """A simple distillation loss computing cross entropy on logits."""

    def forward(self, student_outputs, teacher_outputs, labels=None):
        # Handle both raw tensors and tuples safely
        s_logits = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        return F.cross_entropy(s_logits, labels)


# ==========================================
# 2. FIXTURES
# ==========================================


@pytest.fixture
def mock_data():
    """Provides a basic dataloader with 20 samples (batch_size=10)."""
    inputs = torch.randn(20, 10)
    labels = torch.randint(0, 2, (20,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def engine_setup():
    """Provides instantiated components for the engine."""
    student = DummyModel()
    teacher = DummyModel()
    criterion = DummyLoss()
    optimizer = SGD(student.parameters(), lr=0.1)
    return student, teacher, criterion, optimizer


# ==========================================
# 3. ENGINE TESTS
# ==========================================


def test_engine_initialization_freezes_teacher(engine_setup):
    """Verifies that the engine automatically freezes teacher weights and sets eval mode."""
    student, teacher, criterion, optimizer = engine_setup

    # Ensure teacher requires grad initially
    for param in teacher.parameters():
        param.requires_grad = True

    _ = DistillationEngine(student, teacher, criterion, optimizer, device="cpu")

    # Teacher should be frozen and in eval mode
    assert not teacher.training
    for param in teacher.parameters():
        assert not param.requires_grad

    # Student should remain trainable
    for param in student.parameters():
        assert param.requires_grad


def test_train_epoch_updates_student_only(engine_setup, mock_data):
    """Verifies that a training epoch updates the student's weights but not the teacher's."""
    student, teacher, criterion, optimizer = engine_setup
    engine = DistillationEngine(student, teacher, criterion, optimizer, device="cpu")

    # Save initial weights to check for modifications
    initial_student_weights = student.fc.weight.clone()
    initial_teacher_weights = teacher.fc.weight.clone()

    metrics = engine.train_epoch(mock_data, epoch_idx=1, total_epochs=1)

    assert "loss" in metrics
    assert "accuracy" in metrics

    # Student weights should have changed
    assert not torch.allclose(student.fc.weight, initial_student_weights)

    # Teacher weights should be strictly identical
    assert torch.allclose(teacher.fc.weight, initial_teacher_weights)


def test_evaluate(engine_setup, mock_data):
    """Verifies that the evaluation loop computes validation metrics correctly."""
    student, teacher, criterion, optimizer = engine_setup
    engine = DistillationEngine(student, teacher, criterion, optimizer, device="cpu")

    metrics = engine.evaluate(mock_data)

    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 100.0


def test_fit_with_scheduler_and_callbacks(engine_setup, mock_data):
    """Verifies the complete fit loop handles callbacks, schedulers, and history properly."""
    student, teacher, criterion, optimizer = engine_setup
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    engine = DistillationEngine(
        student, teacher, criterion, optimizer, device="cpu", scheduler=scheduler
    )

    initial_lr = optimizer.param_groups[0]["lr"]

    # Simple callback to track calls
    callback_calls = []

    def dummy_callback(epoch, metrics):
        callback_calls.append(epoch)

    history = engine.fit(
        train_dataloader=mock_data,
        val_dataloader=mock_data,
        epochs=2,
        callbacks=[dummy_callback],
    )

    # History should contain exactly 2 epochs of data
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2

    # Callbacks should have been triggered twice (epoch 1 and 2)
    assert callback_calls == [1, 2]

    # Scheduler should have stepped twice, reducing the LR
    current_lr = optimizer.param_groups[0]["lr"]
    assert current_lr < initial_lr


def test_engine_handles_tuple_outputs(mock_data):
    """Verifies that the engine can process tuples generated by FeatureExtractors seamlessly."""
    student = DummyModel(return_tuple=True)
    teacher = DummyModel(return_tuple=True)
    criterion = DummyLoss()
    optimizer = SGD(student.parameters(), lr=0.1)

    engine = DistillationEngine(student, teacher, criterion, optimizer, device="cpu")

    # This should not raise an error when calculating accuracy (argmax on tuples)
    try:
        metrics = engine.train_epoch(mock_data, epoch_idx=1, total_epochs=1)
        assert "loss" in metrics
    except Exception as e:
        pytest.fail(f"Engine failed to handle tuple outputs: {e}")
