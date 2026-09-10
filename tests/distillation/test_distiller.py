from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shrinkai.adapters import FeatureExtractor
from shrinkai.distillation.distiller import Distiller
from shrinkai.distillation.losses.base import BaseDistillationLoss

# ==========================================
# 1. MOCK CLASSES & FIXTURES
# ==========================================


class DummyModel(nn.Module):
    """A minimal PyTorch model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class DummyLossWithParams(BaseDistillationLoss):
    """A mock loss that includes trainable parameters (simulating a Projector)."""

    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(10, 10)

    def forward(self, student_outputs, teacher_outputs, labels=None):
        return torch.tensor(1.0, requires_grad=True)


@pytest.fixture
def mock_data():
    """Provides a basic dataloader for testing."""
    inputs = torch.randn(20, 10)
    labels = torch.randint(0, 2, (20,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def base_distiller():
    """Provides a basic initialized Distiller."""
    return Distiller(teacher=DummyModel(), student=DummyModel(), optimizer="sgd", device="cpu")


# ==========================================
# 2. INITIALIZATION & OPTIMIZER TESTS
# ==========================================


def test_distiller_resolves_string_optimizers():
    """Verifies that the distiller correctly builds string-based optimizers."""
    teacher, student = DummyModel(), DummyModel()

    dist_adam = Distiller(teacher, student, optimizer="adam")
    assert isinstance(dist_adam.optimizer, torch.optim.Adam)

    dist_adamw = Distiller(teacher, student, optimizer="adamw")
    assert isinstance(dist_adamw.optimizer, torch.optim.AdamW)

    dist_sgd = Distiller(teacher, student, optimizer="sgd")
    assert isinstance(dist_sgd.optimizer, torch.optim.SGD)


def test_distiller_rejects_invalid_optimizer():
    """Verifies that an unsupported optimizer string raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        Distiller(DummyModel(), DummyModel(), optimizer="rmsprop")


def test_distiller_accepts_instantiated_optimizer():
    """Verifies that the distiller accepts an already instantiated optimizer."""
    student = DummyModel()
    custom_opt = torch.optim.RMSprop(student.parameters(), lr=0.01)

    distiller = Distiller(DummyModel(), student, optimizer=custom_opt)
    assert distiller.optimizer is custom_opt


def test_distiller_includes_criterion_parameters():
    """Verifies that projector parameters from the loss are added to the optimizer."""
    student = DummyModel()  # 1 weight tensor, 1 bias tensor = 2 parameters
    criterion = DummyLossWithParams()  # 1 weight tensor, 1 bias tensor = 2 parameters

    distiller = Distiller(
        teacher=DummyModel(), student=student, criterion=criterion, optimizer="adam"
    )

    # Optimizer should track both student params (2) and criterion params (2)
    tracked_params = distiller.optimizer.param_groups[0]["params"]
    assert len(tracked_params) == 4


# ==========================================
# 3. ENGINE DELEGATION TESTS (Fit & Evaluate)
# ==========================================


def test_distiller_fit_delegation(base_distiller, mock_data):
    """Verifies that .fit() properly delegates to the underlying DistillationEngine."""
    # We don't mock the engine here to ensure the full pipeline runs without crashing
    history = base_distiller.fit(mock_data, epochs=1)

    assert "train_loss" in history
    assert "train_accuracy" in history
    assert len(history["train_loss"]) == 1


def test_distiller_evaluate_delegation(base_distiller, mock_data):
    """Verifies that .evaluate() correctly delegates to the engine."""
    metrics = base_distiller.evaluate(mock_data)

    assert "val_loss" in metrics
    assert "val_accuracy" in metrics


# ==========================================
# 4. FEATURE ANALYSIS TESTS
# ==========================================


def test_feature_analysis_raises_on_raw_models(base_distiller, mock_data):
    """Verifies that feature analysis is blocked if models aren't FeatureExtractors."""
    with pytest.raises(
        ValueError, match="requires both models to be wrapped with `FeatureExtractor`"
    ):
        base_distiller.feature_analysis(mock_data)


@patch("shrinkai.distillation.distiller.FeatureAnalyzer")
def test_feature_analysis_success(MockAnalyzer, mock_data):
    """Verifies that feature analysis executes successfully when conditions are met."""
    # Use MagicMock with spec to spoof isinstance(obj, FeatureExtractor)
    mock_teacher = MagicMock(spec=FeatureExtractor)
    mock_student = MagicMock(spec=FeatureExtractor)

    dummy_param = nn.Parameter(torch.tensor([1.0]))
    mock_student.parameters.return_value = [dummy_param]
    mock_teacher.parameters.return_value = []

    distiller = Distiller(teacher=mock_teacher, student=mock_student)

    # Call the method
    distiller.feature_analysis(mock_data, metrics=["cka", "rsa"])

    # Assert Analyzer was initialized with the mock extractors
    MockAnalyzer.assert_called_once_with(
        teacher_extractor=mock_teacher,
        student_extractor=mock_student,
        device=distiller.device,
    )

    # Assert evaluate was called on the analyzer instance
    analyzer_instance = MockAnalyzer.return_value
    analyzer_instance.evaluate.assert_called_once_with(dataloader=mock_data, metrics=["cka", "rsa"])


# ==========================================
# 5. BENCHMARK TESTS
# ==========================================


@patch("shrinkai.distillation.distiller.Profiler.compare")
def test_benchmark_without_dataloader(mock_compare, base_distiller):
    """Verifies benchmark delegates to Profiler without accuracy computation."""
    sample_input = torch.randn(1, 10)

    base_distiller.benchmark(sample_input, teacher_name="T", student_name="S")

    # Verify Profiler.compare was called with accuracy=None
    mock_compare.assert_called_once()
    kwargs = mock_compare.call_args.kwargs
    assert kwargs["teacher_acc"] is None
    assert kwargs["student_acc"] is None
    assert kwargs["teacher_name"] == "T"
    assert kwargs["student_name"] == "S"


@patch("shrinkai.distillation.distiller.Profiler.compare")
def test_benchmark_with_dataloader(mock_compare, base_distiller, mock_data):
    """Verifies benchmark computes accuracy before delegating to Profiler."""
    sample_input = torch.randn(1, 10)

    base_distiller.benchmark(sample_input, val_dataloader=mock_data)

    # Verify Profiler.compare was called with computed accuracies
    mock_compare.assert_called_once()
    kwargs = mock_compare.call_args.kwargs

    # Accuracy should now be floats, not None
    assert isinstance(kwargs["teacher_acc"], float)
    assert isinstance(kwargs["student_acc"], float)
    assert 0.0 <= kwargs["teacher_acc"] <= 100.0


# ==========================================
# 6. SAVE & LOAD TESTS
# ==========================================


def test_save_and_load_student(base_distiller, tmp_path: Path):
    """Verifies that student weights can be correctly saved to and loaded from disk."""
    save_file = tmp_path / "student_model.pt"

    # Record initial weights
    initial_weights = base_distiller.student.fc.weight.clone()

    # Modify weights artificially to simulate training
    with torch.no_grad():
        base_distiller.student.fc.weight.add_(1.0)

    modified_weights = base_distiller.student.fc.weight.clone()
    assert not torch.allclose(initial_weights, modified_weights)

    # Save the modified weights
    base_distiller.save_student(save_file)
    assert save_file.exists()

    # Reset model to initial weights
    with torch.no_grad():
        base_distiller.student.fc.weight.copy_(initial_weights)

    # Load from disk
    base_distiller.load_student(save_file)

    # Verify weights have been restored to the modified version
    restored_weights = base_distiller.student.fc.weight
    assert torch.allclose(restored_weights, modified_weights)
