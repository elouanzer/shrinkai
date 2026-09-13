from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shrinkai.adapters import FeatureExtractor
from shrinkai.distillation.distiller import Distiller
from shrinkai.distillation.engine import DistillationEngine
from shrinkai.distillation.losses.base import BaseDistillationLoss
from shrinkai.distillation.losses.logits import HintonLoss

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


# ==========================================
# 7. MIXED PRECISION / GRADIENT CLIPPING WIRING
# ==========================================


def test_distiller_forwards_amp_and_grad_clip_to_engine(mock_data):
    """Verifies use_amp / grad_clip_norm are passed down to the DistillationEngine."""
    distiller = Distiller(
        teacher=DummyModel(),
        student=DummyModel(),
        optimizer="sgd",
        device="cpu",
        use_amp=True,
        grad_clip_norm=1.0,
    )

    assert distiller._engine.use_amp is True
    assert distiller._engine.grad_clip_norm == 1.0

    # End-to-end sanity check: training must still run fine with both enabled.
    history = distiller.fit(mock_data, epochs=1)
    assert len(history["train_loss"]) == 1


# ==========================================
# 8. CHECKPOINT / RESUME TESTS
# ==========================================


def test_fit_resume_continues_from_last_epoch(base_distiller, mock_data):
    """Verifies fit(resume=True) continues training instead of restarting at epoch 1."""
    base_distiller.fit(mock_data, epochs=2)
    assert len(base_distiller.history["train_loss"]) == 2

    base_distiller.fit(mock_data, epochs=5, resume=True)
    assert len(base_distiller.history["train_loss"]) == 5


def test_fit_without_resume_restarts_history(base_distiller, mock_data):
    """Verifies the default (resume=False) still restarts a fresh history each call."""
    base_distiller.fit(mock_data, epochs=2)
    assert len(base_distiller.history["train_loss"]) == 2

    base_distiller.fit(mock_data, epochs=1)
    assert len(base_distiller.history["train_loss"]) == 1


def test_save_and_load_checkpoint_roundtrip(mock_data, tmp_path: Path):
    """Verifies a checkpoint restores student weights, optimizer and history state."""
    source = Distiller(teacher=DummyModel(), student=DummyModel(), optimizer="sgd", device="cpu")
    source.fit(mock_data, epochs=2)

    checkpoint_path = tmp_path / "checkpoint.pt"
    source.save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()

    target = Distiller(teacher=DummyModel(), student=DummyModel(), optimizer="sgd", device="cpu")
    target.load_checkpoint(checkpoint_path)

    # Student weights restored exactly.
    assert torch.allclose(target.student.fc.weight, source.student.fc.weight)
    # Optimizer state restored (momentum buffers / step counts populated by training).
    target_state_keys = target.optimizer.state_dict()["state"].keys()
    source_state_keys = source.optimizer.state_dict()["state"].keys()
    assert target_state_keys == source_state_keys
    # History restored, so resuming continues from epoch 3.
    assert target.history == source.history

    target.fit(mock_data, epochs=3, resume=True)
    assert len(target.history["train_loss"]) == 3


def test_load_checkpoint_raises_on_scheduler_mismatch(mock_data, tmp_path: Path):
    """Verifies a scheduler-configuration mismatch between save/load raises clearly."""
    from torch.optim.lr_scheduler import StepLR

    with_scheduler = Distiller(teacher=DummyModel(), student=DummyModel(), optimizer="sgd")
    with_scheduler.scheduler = StepLR(with_scheduler.optimizer, step_size=1)
    with_scheduler._engine.scheduler = with_scheduler.scheduler

    checkpoint_path = tmp_path / "with_scheduler.pt"
    with_scheduler.save_checkpoint(checkpoint_path)

    without_scheduler = Distiller(teacher=DummyModel(), student=DummyModel(), optimizer="sgd")
    with pytest.raises(ValueError, match="no scheduler configured"):
        without_scheduler.load_checkpoint(checkpoint_path)

    no_scheduler_checkpoint = tmp_path / "no_scheduler.pt"
    without_scheduler.save_checkpoint(no_scheduler_checkpoint)
    with pytest.raises(ValueError, match="saved without one"):
        with_scheduler.load_checkpoint(no_scheduler_checkpoint)


# ==========================================
# 9. EXPORT
# ==========================================


def test_export_onnx_delegates_to_student(base_distiller, tmp_path: Path):
    """Verifies export_onnx exports self.student, not the teacher."""
    import onnx

    out_path = tmp_path / "student.onnx"
    sample_input = torch.randn(1, 10)

    result = base_distiller.export_onnx(out_path, sample_input)

    assert result == out_path
    assert out_path.exists()
    onnx.checker.check_model(onnx.load(str(out_path)))


def test_export_torchscript_delegates_to_student(base_distiller, tmp_path: Path):
    """Verifies export_torchscript exports self.student and round-trips correctly."""
    out_path = tmp_path / "student.pt"
    sample_input = torch.randn(1, 10)

    base_distiller.export_torchscript(out_path, sample_input=sample_input, method="trace")

    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        expected = base_distiller.student(sample_input)
        actual = loaded(sample_input)
    assert torch.allclose(expected, actual)


# ==========================================
# 10. FLOPS IN BENCHMARK
# ==========================================


@patch("shrinkai.distillation.distiller.Profiler.compare")
def test_benchmark_forwards_compute_flops(mock_compare, base_distiller):
    """Verifies compute_flops is forwarded to Profiler.compare."""
    sample_input = torch.randn(1, 10)

    base_distiller.benchmark(sample_input, compute_flops=True)

    kwargs = mock_compare.call_args.kwargs
    assert kwargs["compute_flops"] is True


# ==========================================
# 11. CUSTOM ENGINE CLASS
# ==========================================


class RecordingEngine(DistillationEngine):
    """Minimal custom engine used to verify `engine_class` wiring end-to-end."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_epoch_calls = 0

    def train_epoch(self, dataloader, epoch_idx, total_epochs):
        self.train_epoch_calls += 1
        return super().train_epoch(dataloader, epoch_idx, total_epochs)


def test_distiller_defaults_to_base_engine_class():
    """Verifies that omitting engine_class keeps the plain DistillationEngine."""
    distiller = Distiller(teacher=DummyModel(), student=DummyModel(), optimizer="sgd", device="cpu")

    assert type(distiller._engine) is DistillationEngine


def test_distiller_uses_custom_engine_class(mock_data):
    """Verifies engine_class is instantiated instead of the default DistillationEngine,
    and that it receives the exact criterion/optimizer Distiller built for it.
    """
    criterion = HintonLoss()
    distiller = Distiller(
        teacher=DummyModel(),
        student=DummyModel(),
        criterion=criterion,
        optimizer="sgd",
        device="cpu",
        engine_class=RecordingEngine,
    )

    assert isinstance(distiller._engine, RecordingEngine)
    assert distiller._engine.criterion is criterion
    assert distiller._engine.optimizer is distiller.optimizer

    history = distiller.fit(mock_data, epochs=2)

    assert distiller._engine.train_epoch_calls == 2
    assert len(history["train_loss"]) == 2
