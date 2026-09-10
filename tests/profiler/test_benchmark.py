from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from shrinkai.profiler.benchmark import BenchmarkReport, ModelProfile, Profiler

# ==========================================
# BENCHMARK REPORT TESTS
# ==========================================


@pytest.fixture
def dummy_profiles():
    """Provides fake teacher and student profiles for math verification."""
    teacher = ModelProfile("Teacher", 1000, 10.0, 50.0, 20.0, 95.0)
    # Student is half the params, 1/5th the size, twice as fast, slightly less accurate
    student = ModelProfile("Student", 500, 2.0, 25.0, 40.0, 93.0)
    return teacher, student


def test_benchmark_report_compute_gains(dummy_profiles):
    """Verifies the compression percentages and speedup factors."""
    teacher, student = dummy_profiles
    report = BenchmarkReport(teacher, student)

    gains = report._compute_gains()

    # Param reduction: (1 - 500/1000)*100 = 50.0%
    assert gains["param_reduction"] == "-50.0%"

    # Size reduction: (1 - 2.0/10.0)*100 = 80.0%, 10.0/2.0 = 5.0x
    assert gains["size_reduction"] == "-80.0% (5.0x smaller)"

    # Speedup: 40.0 FPS / 20.0 FPS = 2.0x
    assert gains["speedup"] == "+2.0x (40.0 FPS)"


def test_benchmark_report_show(dummy_profiles):
    """Verifies that rendering the rich table doesn't crash."""
    teacher, student = dummy_profiles
    report = BenchmarkReport(teacher, student)

    try:
        report.show()
    except Exception as e:
        pytest.fail(f"BenchmarkReport.show() crashed with: {e}")


# ==========================================
# PROFILER COMPARE TESTS
# ==========================================


@patch("shrinkai.profiler.benchmark.count_parameters")
@patch("shrinkai.profiler.benchmark.estimate_model_size_mb")
@patch("shrinkai.profiler.benchmark.measure_latency")
def test_profiler_compare(mock_latency, mock_size, mock_params):
    """Verifies that Profiler.compare gathers data correctly without running real inference."""
    # Set up mock returns
    mock_params.side_effect = [{"total_params": 100}, {"total_params": 50}]
    mock_size.side_effect = [10.0, 5.0]
    mock_latency.side_effect = [
        {"sample_latency_ms": 2.0, "fps": 500.0},
        {"sample_latency_ms": 1.0, "fps": 1000.0},
    ]

    teacher = nn.Linear(10, 10)
    student = nn.Linear(10, 5)
    sample_input = torch.randn(1, 10)

    report = Profiler.compare(
        teacher=teacher,
        student=student,
        sample_input=sample_input,
        device="cpu",
        teacher_acc=98.0,
        student_acc=96.0,
    )

    # Assert correct routing to profiles
    assert report.teacher.total_params == 100
    assert report.student.size_mb == 5.0
    assert report.student.fps == 1000.0
    assert report.teacher.accuracy == 98.0
    assert report.student.accuracy == 96.0

    # Ensure latency function was called exactly twice (once for each model)
    assert mock_latency.call_count == 2
