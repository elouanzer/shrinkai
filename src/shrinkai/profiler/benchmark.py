from dataclasses import dataclass

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from ..utils import resolve_device
from .latency import measure_latency
from .memory import count_parameters, estimate_model_size_mb


@dataclass
class ModelProfile:
    """Container holding profiled metrics for an individual model."""

    name: str
    total_params: int
    size_mb: float
    sample_latency_ms: float
    fps: float
    accuracy: float | None = None


class BenchmarkReport:
    """Holds comparison results between Teacher and Student models."""

    def __init__(self, teacher_profile: ModelProfile, student_profile: ModelProfile) -> None:
        self.teacher = teacher_profile
        self.student = student_profile

    def _compute_gains(self) -> dict[str, str]:
        """Calculates percentage and factor improvements."""
        if self.teacher.total_params > 0:
            param_reduction = (1 - self.student.total_params / self.teacher.total_params) * 100
        else:
            param_reduction = 0.0

        if self.teacher.size_mb > 0:
            size_reduction = (1 - self.student.size_mb / self.teacher.size_mb) * 100
        else:
            size_reduction = 0.0

        size_multiplier = (
            self.teacher.size_mb / self.student.size_mb if self.student.size_mb > 0 else 1.0
        )
        speedup = self.student.fps / self.teacher.fps if self.teacher.fps > 0 else 1.0

        return {
            "param_reduction": f"-{param_reduction:.1f}%",
            "size_reduction": f"-{size_reduction:.1f}% ({size_multiplier:.1f}x smaller)",
            "speedup": f"+{speedup:.1f}x ({self.student.fps:.1f} FPS)",
        }

    def show(self) -> None:
        """Renders an interactive formatted comparison table in the console."""
        console = Console()
        gains = self._compute_gains()

        table = Table(title="Distillation Benchmark Report", header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column(f"Teacher ({self.teacher.name})", justify="right")
        table.add_column(f"Student ({self.student.name})", justify="right")
        table.add_column("Gain / Compression", justify="right", style="green")

        table.add_row(
            "Parameters",
            f"{self.teacher.total_params / 1e6:.2f} M",
            f"{self.student.total_params / 1e6:.2f} M",
            gains["param_reduction"],
        )
        table.add_row(
            "Model Size (Disk)",
            f"{self.teacher.size_mb:.2f} MB",
            f"{self.student.size_mb:.2f} MB",
            gains["size_reduction"],
        )
        latency_speedup = (
            self.teacher.sample_latency_ms / self.student.sample_latency_ms
            if self.student.sample_latency_ms > 0
            else 1.0
        )
        table.add_row(
            "Latency / Sample",
            f"{self.teacher.sample_latency_ms:.2f} ms",
            f"{self.student.sample_latency_ms:.2f} ms",
            f"{latency_speedup:.1f}x faster",
        )
        table.add_row(
            "Throughput (FPS)",
            f"{self.teacher.fps:.1f} img/s",
            f"{self.student.fps:.1f} img/s",
            gains["speedup"],
        )

        if self.teacher.accuracy is not None and self.student.accuracy is not None:
            retention = (
                (self.student.accuracy / self.teacher.accuracy) * 100
                if self.teacher.accuracy > 0
                else 0.0
            )
            table.add_row(
                "Accuracy",
                f"{self.teacher.accuracy:.2f}%",
                f"{self.student.accuracy:.2f}%",
                f"{retention:.1f}% retained",
            )

        console.print(table)


class Profiler:
    """Benchmark runner comparing Teacher and Student architectures."""

    @staticmethod
    def compare(
        teacher: nn.Module,
        student: nn.Module,
        sample_input: torch.Tensor,
        device: torch.device | str = "auto",
        teacher_name: str = "Teacher",
        student_name: str = "Student",
        teacher_acc: float | None = None,
        student_acc: float | None = None,
    ) -> BenchmarkReport:
        """Executes complete profiling suite on both models and generates comparison.

        Args:
            teacher: Teacher PyTorch model.
            student: Student PyTorch model.
            sample_input: Representative tensor input batch.
            device: Device target ('auto', 'mps', 'cuda', 'cpu').
            teacher_name: Display label for teacher.
            student_name: Display label for student.
            teacher_acc: Optional pre-computed teacher accuracy.
            student_acc: Optional pre-computed student accuracy.

        Returns:
            BenchmarkReport: Structured report ready for `.show()`.
        """
        resolved_device = resolve_device(device)

        t_params = count_parameters(teacher)["total_params"]
        t_size = estimate_model_size_mb(teacher)
        t_lat = measure_latency(teacher, sample_input, resolved_device)
        teacher_profile = ModelProfile(
            name=teacher_name,
            total_params=t_params,
            size_mb=t_size,
            sample_latency_ms=t_lat["sample_latency_ms"],
            fps=t_lat["fps"],
            accuracy=teacher_acc,
        )

        s_params = count_parameters(student)["total_params"]
        s_size = estimate_model_size_mb(student)
        s_lat = measure_latency(student, sample_input, resolved_device)
        student_profile = ModelProfile(
            name=student_name,
            total_params=s_params,
            size_mb=s_size,
            sample_latency_ms=s_lat["sample_latency_ms"],
            fps=s_lat["fps"],
            accuracy=student_acc,
        )

        return BenchmarkReport(teacher_profile, student_profile)
