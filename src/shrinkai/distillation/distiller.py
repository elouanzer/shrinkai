from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..adapters import FeatureExtractor
from ..analysis import FeatureAnalyzer, FeatureAnalyzerReport
from ..profiler.accuracy import compute_accuracy
from ..profiler.benchmark import BenchmarkReport, Profiler
from ..utils import resolve_device
from .engine import DistillationEngine
from .losses.base import BaseDistillationLoss
from .losses.logits import HintonLoss


class Distiller:
    """Unified, high-level facade for end-to-end knowledge distillation and benchmarking.

    Examples:
        >>> from shrinkai.distillation import Distiller
        >>> distiller = Distiller(teacher=teacher_vit, student=student_mobilenet)
        >>> distiller.fit(train_loader, val_loader, epochs=5)
        >>> distiller.save_student("distilled_student.pt")
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        criterion: BaseDistillationLoss | None = None,
        optimizer: torch.optim.Optimizer | Literal["adam", "adamw", "sgd"] = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device | str = "auto",
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """Initializes the Distiller.

        Args:
            teacher: Pre-trained teacher neural network module.
            student: Target lightweight student neural network module to train.
            criterion: Distillation loss adhering to `BaseDistillationLoss`.
                Defaults to `HintonLoss()`.
            optimizer: PyTorch optimizer or name string ('adam', 'adamw', 'sgd').
                Defaults to 'adamw'.
            lr: Learning rate applied when creating default optimizer. Defaults to 1e-3.
            weight_decay: Weight decay factor for optimizer. Defaults to 1e-4.
            device: Computing target ('auto', 'mps', 'cuda', 'cpu' or torch.device).
                Defaults to 'auto'.
            scheduler: Optional learning rate scheduler updated per epoch.
        """
        self.teacher = teacher
        self.student = student
        self.device = resolve_device(device)
        self.criterion = criterion if criterion is not None else HintonLoss()
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.criterion.to(self.device)

        if isinstance(optimizer, str):
            self.optimizer = self._build_optimizer(
                opt_name=optimizer.lower(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        self._engine = DistillationEngine(
            student=self.student,
            teacher=self.teacher,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            scheduler=self.scheduler,
        )

    def _build_optimizer(
        self,
        opt_name: str,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Builds standard optimizer for student parameters."""
        params = [p for p in self.student.parameters() if p.requires_grad]
        params += [p for p in self.criterion.parameters() if p.requires_grad]

        if opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        raise ValueError(
            f"Unsupported optimizer '{opt_name}'. Choose from 'adamw', 'adam', 'sgd' "
            "or pass an instantiated `torch.optim.Optimizer`."
        )

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 10,
        callbacks: list[Callable[[int, dict[str, float]], None]] | None = None,
    ) -> dict[str, list[float]]:
        """Trains the student model using knowledge distillation.

        Args:
            train_dataloader: Dataloader yielding training batches.
            val_dataloader: Optional dataloader for evaluation after each epoch.
            epochs: Number of training epochs. Defaults to 10.
            callbacks: Optional list of callback functions triggered at epoch end.

        Returns:
            dict[str, list[float]]: Dictionary tracking history across epochs.
        """
        return self._engine.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            callbacks=callbacks,
        )

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluates student performance on a given dataloader.

        Args:
            dataloader: Dataloader yielding validation/testing batches.

        Returns:
            dict[str, float]: Validation metrics dictionary.
        """
        return self._engine.evaluate(dataloader)

    def benchmark(
        self,
        sample_input: torch.Tensor,
        teacher_name: str = "Teacher",
        student_name: str = "Student",
        val_dataloader: DataLoader | None = None,
    ) -> BenchmarkReport:
        """Runs complete profiling suite on both models and outputs comparison report.

        Args:
            sample_input: Batch tensor matching target inference dimension (e.g., [1, 3, 224, 224]).
            teacher_name: Display label for teacher model. Defaults to "Teacher".
            student_name: Display label for student model. Defaults to "Student".
            val_dataloader: Optional dataloader to compute final accuracy metrics.

        Returns:
            BenchmarkReport: Structured benchmark report ready for `.show()`.
        """
        teacher_acc: float | None = None
        student_acc: float | None = None

        if val_dataloader is not None:
            teacher_acc = compute_accuracy(self.teacher, val_dataloader, self.device)
            student_acc = compute_accuracy(self.student, val_dataloader, self.device)

        return Profiler.compare(
            teacher=self.teacher,
            student=self.student,
            sample_input=sample_input,
            device=self.device,
            teacher_name=teacher_name,
            student_name=student_name,
            teacher_acc=teacher_acc,
            student_acc=student_acc,
        )

    def feature_analysis(
        self, dataloader: DataLoader, metrics: list[str] | None = None
    ) -> FeatureAnalyzerReport:
        """Evaluates how well the student mimics the teacher's internal hidden states.

        This requires both the teacher and student models to have been wrapped
        with `FeatureExtractor` prior to initializing the Distiller.

        Args:
            dataloader: Dataloader yielding validation/testing batches.
            metrics: List of metrics to compute (e.g., 'cka'). Defaults to ["cka"].

        Returns:
            FeatureAnalyzerReport: Structured report ready for `.show()`.

        Raises:
            ValueError: If models are not wrapped with `FeatureExtractor`.
        """

        if not isinstance(self.teacher, FeatureExtractor) or not isinstance(
            self.student, FeatureExtractor
        ):
            raise ValueError(
                "Feature analysis requires both models to be wrapped with `FeatureExtractor`. "
                "If you are only doing logit-based distillation (HintonLoss, etc), internal states "
                "are not accessible."
            )

        if metrics is None:
            metrics = ["cka"]

        analyzer = FeatureAnalyzer(
            teacher_extractor=self.teacher,
            student_extractor=self.student,
            device=self.device,
        )

        return analyzer.evaluate(dataloader=dataloader, metrics=metrics)

    def save_student(self, path: str | Path) -> None:
        """Saves trained student model weights to disk.

        Args:
            path: Destination file path (e.g. 'models/student.pt').
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.student.state_dict(), save_path)

    def load_student(self, path: str | Path) -> None:
        """Loads trained weights into the student model.

        Args:
            path: File path of saved state dictionary.
        """
        load_path = Path(path)
        state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
        self.student.load_state_dict(state_dict)
