from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..adapters import FeatureExtractor
from ..analysis import FeatureAnalyzer, FeatureAnalyzerReport
from ..export import export_onnx, export_torchscript
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
        use_amp: bool = False,
        grad_clip_norm: float | None = None,
        engine_class: type[DistillationEngine] | None = None,
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
            use_amp: If True, trains under mixed precision (fp16+scaling on CUDA,
                bf16 on CPU/MPS). Defaults to False.
            grad_clip_norm: If set, clips the student's gradient global L2 norm to
                this value before each optimizer step. Defaults to None.
            engine_class: The engine class for distillation.
                If None, it uses a basic DistillationEngine.
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
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        if engine_class is None:
            self._engine = DistillationEngine(
                student=self.student,
                teacher=self.teacher,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=self.scheduler,
                use_amp=use_amp,
                grad_clip_norm=grad_clip_norm,
            )
        else:
            self._engine = engine_class(
                student=self.student,
                teacher=self.teacher,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=self.scheduler,
                use_amp=use_amp,
                grad_clip_norm=grad_clip_norm,
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
        resume: bool = False,
    ) -> dict[str, list[float]]:
        """Trains the student model using knowledge distillation.

        Args:
            train_dataloader: Dataloader yielding training batches.
            val_dataloader: Optional dataloader for evaluation after each epoch.
            epochs: Total number of epochs to train up to (1-indexed, inclusive).
                Defaults to 10.
            callbacks: Optional list of callback functions triggered at epoch end.
                A callback exposing a truthy `stop` attribute (e.g. `EarlyStopping`)
                interrupts training at the end of that epoch.
            resume: If True, continues training from `self.history` (populated by a
                previous `fit()` call or by `load_checkpoint()`) instead of starting
                a fresh run from epoch 1. Defaults to False.

        Returns:
            dict[str, list[float]]: Dictionary tracking history across epochs. Also
            stored on `self.history` for later checkpointing.
        """
        start_epoch = len(self.history["train_loss"]) + 1 if resume else 1
        history = self.history if resume else None

        self.history = self._engine.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            callbacks=callbacks,
            start_epoch=start_epoch,
            history=history,
        )
        return self.history

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
        compute_flops: bool = False,
    ) -> BenchmarkReport:
        """Runs complete profiling suite on both models and outputs comparison report.

        Args:
            sample_input: Batch tensor matching target inference dimension (e.g., [1, 3, 224, 224]).
            teacher_name: Display label for teacher model. Defaults to "Teacher".
            student_name: Display label for student model. Defaults to "Student".
            val_dataloader: Optional dataloader to compute final accuracy metrics.
            compute_flops: If True, also reports FLOPs per sample for both models.
                Defaults to False. See `Profiler.compare`.

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
            compute_flops=compute_flops,
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

    def export_onnx(
        self,
        path: str | Path,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> Path:
        """Exports the trained student to ONNX. See `shrinkai.export.export_onnx`.

        Args:
            path: Destination `.onnx` file path.
            sample_input: Representative input tensor (or tuple of tensors).
            **kwargs: Forwarded to `shrinkai.export.export_onnx` (e.g.
                `dynamic_batch`, `opset_version`, `input_names`).

        Returns:
            Path: The path the model was exported to.
        """
        return export_onnx(self.student, sample_input, path, **kwargs)

    def export_torchscript(
        self,
        path: str | Path,
        sample_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        method: Literal["trace", "script"] = "trace",
    ) -> Path:
        """Exports the trained student to TorchScript. See
        `shrinkai.export.export_torchscript`.

        Args:
            path: Destination file path.
            sample_input: Required when `method="trace"`.
            method: "trace" (default) or "script".

        Returns:
            Path: The path the model was exported to.
        """
        return export_torchscript(self.student, path, sample_input=sample_input, method=method)

    def save_checkpoint(self, path: str | Path) -> None:
        """Saves a full training checkpoint (student, optimizer, scheduler, history).

        Unlike `save_student`, which only persists inference weights, this saves
        everything needed to resume training later via `load_checkpoint` followed by
        `fit(..., resume=True)`.

        Args:
            path: Destination file path (e.g. 'checkpoints/epoch_10.pt').
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "history": self.history,
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restores a full training checkpoint saved by `save_checkpoint`.

        After calling this, resume training with `fit(..., resume=True)`, it will
        continue from the epoch right after the last one recorded in the restored
        history.

        Args:
            path: File path of the saved checkpoint.

        Raises:
            ValueError: If this Distiller's scheduler configuration (present vs.
                absent) does not match the one the checkpoint was saved with.
        """
        load_path = Path(path)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is None and scheduler_state is not None:
            raise ValueError(
                "The checkpoint contains a scheduler state, but this Distiller "
                "has no scheduler configured."
            )
        if self.scheduler is not None and scheduler_state is None:
            raise ValueError(
                "This Distiller has a scheduler configured, but the checkpoint "
                "was saved without one."
            )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.history = checkpoint.get(
            "history",
            {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []},
        )
