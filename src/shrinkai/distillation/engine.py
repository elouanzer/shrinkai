import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils import resolve_device
from .losses.base import BaseDistillationLoss


class DistillationEngine:
    """Generic training engine for knowledge distillation across hardware targets.

    Handles training/validation loops, accelerator management (MPS, CUDA, CPU),
    teacher state freezing, and metrics tracking.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        criterion: BaseDistillationLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "auto",
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        use_amp: bool = False,
        grad_clip_norm: float | None = None,
    ) -> None:
        """Initializes the DistillationEngine.

        Args:
            student: Student neural network module to train.
            teacher: Pre-trained Teacher neural network module providing soft targets.
            criterion: Loss function adhering to `BaseDistillationLoss`.
            optimizer: PyTorch optimizer targeting student parameters.
            device: Computing device ('auto', 'mps', 'cuda', 'cpu' or torch.device).
            scheduler: Optional learning rate scheduler updated per epoch.
            use_amp: If True, runs the forward passes and loss computation under
                mixed precision (`torch.autocast`). Uses fp16 with gradient scaling
                on CUDA, and bf16 (no scaling needed) on CPU/MPS. Defaults to False.
            grad_clip_norm: If set, clips the student's gradient global L2 norm to
                this value before each optimizer step. Defaults to None (no clipping).
        """
        self.device = resolve_device(device)
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self._amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        self._use_scaler = self.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self._use_scaler)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _unpack_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """Unpacks tuple/list or dict batches and moves them to the device.

        Provides compatibility with both standard PyTorch Datasets (tuples)
        and Hugging Face Datasets (dictionaries).
        """
        if isinstance(batch, dict):
            # Hugging Face style batch
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            if "labels" not in batch:
                raise ValueError("Batch dictionary must contain a 'labels' key.")
            labels = batch.pop("labels")
            return batch, labels

        elif isinstance(batch, list | tuple):
            # Standard PyTorch style batch
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            return inputs, labels

        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _forward_model(self, model: nn.Module, inputs: Any) -> Any:
        """Executes forward pass adapting to the input type (kwargs or args)."""
        if isinstance(inputs, dict):
            return model(**inputs)
        return model(inputs)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        total_epochs: int,
    ) -> dict[str, float]:
        """Runs a single training epoch over the provided dataloader.

        Args:
            dataloader: Training dataloader yielding (inputs, labels) batches.
            epoch_idx: Current 1-based epoch index.
            total_epochs: Total number of planned epochs.

        Returns:
            dict[str, float]: Aggregated training metrics (loss, accuracy).
        """
        self.student.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        desc = f"Epoch [{epoch_idx}/{total_epochs}] Training"
        pbar = tqdm(dataloader, desc=desc, leave=False)

        for batch in pbar:
            inputs, labels = self._unpack_batch(batch)

            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type, dtype=self._amp_dtype, enabled=self.use_amp
                ),
            ):
                teacher_outputs = self._forward_model(self.teacher, inputs)

            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=self._amp_dtype, enabled=self.use_amp
            ):
                student_outputs = self._forward_model(self.student, inputs)
                loss = self.criterion(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=labels,
                )

            if not math.isfinite(loss.item()):
                raise RuntimeError(
                    f"Loss diverged to {loss.item()} during training at epoch {epoch_idx}. "
                    "Check learning rate or data scaling."
                )

            if self._use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            student_logits = (
                student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
            )

            if student_logits.dim() == 3:
                # LLM Causal Shift
                preds = torch.argmax(student_logits[..., :-1, :].contiguous(), dim=-1)
                shifted_labels = labels[..., 1:].contiguous()
                correct += (preds == shifted_labels).sum().item()
                batch_samples = shifted_labels.numel()
            else:
                preds = torch.argmax(student_logits, dim=-1)
                correct += (preds == labels).sum().item()
                batch_samples = labels.size(0)

            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

            current_loss = total_loss / total_samples
            current_acc = (correct / total_samples) * 100.0
            pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

        return {
            "loss": total_loss / total_samples,
            "accuracy": (correct / total_samples) * 100.0,
        }

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluates student performance on validation/test data.

        Args:
            dataloader: Validation dataloader yielding (inputs, labels) batches.

        Returns:
            dict[str, float]: Validation metrics (loss, accuracy).
        """
        self.student.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type, dtype=self._amp_dtype, enabled=self.use_amp
            ),
        ):
            for batch in dataloader:
                inputs, labels = self._unpack_batch(batch)

                teacher_outputs = self._forward_model(self.teacher, inputs)
                student_outputs = self._forward_model(self.student, inputs)

                loss = self.criterion(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=labels,
                )
                student_logits = (
                    student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
                )

                if student_logits.dim() == 3:
                    # LLM Causal Shift
                    preds = torch.argmax(student_logits[..., :-1, :].contiguous(), dim=-1)
                    shifted_labels = labels[..., 1:].contiguous()
                    correct += (preds == shifted_labels).sum().item()
                    batch_samples = shifted_labels.numel()
                else:
                    preds = torch.argmax(student_logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    batch_samples = labels.size(0)

                total_loss += loss.item() * batch_samples
                total_samples += batch_samples

        return {
            "val_loss": total_loss / total_samples,
            "val_accuracy": (correct / total_samples) * 100.0,
        }

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 10,
        callbacks: list[Callable[[int, dict[str, float]], None]] | None = None,
        start_epoch: int = 1,
        history: dict[str, list[float]] | None = None,
    ) -> dict[str, list[float]]:
        """Executes the full distillation training loop.

        Args:
            train_dataloader: Dataloader containing training dataset.
            val_dataloader: Optional dataloader for epoch-end validation.
            epochs: Total number of epochs to train up to (1-indexed, inclusive).
                Defaults to 10.
            callbacks: Optional list of callback functions triggered each epoch.
                A callback exposing a truthy `stop` attribute after being called
                (e.g. `EarlyStopping`) interrupts training at the end of that epoch.
            start_epoch: 1-based epoch index to resume training from. Defaults to 1
                (a fresh run). Used together with `history` when resuming from a
                checkpoint saved via `Distiller.save_checkpoint`.
            history: Existing training history to append to, as returned by a
                previous call to `fit`. Defaults to None (starts a fresh history).

        Returns:
            dict[str, list[float]]: Training history tracking loss and metrics,
            covering both the resumed epochs (if any) and the new ones.
        """
        if history is None:
            history = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
            }

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_dataloader, epoch, epochs)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            val_metrics: dict[str, float] = {}
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])

            if self.scheduler is not None:
                self.scheduler.step()

            status = (
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            if val_metrics:
                status += (
                    f" | Val Loss: {val_metrics['val_loss']:.4f} - "
                    f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
                )
            tqdm.write(status)

            if callbacks:
                epoch_summary = {**train_metrics, **val_metrics}
                for callback in callbacks:
                    callback(epoch, epoch_summary)

                if any(getattr(callback, "stop", False) for callback in callbacks):
                    tqdm.write(f"Training stopped early at epoch {epoch}/{epochs}.")
                    break

        return history
