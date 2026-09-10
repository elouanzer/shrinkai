"""Base class for Knowledge Distillation loss functions."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn


class BaseDistillationLoss(nn.Module, ABC):
    """
    Abstract base class for all distillation loss functions.

    All concrete implementations must override the `forward` method.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        student_outputs: torch.Tensor | dict[str, torch.Tensor],
        teacher_outputs: torch.Tensor | dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the distillation loss.

        Args:
            student_outputs: Output tensor (or dict of activations) from the student model.
            teacher_outputs: Output tensor (or dict of activations) from the teacher model.
            labels: Ground-truth task labels (optional depending on the loss type).

        Returns:
            torch.Tensor: Scalar loss tensor for backpropagation.
        """
        pass

    def _apply_feature_distance(
        self,
        student_outputs: torch.Tensor | dict[str, torch.Tensor],
        teacher_outputs: torch.Tensor | dict[str, torch.Tensor],
        distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Helper method to apply a distance function across tensors or dicts of tensors.

        Args:
            student_outputs: Tensor or dict of intermediate feature tensors.
            teacher_outputs: Tensor or dict of intermediate feature tensors.
            distance_fn: A function that computes the distance between two tensors.

        Returns:
            torch.Tensor: The averaged distance.
        """
        if isinstance(student_outputs, dict) and isinstance(teacher_outputs, dict):
            if student_outputs.keys() != teacher_outputs.keys():
                raise ValueError(
                    f"Dictionary key mismatch. Student keys: {list(student_outputs.keys())}, "
                    f"Teacher keys: {list(teacher_outputs.keys())}"
                )

            total_loss = torch.tensor(0.0, device=next(iter(student_outputs.values())).device)
            for key in student_outputs.keys():
                total_loss += distance_fn(student_outputs[key], teacher_outputs[key])

            return total_loss / len(student_outputs)

        if isinstance(student_outputs, torch.Tensor) and isinstance(teacher_outputs, torch.Tensor):
            return distance_fn(student_outputs, teacher_outputs)

        raise TypeError("Student and teacher outputs must be both Tensors or both dictionaries.")
