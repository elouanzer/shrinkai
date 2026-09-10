"""Logit-based Knowledge Distillation loss functions.

This module provides various loss functions that operate on the final output logits
of the student and teacher models. Logit-based distillation (or "response-based"
distillation) is the most common form of Knowledge Distillation.

These methods typically apply temperature scaling to the models' logits to soften
the probability distributions. This exposes the "dark knowledge" of the teacher
(i.e., the relative probabilities assigned to non-target classes), allowing the
student to learn the teacher's generalization capabilities.

Available Losses:
    - HintonLoss: The standard KD loss (Cross-Entropy + KL Divergence).
    - PureKDLoss: Pure distillation without ground-truth labels (KL Divergence only).
    - ReverseKLLoss: Mode-seeking distillation, highly effective for LLMs.
    - BCEKDLoss: Distillation for multi-label classification tasks (Sigmoid + BCE).
    - JSDLoss: Symmetric distillation using Jensen-Shannon Divergence.

Examples:
    >>> from shrinkai.distillation.losses import HintonLoss
    >>> criterion = HintonLoss(temperature=4.0, alpha=0.5)
    >>> loss = criterion(student_logits, teacher_logits, labels)
"""

from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDistillationLoss


def expects_logits(func):
    """Decorator to automatically extract logits from FeatureExtractor tuples."""

    @wraps(func)
    def wrapper(self, student_outputs, teacher_outputs, *args, **kwargs):
        s_logits = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        t_logits = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
        return func(self, s_logits, t_logits, *args, **kwargs)

    return wrapper


class HintonLoss(BaseDistillationLoss):
    r"""Knowledge Distillation loss (Geoffrey Hinton et al. (2015)).

    Combines standard task loss (Cross-Entropy with hard ground-truth labels)
    and distillation loss (Kullback-Leibler divergence on softened probabilities
    produced by a teacher model at a given temperature).

    Equation:
        $$L = (1 - \alpha) \cdot L_{ce}(z_s, y) + \alpha \cdot T^2 \cdot L_{kl}\left(\text{softmax}\left(\frac{z_s}{T}\right), \text{softmax}\left(\frac{z_t}{T}\right)\right)$$

    Attributes:
        temperature (float): Softening factor for logits. Higher values produce
            smoother probability distributions over classes. Must be > 0.
        alpha (float): Weight balancing factor between hard label loss and
            distillation loss. Must be in range [0.0, 1.0].
    """  # noqa: E501

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        """Initializes the HintonLoss module.

        Args:
            temperature: Temperature scaling factor (T > 0). Defaults to 4.0.
            alpha: Weight for distillation loss (0.0 <= alpha <= 1.0). Defaults to 0.5.

        Raises:
            ValueError: If `temperature` <= 0 or `alpha` is not in [0.0, 1.0].
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, current temperature is {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, current alpha is {alpha}")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.cross_entropy = nn.CrossEntropyLoss()

    @expects_logits
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes combined Cross-Entropy and KD divergence losses.

        Args:
            student_outputs: Raw unnormalized logits from the student model (Shape: [B, C]).
            teacher_outputs: Raw unnormalized logits from the teacher model (Shape: [B, C]).
            labels: Ground-truth class indices (Shape: [B]). Optional if alpha == 1.0.

        Returns:
            torch.Tensor: Weighted scalar loss value.

        Raises:
            ValueError: If `labels` is None when `alpha` < 1.0.
            ValueError: If shapes of `student_outputs` and `teacher_outputs` mismatch.
        """
        if student_outputs.shape != teacher_outputs.shape:
            raise ValueError(
                f"Shape mismatch: student logits {student_outputs.shape} "
                f"vs teacher logits {teacher_outputs.shape}"
            )

        soft_student = F.log_softmax(student_outputs / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=-1)
        kd_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature**2)

        if self.alpha == 1.0:
            return kd_loss

        if labels is None:
            raise ValueError("Ground-truth `labels` are required when `alpha` < 1.0.")

        ce_loss = self.cross_entropy(student_outputs, labels)
        return (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss


class PureKDLoss(HintonLoss):
    r"""Pure Knowledge Distillation loss.

    Implement distillation loss (Kullback-Leibler divergence on softened probabilities
    produced by a teacher model at a given temperature). In fact, a `PureKDLoss`object
    is just a `HintonLoss` object with `alpha` set to 1.

    Equation:
        $$L = T^2 \cdot L_{kl}\left(\text{softmax}\left(\frac{z_s}{T}\right), \text{softmax}\left(\frac{z_t}{T}\right)\right)$$

    Attributes:
        temperature (float): Softening factor for logits. Higher values produce
            smoother probability distributions over classes. Must be > 0.
    """  # noqa: E501

    def __init__(self, temperature: float = 4.0) -> None:
        """Initializes the PureKDLoss module.

        Args:
            temperature: Temperature scaling factor (T > 0). Defaults to 4.0.

        Raises:
            ValueError: If `temperature` <= 0.
        """
        super().__init__(temperature=temperature, alpha=1.0)


class ReverseKLLoss(BaseDistillationLoss):
    r"""Reverse Kullback-Leibler divergence for Knowledge Distillation.

    Standard KD (Forward KL) computes KL(P_teacher || P_student), which is "mode-covering".
    Reverse KL computes KL(P_student || P_teacher), which is "mode-seeking".

    For Large Language Models (LLMs), Reverse KL is highly effective because it strongly
    penalizes the student for assigning high probabilities to tokens that the teacher
    considers unlikely, thereby reducing hallucinations and degeneration.

    Equation:
        $$L = T^2 \cdot \sum \left( P_s \cdot (\log(P_s) - \log(P_t)) \right)$$

    Attributes:
        temperature (float): Softening factor for logits. Must be > 0.
            Note: In LLM distillation, temperature is often set close to 1.0.
    """  # noqa: E501

    def __init__(self, temperature: float = 1.0) -> None:
        """Initializes the ReverseKLLoss module.

        Args:
            temperature: Temperature scaling factor (T > 0). Defaults to 1.0.

        Raises:
            ValueError: If `temperature` <= 0.
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, current temperature is {temperature}")

        self.temperature = float(temperature)
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)

    @expects_logits
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the Reverse KL divergence loss.

        Args:
            student_outputs: Raw unnormalized logits from the student model
                (Shape: [B, C] or [B, S, C]).
            teacher_outputs: Raw unnormalized logits from the teacher model
                (Shape: [B, C] or [B, S, C]).
            labels: Ground-truth labels (ignored, kept for API compatibility).

        Returns:
            torch.Tensor: Scalar loss value.

        Raises:
            ValueError: If shapes of `student_outputs` and `teacher_outputs` mismatch.
        """
        if student_outputs.shape != teacher_outputs.shape:
            raise ValueError(
                f"Shape mismatch: student logits {student_outputs.shape} "
                f"vs teacher logits {teacher_outputs.shape}"
            )

        # IMPORTANT: Do not swap `input` and `target`!
        # PyTorch's KLDivLoss computes KL(target || input).
        # Standard KD (Forward KL) passes `target=teacher, input=student` -> KL(teacher || student).
        # For Reverse KL, we want KL(student || teacher), so we MUST pass:
        # `target=student, input=teacher`.
        log_soft_student = F.log_softmax(student_outputs / self.temperature, dim=-1)
        log_soft_teacher = F.log_softmax(teacher_outputs / self.temperature, dim=-1)
        kd_loss = self.kl_div(input=log_soft_teacher, target=log_soft_student)
        return kd_loss * (self.temperature**2)


class BCEKDLoss(BaseDistillationLoss):
    r"""Knowledge Distillation loss for Multi-Label classification tasks.

    Unlike HintonLoss which uses Softmax (classes are mutually exclusive),
    this loss uses Sigmoid to treat each class independently. It computes
    the Binary Cross Entropy (BCE) between the student's logits and the
    teacher's softened targets.

    Equation:
        $$L = (1 - \alpha) \cdot \text{BCE}(z_s, y) + \alpha \cdot T^2 \cdot \text{BCE}\left(\frac{z_s}{T}, \text{sigmoid}\left(\frac{z_t}{T}\right)\right)$$

    Attributes:
        temperature (float): Softening factor for logits. Must be > 0.
        alpha (float): Weight balancing factor. Must be in range [0.0, 1.0].
    """  # noqa: E501

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    @expects_logits
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes combined hard BCE and soft BCE losses.

        Args:
            student_outputs: Raw unnormalized logits from the student model
                (Shape: [B, C] or [B, S, C]).
            teacher_outputs: Raw unnormalized logits from the teacher model
                (Shape: [B, C] or [B, S, C]).
            labels: Ground-truth labels (ignored, kept for API compatibility).

        Returns:
            torch.Tensor: Scalar loss value.

        Raises:
            ValueError: If shapes of `student_outputs` and `teacher_outputs` mismatch.
        """
        if student_outputs.shape != teacher_outputs.shape:
            raise ValueError("Shape mismatch between student and teacher logits.")

        with torch.no_grad():
            soft_targets = torch.sigmoid(teacher_outputs / self.temperature)

        kd_loss = F.binary_cross_entropy_with_logits(
            student_outputs / self.temperature, soft_targets
        )
        kd_loss = kd_loss * (self.temperature**2)

        if self.alpha == 1.0:
            return kd_loss

        if labels is None:
            raise ValueError("Ground-truth `labels` are required when `alpha` < 1.0.")

        hard_loss = self.bce_with_logits(student_outputs, labels.float())
        return (1.0 - self.alpha) * hard_loss + self.alpha * kd_loss


class JSDLoss(BaseDistillationLoss):
    r"""Jensen-Shannon Divergence (JSD) loss for Knowledge Distillation.

    JSD is a symmetric and bounded alternative to the standard KL Divergence.
    It computes the divergence of both distributions from their average distribution M.
    This boundedness prevents gradient explosion, especially early in training when
    the student's predictions might diverge heavily from the teacher's.

    Equation:
        $$
        \begin{aligned}
        M &= 0.5 \cdot (P_{student} + P_{teacher}) \\
        L_{JSD} &= 0.5 \cdot \text{KL}(P_{student} \parallel M) + 0.5 \cdot \text{KL}(P_{teacher} \parallel M)
        \end{aligned}
        $$

    Attributes:
        temperature (float): Softening factor for logits. Must be > 0.
    """  # noqa: E501

    def __init__(self, temperature: float = 4.0) -> None:
        """Initializes the JSDLoss module.

        Args:
            temperature: Softening factor for logits (T > 0). Defaults to 4.0.

        Raises:
            ValueError: If `temperature` <= 0.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = float(temperature)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    @expects_logits
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the JSD between softened student and teacher distributions.

         Args:
            student_outputs: Raw unnormalized logits from the student model (Shape: [B, C] or [B, S, C]).
            teacher_outputs: Raw unnormalized logits from the teacher model (Shape: [B, C] or [B, S, C]).
            labels: Ground-truth labels (ignored, kept for API compatibility).

        Returns:
            torch.Tensor: Scalar loss value.

        Raises:
            ValueError: If shapes of `student_outputs` and `teacher_outputs` mismatch.
        """  # noqa: E501
        if student_outputs.shape != teacher_outputs.shape:
            raise ValueError("Shape mismatch between student and teacher logits.")

        p_s = F.softmax(student_outputs / self.temperature, dim=-1)
        p_t = F.softmax(teacher_outputs / self.temperature, dim=-1)

        m = 0.5 * (p_s + p_t)

        log_m = torch.log(m.clamp(min=1e-8))
        kl_s_m = self.kl_div(input=log_m, target=p_s)
        kl_t_m = self.kl_div(input=log_m, target=p_t)

        jsd = 0.5 * (kl_s_m + kl_t_m)

        return jsd * (self.temperature**2)
