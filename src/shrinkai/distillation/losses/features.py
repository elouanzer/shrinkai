"""Feature-based Knowledge Distillation loss functions.

This module provides loss functions designed to operate on the intermediate
representations (hidden layers, attention maps, feature maps) of neural networks.

Unlike logit-based distillation which only penalizes the final output, feature
distillation forces the student to learn the internal reasoning process and
hierarchical representations of the teacher. This is particularly crucial when
the teacher is much deeper than the student (e.g., FitNets, TinyBERT).

Available Losses:
    - FeatureLoss: Aligns standard activations (Conv/Linear) using MSE, L1, or Cosine.
    - AttentionMapLoss: Aligns Transformer attention matrices (TinyBERT style).
    - GramMatrixLoss: Aligns feature co-occurrences for style/texture transfer.

Note:
    Feature dimensions often differ between student and teacher models.
    It is the user's responsibility to apply a projection layer (e.g., a 1x1 Conv
    or a Linear layer) to the student's features before passing them to these losses.
    To deal with, you can use a `ProjectedFeatureLoss`, it is a solution, but not
    necessarily the best one for your problem.

Examples:
    >>> from shrinkai.distillation.losses import FeatureLoss
    >>> criterion = FeatureLoss(loss_type="mse", normalize=True)
    >>> loss = criterion(student_hidden_states, teacher_hidden_states)
"""

from typing import Literal

import torch
import torch.nn.functional as F

from .base import BaseDistillationLoss


class FeatureLoss(BaseDistillationLoss):
    """Computes the loss between intermediate feature maps of the Teacher and Student.

    This loss encourages the Student to mimic the internal representations (activations)
    of the Teacher. It assumes that the spatial and channel dimensions of the compared
    features have already been matched.

    Attributes:
        loss_type (str): Type of loss to compute ('mse', 'l1', or 'cosine').
        normalize (bool): If True, L2-normalizes the feature vectors before computing
            the loss. This is often useful to match the "direction" of features
            regardless of their magnitude.
    """

    def __init__(
        self,
        loss_type: Literal["mse", "l1", "cosine"] = "mse",
        normalize: bool = False,
    ) -> None:
        """Initializes the FeatureLoss.

        Args:
            loss_type: Distance metric to use ('mse', 'l1', or 'cosine'). Defaults to 'mse'.
            normalize: Whether to apply L2 normalization to features before comparison.
                Defaults to False.

        Raises:
            ValueError: If an unsupported `loss_type` is provided.
        """
        super().__init__()
        if loss_type not in ["mse", "l1", "cosine"]:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Use 'mse', 'l1', or 'cosine'.")

        self.loss_type = loss_type
        self.normalize = normalize

    def _compute_distance(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """Computes the specified distance metric between two feature tensors."""
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                f"Feature shape mismatch: Student {student_feat.shape} vs"
                f"Teacher {teacher_feat.shape}."
                "Ensure dimensions match, or apply a projection layer"
                "to the student features before computing the loss."
            )

        if self.normalize:
            student_feat = F.normalize(student_feat.view(student_feat.size(0), -1), p=2, dim=-1)
            teacher_feat = F.normalize(teacher_feat.view(teacher_feat.size(0), -1), p=2, dim=-1)

        if self.loss_type == "mse":
            return F.mse_loss(student_feat, teacher_feat)
        if self.loss_type == "l1":
            return F.l1_loss(student_feat, teacher_feat)
        if self.loss_type == "cosine":
            s_flat = student_feat.view(student_feat.size(0), -1)
            t_flat = teacher_feat.view(teacher_feat.size(0), -1)
            target = torch.ones(s_flat.size(0), device=s_flat.device)
            return F.cosine_embedding_loss(s_flat, t_flat, target)

        raise ValueError("Invalid loss type.")

    def forward(
        self,
        student_outputs: torch.Tensor | dict[str, torch.Tensor],
        teacher_outputs: torch.Tensor | dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the feature distillation loss.

        Supports comparing single tensors or dictionaries of tensors. If dictionaries
        are provided, it computes the average loss across all matching keys.

        Args:
            student_outputs: Tensor or dict of intermediate feature tensors from the student.
            teacher_outputs: Tensor or dict of intermediate feature tensors from the teacher.
            labels: Ground-truth labels (ignored, kept for API compatibility).

        Returns:
            torch.Tensor: Aggregated scalar loss value.

        Raises:
            TypeError: If input types for student and teacher do not match.
            ValueError: If dict keys do not match between student and teacher.
        """
        return self._apply_feature_distance(
            student_outputs, teacher_outputs, self._compute_distance
        )


class AttentionMapLoss(BaseDistillationLoss):
    """Distillation loss for Transformer attention maps.

    This loss forces the student model to mimic the attention patterns of the teacher.
    It operates on attention matrices, typically of shape:
    [Batch, Num_Heads, Seq_Len, Seq_Len].

    Common practices:
        - TinyBERT: Uses MSE on unnormalized attention scores (before softmax).
        - DistilBERT / MiniLM: Uses KL Divergence on attention probabilities (after softmax).

    Attributes:
        loss_type (str): The metric to use ('mse' or 'kl').
    """

    def __init__(self, loss_type: Literal["mse", "kl"] = "mse") -> None:
        """Initializes the AttentionMapLoss.

        Args:
            loss_type: Distance metric ('mse' or 'kl'). Defaults to 'mse'.
                    If 'kl' is used, inputs must be unnormalized logits (before softmax),
                    and the KL divergence will be applied across the last dimension.

        Raises:
            ValueError: If an unsupported `loss_type` is provided.
        """
        super().__init__()
        if loss_type not in ["mse", "kl"]:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Use 'mse' or 'kl'.")

        self.loss_type = loss_type

    def _compute_distance(self, s_map: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:
        """Computes the loss between two attention matrices."""
        if s_map.shape != t_map.shape:
            raise ValueError(
                f"Attention map shape mismatch: Student {s_map.shape} vs Teacher {t_map.shape}. "
                "The number of heads and sequence lengths must match. If your student has "
                "fewer heads, consider aligning specific heads before passing them to this loss."
            )

        if self.loss_type == "mse":
            return F.mse_loss(s_map, t_map)

        elif self.loss_type == "kl":
            log_soft_s = F.log_softmax(s_map, dim=-1)
            soft_t = F.softmax(t_map, dim=-1)
            return F.kl_div(input=log_soft_s, target=soft_t, reduction="batchmean")

        raise ValueError("Invalid loss type.")

    def forward(
        self,
        student_outputs: torch.Tensor | dict[str, torch.Tensor],
        teacher_outputs: torch.Tensor | dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the attention map distillation loss.

        Supports comparing single tensors or dictionaries of tensors. If dictionaries
        are provided, it computes the average loss across all matching keys.

        Args:
            student_outputs: Tensor or dict of attention matrices from the student.
            teacher_outputs: Tensor or dict of attention matrices from the teacher.
            Ground-truth labels (ignored, kept for API compatibility).

        Returns:
            torch.Tensor: Aggregated scalar loss value.
        """
        return self._apply_feature_distance(
            student_outputs, teacher_outputs, self._compute_distance
        )


class GramMatrixLoss(BaseDistillationLoss):
    """Distillation loss based on Gram Matrices for style and texture transfer.

    Instead of forcing the student to match the exact spatial activations of the
    teacher (which is strict and requires identical spatial dimensions), this loss
    forces the student to match the channel-wise feature correlations (co-occurrence).

    This is highly effective for Generative tasks, Super-Resolution, or making a
    student network mimic the global "texture" representation of a teacher.

    Attributes:
        loss_type (str): The distance metric to apply on the Gram matrices ('mse' or 'l1').
    """

    def __init__(self, loss_type: Literal["mse", "l1", "cosine"] = "mse") -> None:
        """Initializes the GramMatrixLoss.

        Args:
            loss_type: Distance metric ('mse' or 'l1'). Defaults to 'mse'.

        Raises:
            ValueError: If an unsupported `loss_type` is provided.
        """
        super().__init__()
        if loss_type not in ["mse", "l1", "cosine"]:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Use 'mse', 'l1' or 'cosine'.")

        self.loss_type = loss_type

    def _compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the normalized Gram matrix of a feature tensor.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, C, L].

        Returns:
            torch.Tensor: Gram matrix of shape [B, C, C].
        """
        if x.dim() < 3:
            raise ValueError(
                f"Gram matrix requires at least 3D tensors (Batch, Channels, Spatial/Temporal). "
                f"Got tensor of shape {x.shape} (dim={x.dim()})."
            )
        batch_size, channels = x.size(0), x.size(1)

        # [B, C, H, W] -> [B, C, H*W]
        x_flat = x.view(batch_size, channels, -1)
        num_elements = x_flat.size(2)

        gram = torch.bmm(x_flat, x_flat.transpose(1, 2))
        gram = gram / (channels * num_elements)

        return gram

    def _compute_distance(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Computes the distance between the Gram matrices of student and teacher."""
        if s_feat.size(1) != t_feat.size(1):
            raise ValueError(
                f"Channel dimension mismatch: Student has {s_feat.size(1)} channels, "
                f"Teacher has {t_feat.size(1)} channels. The Gram matrix requires identical "
                f"channel dimensions. Apply a 1x1 Conv (projector) to the student features first."
            )

        s_gram = self._compute_gram_matrix(s_feat)
        t_gram = self._compute_gram_matrix(t_feat)

        if self.loss_type == "mse":
            return F.mse_loss(s_gram, t_gram)
        elif self.loss_type == "l1":
            return F.l1_loss(s_gram, t_gram)
        elif self.loss_type == "cosine":
            s_flat = s_gram.view(s_gram.size(0), -1)
            t_flat = t_gram.view(t_gram.size(0), -1)
            target = torch.ones(s_flat.size(0), device=s_flat.device)
            return F.cosine_embedding_loss(s_flat, t_flat, target)

        raise ValueError("Invalid loss type.")

    def forward(
        self,
        student_outputs: torch.Tensor | dict[str, torch.Tensor],
        teacher_outputs: torch.Tensor | dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the Gram matrix distillation loss.

        Args:
            student_outputs: Tensor or dict of intermediate features [B, C, H, W].
            teacher_outputs: Tensor or dict of intermediate features [B, C, H, W].
            labels: Ground-truth labels (ignored, kept for API compatibility).

        Returns:
            torch.Tensor: Aggregated scalar loss value.
        """
        return self._apply_feature_distance(
            student_outputs, teacher_outputs, self._compute_distance
        )
