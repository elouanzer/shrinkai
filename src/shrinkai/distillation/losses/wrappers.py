"""Wrapper modules for composing and routing distillation losses.

These classes do not compute mathematical distances themselves. Instead, they
orchestrate other loss functions, handle tensor dimension mapping (projection),
and combine multiple objectives into a single criterion.

Available Losses:
    - ProjectedFeatureLoss: Uses FeatureProjector to connect tensors with different dimensions.
    - HybridLoss: Combines a logit loss with a feature loss.
    - CombinedLoss: Weighted sum of several losses.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDistillationLoss


class ProjectedFeatureLoss(BaseDistillationLoss):
    """Bridge between FeatureExtractors, FeatureProjectors, and Feature Losses.

    When using `FeatureExtractor`, the model outputs a tuple: `(logits, features_dict)`.
    However, feature losses (like `FeatureLoss`, `AttentionMapLoss`) expect pure
    dictionaries or tensors.

    This wrapper seamlessly unpacks the tuples, applies the `FeatureProjector` to
    align the student's feature dimensions with the teacher's, and computes the
    underlying feature loss.

    Attributes:
        projector (nn.Module): The module responsible for projecting student features.
        feature_loss (BaseDistillationLoss): The loss function to apply to the aligned features.
        project_teacher (bool): If False (default), the projector is applied to the
            student's features to match the teacher's larger dimensions.
            If True, the projector is applied to the teacher's features to
            downscale them (e.g., filtering a 12-head teacher down to match
            a 2-head student in Transformer attention distillation).
    """

    def __init__(
        self,
        projector: nn.Module,
        feature_loss: BaseDistillationLoss,
        project_teacher: bool = False,
    ) -> None:
        """Initializes the ProjectedFeatureLoss.

        Args:
            projector: An instantiated FeatureProjector.
            feature_loss: An instantiated feature distillation loss (e.g., FeatureLoss).
            project_teacher: Either the projector is applied to the teacher or not.
        """
        super().__init__()
        self.projector = projector
        self.feature_loss = feature_loss
        self.project_teacher = project_teacher

    def forward(
        self,
        student_outputs: tuple[torch.Tensor, dict[str, torch.Tensor]],
        teacher_outputs: tuple[torch.Tensor, dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unpacks outputs, projects features (student or teacher), and computes the loss."""
        if not isinstance(student_outputs, tuple) or not isinstance(teacher_outputs, tuple):
            raise ValueError(
                "ProjectedFeatureLoss expects outputs to be a tuple of (logits, features_dict). "
                "Ensure your models are wrapped with `FeatureExtractor`."
            )

        _, s_features_dict = student_outputs
        _, t_features_dict = teacher_outputs

        # 1. projector aligns dimensions
        if self.project_teacher:
            s_current = dict(s_features_dict)
            t_current = self.projector(t_features_dict)
        else:
            s_current = self.projector(s_features_dict)
            t_current = dict(t_features_dict)

        # 2. Spatial / temporal alignment
        for key in s_current.keys():
            if key not in t_current:
                continue

            s_tensor = s_current[key]
            t_tensor = t_current[key]

            # A : vision (CNN) -> 4D tensors [Batch, Channels, Height, Width]
            if s_tensor.dim() == 4 and t_tensor.dim() == 4:
                if s_tensor.shape[2:] != t_tensor.shape[2:]:
                    s_current[key] = F.adaptive_avg_pool2d(s_tensor, output_size=t_tensor.shape[2:])

            # B : NLP / series (Transformers, RNN) -> 3D tensors [Batch, Seq_Len, Hidden_Dim]
            elif s_tensor.dim() == 3 and t_tensor.dim() == 3:
                # In PyTorch NLP, we should have [B, L, D], but if L is different
                if s_tensor.shape[1] != t_tensor.shape[1]:
                    # adaptive_avg_pool1d expects [Batch, Channels, Length]
                    # transpose [B, L, D] -> [B, D, L], then pool, then re-transpose
                    s_transposed = s_tensor.transpose(1, 2)
                    target_length = t_tensor.shape[1]
                    s_pooled = F.adaptive_avg_pool1d(s_transposed, output_size=target_length)
                    s_current[key] = s_pooled.transpose(1, 2)

        # 3. final loss (MSE, KL, etc.)
        return self.feature_loss(
            student_outputs=s_current, teacher_outputs=t_current, labels=labels
        )


class CombinedLoss(BaseDistillationLoss):
    """Combines an arbitrary number of distillation losses with specific weights.

    This acts as a transparent router. It passes the raw inputs (whether they are
    tensors or tuples) to each underlying loss.
    """

    def __init__(self, weighted_losses: list[tuple[BaseDistillationLoss, float]]) -> None:
        """Initializes the CombinedLoss.

        Args:
            weighted_losses: A list of tuple with instantiated loss modules and their
                corresponding weights.
        """
        super().__init__()
        if not all(isinstance(item, tuple) and len(item) == 2 for item in weighted_losses):
            raise ValueError(
                "weighted_losses must be a list of tuples in the"
                "format: [(loss_module, weight), ...]"
            )
        self.losses = nn.ModuleList([loss for loss, _ in weighted_losses])
        self.weights = [weight for _, weight in weighted_losses]

    def forward(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the weighted sum of all configured losses.

        Args:
            student_outputs: Tuple of (student_logits, student_features_dict).
            teacher_outputs: Tuple of (teacher_logits, teacher_features_dict).
            labels: Ground-truth labels.

        Returns:
            torch.Tensor: Aggregated scalar loss value.
        """
        total_loss = 0.0

        for loss_fn, weight in zip(self.losses, self.weights, strict=True):
            if weight > 0.0:
                total_loss += weight * loss_fn(student_outputs, teacher_outputs, labels)

        return total_loss


class HybridLoss(CombinedLoss):
    r"""Combines a primary logit-based loss and a feature-based loss using a convex combination.

    Equation:
        $$
        \begin{aligned}
        \text{Additive } (\texttt{convex_weighting=False})\colon \quad & L = L_{\text{primary}} + \lambda \cdot L_{\text{feature}} \\\\
        \text{Convex } (\texttt{convex_weighting=True})\colon \quad & L = (1 - \lambda) \cdot L_{\text{primary}} + \lambda \cdot L_{\text{feature}}
        \end{aligned}
        $$

        where $\lambda = \text{feature_weight}$.

    This loss expects the models (or the FeatureExtractor wrapper) to return
    a tuple containing `(logits, features_dict)`.
    """  # noqa: E501

    def __init__(
        self,
        logit_loss: BaseDistillationLoss,
        feature_loss: BaseDistillationLoss,
        feature_weight: float = 1.0,
        convex_weighting: bool = False,
    ) -> None:
        """Initializes the HybridLoss.

        Args:
            logit_loss: Loss applied to the final logits (e.g., a HintonLoss object).
            feature_loss: Loss applied to the intermediate features (e.g., a FeatureLoss object).
            feature_weight: Balancing factor (0.0 <= weight <= 1.0). Defaults to 1.
            convex_weighting: If True, applies (1-w) to primary loss and (w) to feature loss.
                If False, strictly adds (w * feature_loss) to primary loss. Defaults to False.

        Raises:
            ValueError: If `feature_weight` is not in the [0.0, 1.0] range.
        """
        if convex_weighting and not (0.0 <= feature_weight <= 1.0):
            raise ValueError(
                "For convex weighting, feature_weight must be between 0.0"
                f"and 1.0, got {feature_weight}"
            )

        if convex_weighting:
            super().__init__(
                weighted_losses=[
                    (logit_loss, 1 - feature_weight),
                    (feature_loss, feature_weight),
                ]
            )

        else:
            super().__init__(weighted_losses=[(logit_loss, 1), (feature_loss, feature_weight)])
