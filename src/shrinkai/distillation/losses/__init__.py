"""Knowledge Distillation loss functions.

This module provides a comprehensive suite of loss functions for Knowledge Distillation (KD),
structured into three main categories based on where they operate within the neural network:

* **Logit-based Distillation (Response-based):** Operates on the final output logits.
  By applying temperature scaling, these functions transfer the "dark knowledge" of the
  teacher (relative probabilities assigned to non-target classes) to the student,
  improving its generalization capabilities.
  * *Available Losses:* `HintonLoss`, `PureKDLoss`, `ReverseKLLoss`, `BCEKDLoss`, `JSDLoss`.

* **Feature-based Distillation:** Focuses on aligning intermediate representations
  (hidden layers, attention maps, Gram matrices). Unlike logit-based KD, this forces
  the student to learn the internal reasoning process and hierarchical representations
  of the teacher.
  * *Available Losses:* `FeatureLoss`, `AttentionMapLoss`, `GramMatrixLoss`.

* **Wrappers & Composition:** Orchestration modules that do not compute mathematical
  distances themselves. Instead, they handle tensor dimension mapping (projection)
  and combine multiple distillation objectives into a single criterion.
  * *Available Losses:* `ProjectedFeatureLoss`, `HybridLoss`, `CombinedLoss`.

> **Note on Dimension Alignment:**
> When using feature-based methods, feature dimensions often differ between student and
> teacher models. It is the user's responsibility to apply a projection layer (e.g., a
> 1x1 Conv or a Linear layer) to the student's features. You can use the `ProjectedFeatureLoss`
> wrapper to automate this, though a custom manual projection might sometimes be preferable
> depending on your architecture.
"""

from .base import BaseDistillationLoss
from .features import AttentionMapLoss, FeatureLoss, GramMatrixLoss
from .logits import BCEKDLoss, HintonLoss, JSDLoss, PureKDLoss, ReverseKLLoss
from .wrappers import CombinedLoss, HybridLoss, ProjectedFeatureLoss

__all__ = [
    "BaseDistillationLoss",
    "HintonLoss",
    "PureKDLoss",
    "ReverseKLLoss",
    "BCEKDLoss",
    "JSDLoss",
    "FeatureLoss",
    "AttentionMapLoss",
    "GramMatrixLoss",
    "ProjectedFeatureLoss",
    "HybridLoss",
    "CombinedLoss",
]
