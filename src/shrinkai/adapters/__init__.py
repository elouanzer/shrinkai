"""Model-agnostic adapters bridging teacher/student architectures for distillation.

`FeatureExtractor` wraps any `nn.Module` to capture intermediate activations via
forward hooks, without modifying the model's source code. `FeatureProjector` and
`AttentionHeadSelector` then reconcile dimension mismatches between a teacher's
and a student's internal representations (channel counts, attention head
counts, ...) so that feature-based losses (`shrinkai.distillation.losses`) can
compare them directly.
"""

from .extractor import FeatureExtractor
from .projector import AttentionHeadSelector, FeatureProjector

__all__ = ["FeatureExtractor", "FeatureProjector", "AttentionHeadSelector"]
