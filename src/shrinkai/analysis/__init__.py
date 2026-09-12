"""Representation-alignment analysis between a teacher and a student model.

`FeatureAnalyzer` runs both models (wrapped in `shrinkai.adapters.FeatureExtractor`)
over a dataloader and scores, layer by layer, how well the student's internal
representations match the teacher's using metrics such as `CKAMetric` (Centered
Kernel Alignment), `RSAMetric` (Representational Similarity Analysis), and
`SpatialAttentionMetric`. This is diagnostic tooling: it helps decide where a
feature-based distillation loss would help most, rather than being a loss itself.
"""

from .analyzer import FeatureAnalyzer, FeatureAnalyzerReport
from .metrics import BaseFeatureMetric, CKAMetric, RSAMetric, SpatialAttentionMetric

__all__ = [
    "FeatureAnalyzerReport",
    "FeatureAnalyzer",
    "BaseFeatureMetric",
    "CKAMetric",
    "RSAMetric",
    "SpatialAttentionMetric",
]
