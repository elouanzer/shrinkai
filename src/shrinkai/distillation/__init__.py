"""Knowledge distillation: training a small student model to mimic a larger teacher.

`Distiller` is the high-level facade most users start from: give it a teacher,
a student, and a loss (`shrinkai.distillation.losses`), and it handles the
training loop (`fit`), evaluation, benchmarking, checkpointing, and deployment
export. `DistillationEngine` is the lower-level training loop it delegates to
(device management, mixed precision, gradient clipping, teacher freezing, ...),
usable directly for custom orchestration. `EarlyStopping`/`ModelCheckpoint` are
ready-to-use `fit(callbacks=[...])` callbacks.
"""

from .callbacks import EarlyStopping, ModelCheckpoint
from .distiller import Distiller
from .engine import DistillationEngine

__all__ = ["DistillationEngine", "Distiller", "EarlyStopping", "ModelCheckpoint"]
