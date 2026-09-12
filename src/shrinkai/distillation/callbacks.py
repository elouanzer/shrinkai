"""Ready-to-use training callbacks for `DistillationEngine.fit` / `Distiller.fit`.

A callback is any callable accepting `(epoch: int, metrics: dict[str, float])`, as
already supported by `fit(..., callbacks=[...])`. `EarlyStopping` additionally exposes
a `stop` boolean attribute: after invoking all callbacks, the training loop checks
`getattr(callback, "stop", False)` on each of them and breaks out of the epoch loop
if any callback requests it. Plain function callbacks are unaffected by this check.
"""

import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when a monitored metric has stopped improving.

    Attributes:
        stop (bool): Set to True once `patience` is exhausted. Read by
            `DistillationEngine.fit` after each epoch to interrupt the loop early.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        """Initializes the EarlyStopping callback.

        Args:
            monitor: Metric key to watch in the epoch summary dict (e.g. "val_loss").
            patience: Number of consecutive non-improving epochs tolerated before
                training is stopped.
            mode: "min" if lower values of `monitor` are better, "max" otherwise.
            min_delta: Minimum absolute change to qualify as an improvement.

        Raises:
            ValueError: If `mode` is not "min" or "max".
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Updates internal state and sets `self.stop` if patience is exhausted.

        Args:
            epoch: Current 1-based epoch index.
            metrics: Epoch summary dict, as passed by `DistillationEngine.fit`.
        """
        if self.monitor not in metrics:
            logger.warning(
                "EarlyStopping: metric '%s' not found in epoch %d summary. Skipping check.",
                self.monitor,
                epoch,
            )
            return

        current = metrics[self.monitor]
        if self._is_improvement(current):
            self.best_score = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            logger.info(
                "EarlyStopping: '%s' did not improve for %d epoch(s). Stopping at epoch %d.",
                self.monitor,
                self.patience,
                epoch,
            )
            self.stop = True


class ModelCheckpoint:
    """Saves a model's weights to disk during training.

    Holds a direct reference to the module to save (typically the student), so it
    plugs into `fit(callbacks=[...])` without changing the existing
    `(epoch, metrics) -> None` callback signature.
    """

    def __init__(
        self,
        model: nn.Module,
        filepath: str | Path,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_best_only: bool = True,
    ) -> None:
        """Initializes the ModelCheckpoint callback.

        Args:
            model: The module whose `state_dict()` is saved (e.g. `distiller.student`).
            filepath: Destination path for the saved weights.
            monitor: Metric key to watch when `save_best_only` is True.
            mode: "min" if lower values of `monitor` are better, "max" otherwise.
            save_best_only: If True, only overwrite `filepath` when `monitor` improves
                over its best value so far. If False, save unconditionally every epoch.

        Raises:
            ValueError: If `mode` is not "min" or "max".
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.model = model
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score: float | None = None

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.filepath)

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Saves the model's weights, respecting `save_best_only`.

        Args:
            epoch: Current 1-based epoch index.
            metrics: Epoch summary dict, as passed by `DistillationEngine.fit`.
        """
        if not self.save_best_only:
            self._save()
            return

        if self.monitor not in metrics:
            logger.warning(
                "ModelCheckpoint: metric '%s' not found in epoch %d summary. Skipping save.",
                self.monitor,
                epoch,
            )
            return

        current = metrics[self.monitor]
        if self._is_improvement(current):
            self.best_score = current
            self._save()
            logger.info(
                "ModelCheckpoint: '%s' improved to %.4f at epoch %d. Saved to %s.",
                self.monitor,
                current,
                epoch,
                self.filepath,
            )
