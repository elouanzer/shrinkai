from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from shrinkai.distillation.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# 1. EARLY STOPPING
# ==========================================


def test_early_stopping_rejects_invalid_mode():
    """Verifies that an unsupported mode string raises a ValueError."""
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        EarlyStopping(mode="invalid")


def test_early_stopping_min_mode_triggers_after_patience():
    """Verifies training is flagged to stop once patience is exhausted (min mode)."""
    early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")

    early_stop(1, {"val_loss": 1.0})
    assert not early_stop.stop

    # No improvement for 2 consecutive epochs -> patience exhausted
    early_stop(2, {"val_loss": 1.1})
    assert not early_stop.stop
    early_stop(3, {"val_loss": 1.2})
    assert early_stop.stop


def test_early_stopping_resets_patience_on_improvement():
    """Verifies that an improving metric resets the bad-epoch counter."""
    early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")

    early_stop(1, {"val_loss": 1.0})
    early_stop(2, {"val_loss": 1.1})  # 1 bad epoch
    early_stop(3, {"val_loss": 0.5})  # improvement resets counter
    assert not early_stop.stop
    assert early_stop.num_bad_epochs == 0


def test_early_stopping_max_mode():
    """Verifies max mode considers higher values as improvements."""
    early_stop = EarlyStopping(monitor="val_accuracy", patience=1, mode="max")

    early_stop(1, {"val_accuracy": 80.0})
    early_stop(2, {"val_accuracy": 79.0})  # worse -> 1 bad epoch, patience exhausted
    assert early_stop.stop


def test_early_stopping_ignores_missing_metric():
    """Verifies that a missing monitored key is skipped without raising."""
    early_stop = EarlyStopping(monitor="val_loss", patience=1)
    early_stop(1, {"train_loss": 1.0})  # no 'val_loss' key
    assert not early_stop.stop
    assert early_stop.best_score is None


# ==========================================
# 2. MODEL CHECKPOINT
# ==========================================


def test_model_checkpoint_rejects_invalid_mode():
    """Verifies that an unsupported mode string raises a ValueError."""
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        ModelCheckpoint(nn.Linear(2, 2), filepath="unused.pt", mode="invalid")


def test_model_checkpoint_saves_only_on_improvement(tmp_path: Path):
    """Verifies save_best_only=True only writes the file when the metric improves."""
    model = nn.Linear(2, 2)
    filepath = tmp_path / "best.pt"
    checkpoint = ModelCheckpoint(model, filepath=filepath, monitor="val_loss", mode="min")

    checkpoint(1, {"val_loss": 1.0})
    assert filepath.exists()
    first_mtime = filepath.stat().st_mtime_ns

    checkpoint(2, {"val_loss": 1.5})  # worse, should not overwrite
    assert filepath.stat().st_mtime_ns == first_mtime

    checkpoint(3, {"val_loss": 0.5})  # improvement, should overwrite
    assert filepath.stat().st_mtime_ns >= first_mtime


def test_model_checkpoint_saves_every_epoch_when_disabled(tmp_path: Path):
    """Verifies save_best_only=False saves unconditionally at every call."""
    model = nn.Linear(2, 2)
    filepath = tmp_path / "every_epoch.pt"
    checkpoint = ModelCheckpoint(model, filepath=filepath, save_best_only=False)

    checkpoint(1, {"val_loss": 1.0})
    assert filepath.exists()

    checkpoint(2, {"val_loss": 999.0})  # worse metric, still saved
    assert filepath.exists()


def test_model_checkpoint_skips_when_metric_missing(tmp_path: Path):
    """Verifies no file is written if the monitored metric is absent."""
    model = MagicMock(spec=nn.Module)
    filepath = tmp_path / "unused.pt"
    checkpoint = ModelCheckpoint(model, filepath=filepath, monitor="val_loss")

    checkpoint(1, {"train_loss": 1.0})
    assert not filepath.exists()
    model.state_dict.assert_not_called()
