from __future__ import annotations

from pathlib import Path
import re

import tensorflow as tf


_EPOCH_CHECKPOINT_RE = re.compile(r"epoch_(\d{3})\.keras$")


def _checkpoint_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoints"


def _latest_epoch_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoint_dir = _checkpoint_dir(output_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = sorted(checkpoint_dir.glob("epoch_*.keras"))
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: int(_EPOCH_CHECKPOINT_RE.search(path.name).group(1)) if _EPOCH_CHECKPOINT_RE.search(path.name) else -1,
    )


def resume_training_model(model, output_dir: str | Path):
    latest_checkpoint = _latest_epoch_checkpoint(output_dir)
    if latest_checkpoint is None:
        return model, 0

    match = _EPOCH_CHECKPOINT_RE.search(latest_checkpoint.name)
    initial_epoch = int(match.group(1)) if match else 0
    resumed_model = tf.keras.models.load_model(latest_checkpoint)
    return resumed_model, initial_epoch


def make_callbacks(output_dir: str | Path, monitor: str = "val_loss"):
    output_dir = Path(output_dir)
    checkpoint_dir = _checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_logger_path = output_dir / "training_log.csv"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
            save_best_only=False,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_logger_path), append=csv_logger_path.exists()),
    ]
