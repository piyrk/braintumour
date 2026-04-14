from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
import os
import sys

import numpy as np
import tensorflow as tf


def _device_name(device: tf.config.PhysicalDevice) -> str:
    details = tf.config.experimental.get_device_details(device)
    return str(details.get("device_name", ""))


def _find_matching_gpu(
    gpus: Iterable[tf.config.PhysicalDevice],
    require_name_contains: str | None,
) -> tf.config.PhysicalDevice | None:
    gpu_list = list(gpus)
    if not gpu_list:
        return None
    if not require_name_contains:
        return gpu_list[0]

    needle = require_name_contains.lower()
    for gpu in gpu_list:
        if needle in _device_name(gpu).lower():
            return gpu
    return None


def _is_windows_directml_runtime() -> bool:
    if os.name != "nt":
        return False
    try:
        metadata.version("tensorflow-directml-plugin")
        return True
    except metadata.PackageNotFoundError:
        return False


def _validate_windows_directml_compatibility() -> None:
    if os.name != "nt":
        return

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "DirectML TensorFlow setup expects Python 3.10 on Windows. "
            f"Current Python is {sys.version_info.major}.{sys.version_info.minor}."
        )

    if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0"):
        raise RuntimeError(
            "DirectML TensorFlow setup requires NumPy < 2. "
            f"Current NumPy is {np.__version__}."
        )

    try:
        metadata.version("tensorflow-directml-plugin")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing tensorflow-directml-plugin. Install dependencies from "
            "requirements.windows_gpu_tf210.txt."
        ) from exc


def configure_training_gpu(require_name_contains: str | None = None, require_gpu: bool = True) -> str:
    """Select a single GPU for TensorFlow training and return its display name."""
    _validate_windows_directml_compatibility()

    gpus = tf.config.list_physical_devices("GPU")
    selected_gpu = _find_matching_gpu(gpus, require_name_contains=require_name_contains)

    if selected_gpu is None and require_gpu:
        available = [name for name in (_device_name(g) for g in gpus) if name]
        requirement = f" matching '{require_name_contains}'" if require_name_contains else ""
        if _is_windows_directml_runtime():
            requirement += " via DirectML"
        raise RuntimeError(
            "No TensorFlow GPU detected"
            f"{requirement}. Available GPU names: {available or ['<none>']}"
        )

    if selected_gpu is None:
        return "CPU"

    tf.config.set_visible_devices(selected_gpu, "GPU")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            # Some backends (including DirectML builds) may not expose memory growth controls.
            continue

    return _device_name(selected_gpu) or selected_gpu.name