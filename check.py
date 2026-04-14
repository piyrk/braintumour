from importlib import metadata
import sys

import numpy as np
import tensorflow as tf


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"Python 3.10 is required for this DirectML setup. Found {sys.version_info.major}.{sys.version_info.minor}."
        )

    if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0"):
        raise RuntimeError(f"NumPy < 2 is required. Found {np.__version__}.")

    try:
        dml_version = metadata.version("tensorflow-directml-plugin")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("tensorflow-directml-plugin is not installed.") from exc

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No TensorFlow GPU device found. DirectML is not active.")

    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"tensorflow-directml-plugin: {dml_version}")
    print(f"Detected GPU devices: {gpus}")

    with tf.device("/GPU:0"):
        a = tf.random.normal([2048, 2048])
        b = tf.random.normal([2048, 2048])
        c = tf.matmul(a, b)

    print(f"MatMul output shape: {c.shape}")
    print(f"Operation device: {c.device}")
    print("DirectML GPU test completed.")


if __name__ == "__main__":
    main()