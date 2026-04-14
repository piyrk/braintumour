from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(path: str | Path, image_size: tuple[int, int], grayscale: bool = True) -> np.ndarray:
    image = Image.open(path)
    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    image = image.resize(image_size)
    array = np.asarray(image).astype("float32") / 127.5 - 1.0
    if grayscale:
        array = array[..., np.newaxis]
    return array


def load_mask(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize(image_size)
    array = np.asarray(mask).astype("float32") / 255.0
    return array[..., np.newaxis]


def count_files(folder: str | Path, suffixes=(".png", ".jpg", ".jpeg")) -> int:
    folder = Path(folder)
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in suffixes)


def _dataset_from_paths(paths, loader, batch_size: int, shuffle: bool = True):
    paths = [str(path) for path in paths]
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        value = tf.numpy_function(loader, [path], tf.float32)
        return value

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 1024))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_binary_detection_dataset(root_dir: str | Path, image_size=(224, 224), batch_size: int = 32, shuffle: bool = True):
    root_dir = Path(root_dir)
    class_names = ["normal", "tumour"]
    image_paths = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                image_paths.append(path)
                labels.append(class_index)

    def _load(path, label):
        image = tf.numpy_function(lambda p: load_image(p.decode("utf-8"), image_size), [path], tf.float32)
        image.set_shape([image_size[0], image_size[1], 1])
        label = tf.cast(label, tf.float32)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((
        [str(path) for path in image_paths],
        labels,
    )).map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1024))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), class_names


def load_multiclass_dataset(root_dir: str | Path, image_size=(224, 224), batch_size: int = 16, shuffle: bool = True):
    root_dir = Path(root_dir)
    class_names = [path.name for path in sorted(root_dir.iterdir()) if path.is_dir()]
    image_paths = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                image_paths.append(path)
                labels.append(class_index)

    def _load(path, label):
        image = tf.numpy_function(lambda p: load_image(p.decode("utf-8"), image_size), [path], tf.float32)
        image.set_shape([image_size[0], image_size[1], 1])
        label = tf.one_hot(tf.cast(label, tf.int32), depth=len(class_names))
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((
        [str(path) for path in image_paths],
        labels,
    )).map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1024))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), class_names


def load_segmentation_dataset(images_dir: str | Path, masks_dir: str | Path, image_size=(256, 256), batch_size: int = 8, shuffle: bool = True):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_paths = [
        path for path in sorted(images_dir.rglob("*"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    mask_paths = []
    for image_path in image_paths:
        candidate = masks_dir / image_path.name
        if candidate.exists():
            mask_paths.append(candidate)
            continue
        stem_candidate = masks_dir / f"{image_path.stem}.png"
        if stem_candidate.exists():
            mask_paths.append(stem_candidate)
            continue
        raise FileNotFoundError(f"Missing mask for {image_path.name}")

    def _load(image_path, mask_path):
        image = tf.numpy_function(lambda p: load_image(p.decode("utf-8"), image_size), [image_path], tf.float32)
        image.set_shape([image_size[0], image_size[1], 1])
        mask = tf.numpy_function(lambda p: load_mask(p.decode("utf-8"), image_size), [mask_path], tf.float32)
        mask.set_shape([image_size[0], image_size[1], 1])
        return image, mask

    ds = tf.data.Dataset.from_tensor_slices((
        [str(path) for path in image_paths],
        [str(path) for path in mask_paths],
    )).map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1024))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_unlabeled_image_dataset(image_dir: str | Path, image_size=(64, 64), batch_size: int = 64, shuffle: bool = True):
    image_dir = Path(image_dir)
    image_paths = [
        path for path in sorted(image_dir.rglob("*"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    def _load(path):
        image = tf.numpy_function(lambda p: load_image(p.decode("utf-8"), image_size), [path], tf.float32)
        image.set_shape([image_size[0], image_size[1], 1])
        return image

    ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths]).map(
        _load,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1024))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_dataset(dataset, validation_fraction: float = 0.2):
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    if total_batches < 0:
        return dataset, None
    validation_batches = max(1, int(total_batches * validation_fraction))
    train_batches = max(0, total_batches - validation_batches)
    train_ds = dataset.take(train_batches)
    val_ds = dataset.skip(train_batches)
    return train_ds, val_ds
