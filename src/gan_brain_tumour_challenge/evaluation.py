from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import linalg
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf


def detection_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    y_hat = (y_pred >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_hat).tolist(),
    }


def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = np.asarray(y_true).astype(np.float32).ravel()
    y_pred = np.asarray(y_pred).astype(np.float32).ravel()
    intersection = np.sum(y_true * y_pred)
    return float((2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth))


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = np.asarray(y_true).astype(np.float32).ravel()
    y_pred = np.asarray(y_pred).astype(np.float32).ravel()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


@lru_cache(maxsize=1)
def feature_extractor():
    with tf.device("/CPU:0"):
        return tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(299, 299, 3),
        )


def resize_for_fid(images):
    images = tf.convert_to_tensor(images)
    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize(images, (299, 299))
    images = (images + 1.0) * 127.5 if images.dtype.is_floating else tf.cast(images, tf.float32)
    return images


def activations(images):
    with tf.device("/CPU:0"):
        model = feature_extractor()
        images = resize_for_fid(images)
        return model(images, training=False).numpy()


def frechet_distance(real_images, fake_images):
    real_act = activations(real_images)
    fake_act = activations(fake_images)

    mu1, sigma1 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu2, sigma2 = fake_act.mean(axis=0), np.cov(fake_act, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def fs_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))
