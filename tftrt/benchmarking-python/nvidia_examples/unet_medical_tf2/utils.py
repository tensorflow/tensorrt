import numpy as np

from collections import deque
from PIL import (Image, ImageSequence)

import tensorflow as tf


def get_val_train_indices(length, fold, ratio=0.8):
    if ratio <= 0 or ratio > 1:
        raise ValueError("Train/total data ratio must be in range (0.0, 1.0]")

    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int)
    np.random.shuffle(indices)
    indices = deque(indices)
    indices.rotate(fold * int((1.0-ratio) * length))
    indices = np.array(indices)
    train_indices = indices[:int(ratio * len(indices))]
    val_indices = indices[int(ratio * len(indices)):]

    return train_indices, val_indices


def load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return np.array([
        np.array(p) for p in ImageSequence.Iterator(Image.open(path))
    ])


def normalize_inputs(inputs):
    """Normalize inputs"""
    inputs = tf.expand_dims(tf.cast(inputs, tf.float32), -1)

    # Center around zero
    inputs = tf.divide(inputs, 127.5) - 1
    # Resize to match output size
    inputs = tf.image.resize(inputs, (388, 388))

    return tf.image.resize_with_crop_or_pad(inputs, 572, 572)


def normalize_labels(labels):
    """Normalize labels"""
    labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
    labels = tf.divide(labels, 255)

    # Resize to match output size
    labels = tf.image.resize(labels, (388, 388))
    labels = tf.image.resize_with_crop_or_pad(labels, 572, 572)

    cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    labels = tf.where(
        cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels))
    )

    return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)


def preproc_samples(inputs, labels, precision):
    """Preprocess samples and perform random augmentations"""
    inputs = normalize_inputs(inputs)
    labels = normalize_labels(labels)

    # Bring back labels to network's output size and remove interpolation artifacts
    labels = tf.image.resize_with_crop_or_pad(
        labels, target_width=388, target_height=388
    )
    cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
    labels = tf.where(
        cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels))
    )
    return tf.cast(inputs, precision), labels


def dice_coef(predict, target, axis=1, eps=1e-6):
    from scipy.special import softmax

    predict = softmax(predict, axis=-1)

    intersection = np.sum(predict * target, axis=axis)
    union = np.sum(predict*predict + target*target, axis=axis)

    dice = (2.*intersection + eps) / (union+eps)

    return np.mean(dice)
