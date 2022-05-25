#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import time
import tensorflow as tf

from benchmark_utils import force_gpu_resync


class SyntheticDataset(object):
    def __iter__(self):
        data = 0

    def __init__(self, dataset, device):
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                device,
                buffer_size=tf.data.experimental.AUTOTUNE
            )
        )
        self._ds_iter = iter(dataset)
        self._device = device

    def __iter__(self):

        data_batch = next(self._ds_iter)

        while True:
            yield data_batch


def ensure_dataset_on_gpu(dataset, device):
    try:
        ds_device = dataset._variant_tensor_attr.device.lower()
    except AttributeError:
        return dataset

    if device.lower() not in ds_device:
        return dataset.apply(
            tf.data.experimental.prefetch_to_device(
                device=device,
                buffer_size=tf.data.experimental.AUTOTUNE
            )
        )

    else:
        return dataset


def get_dequeue_batch_fn(ds_iter):

    @force_gpu_resync
    def dequeue_batch_fn():
        """This function should not use tf.function().
        It would create two unwanted effects:
            - The dataset does not stop when it reaches the end
            - A very large overhead is added: 5X slower
        """
        return next(ds_iter)

    return dequeue_batch_fn


def get_force_data_on_gpu_fn(device="/gpu:0", use_xla=False):

    @force_gpu_resync
    @tf.function(jit_compile=use_xla)
    def force_data_on_gpu_fn(data):
        with tf.device(device):
            if isinstance(data, (list, tuple)):
                return tf.identity_n(data)
            elif isinstance(data, dict):
                return dict(zip(data.keys(), tf.identity_n(list(data.values()))))
            else:
                return tf.identity(data)

    return force_data_on_gpu_fn
