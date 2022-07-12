#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import time
import tensorflow as tf

from benchmark_autotuner import auto_tf_func_tuner


def SyntheticDataset(dataset, device):
    dataset = dataset.take(count=1)  # loop over 1 batch
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    dataset = ensure_dataset_on_gpu(dataset, device)
    return dataset


def ensure_dataset_on_gpu(dataset, device):

    # ensuring no tensor dtype == int32
    input_batch = next(iter(dataset))
    if isinstance(input_batch, dict):
        input_batch = input_batch.values()
    elif not isinstance(input_batch, (tuple, list)):
        input_batch = [input_batch]

    if any([t.dtype == tf.int32 for t in input_batch]):
        print("[WARNING] The dataloader generates INT32 tensors. Prefetch to "
              "GPU not supported")
        return dataset

    try:
        ds_device = dataset._variant_tensor_attr.device.lower()
    except AttributeError as e:
        print(
            f"[ERROR] Impossible to find the device from the dataset.\n"
            f"Error: {e}."
        )
        return dataset

    if device.lower() not in ds_device:
        print(f"[INFO] Adding prefetch to device `{device}` to the dataset.")
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                device=device,
                buffer_size=tf.data.experimental.AUTOTUNE
            )
        )
        return dataset

    return dataset


def get_dequeue_batch_fn(ds_iter, use_xla=False, use_synthetic_data=False):

    @auto_tf_func_tuner(use_xla=use_xla, use_synthetic_data=use_synthetic_data)
    def dequeue_batch_fn():
        """This function should not use tf.function().
        It would create two unwanted effects:
            - The dataset does not stop when it reaches the end
            - A very large overhead is added: 5X slower
        """
        return next(ds_iter)

    return dequeue_batch_fn


def get_force_data_on_gpu_fn(device="/gpu:0", use_xla=False, use_synthetic_data=False):

    @auto_tf_func_tuner(use_xla=use_xla, use_synthetic_data=use_synthetic_data)
    def force_data_on_gpu_fn(data):
        with tf.device(device):
            if isinstance(data, (list, tuple)):
                return tf.identity_n(data)
            elif isinstance(data, dict):
                return dict(zip(data.keys(), tf.identity_n(list(data.values()))))
            else:
                return tf.identity(data)

    return force_data_on_gpu_fn
