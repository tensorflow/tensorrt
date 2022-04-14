#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import tensorflow as tf


class SyntheticDataset(object):
    def __iter__(self):
        data = 0

    def __init__(self, dataset, device):

        self._ds_iter = iter(dataset)
        self._device = device

    def __iter__(self):

        with tf.device(self._device):

            data_batch = None
            tf.random.set_seed(666)

            def get_random_tensor(t_shape, t_dtype):
                if t_dtype == tf.bool:
                    return (
                        tf.random.uniform(shape=t_shape, dtype=tf.float32) < 0.5
                    )

                elif t_dtype in [tf.int32, tf.int64]:
                    return tf.random.uniform(
                        shape=t_shape, dtype=t_dtype, maxval=5
                    )

                else:
                    return tf.random.uniform(shape=t_shape, dtype=t_dtype)

            ds_batch = next(self._ds_iter)

            if isinstance(ds_batch, (list, tuple)):
                data_batch = list()
                for t in ds_batch:
                    data_batch.append(get_random_tensor(t.shape, t.dtype))

            elif isinstance(ds_batch, dict):
                data_batch = dict()
                for k, v in ds_batch.items():
                    data_batch[k] = get_random_tensor(v.shape, v.dtype)

            else:
                data_batch = get_random_tensor(ds_batch.shape, ds_batch.dtype)

            while True:
                yield data_batch
