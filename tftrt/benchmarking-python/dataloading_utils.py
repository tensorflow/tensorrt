#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import itertools
import time
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils

from benchmark_autotuner import auto_tf_func_tuner
from benchmark_logger import logging
from benchmark_utils import force_gpu_resync


def SyntheticDataset(dataset, device):
    data_batch = next(iter(dataset))

    def copy_on_device(data):
        if isinstance(data, (tuple, list)):
            return [copy_on_device(t) for t in data]
        elif isinstance(data, dict):
            return {k: copy_on_device(t) for k, t in data.items()}
        else:
            try:
                if data.dtype != tf.int32:
                    with tf.device(device):
                        return tf.identity(data)
            except AttributeError:
                pass

            return data

    data_batch = copy_on_device(data_batch)

    return itertools.repeat(data_batch)


def _validate_data_gpu_compatible(data):
    if isinstance(data, dict):
        return all([_validate_data_gpu_compatible(x) for x in data.values()])

    elif isinstance(data, (tuple, list)):
        return all([_validate_data_gpu_compatible(x) for x in data])

    else:
        return data.dtype != tf.int32


def ensure_dataset_on_gpu(dataset, device):

    # ensuring no tensor dtype == int32
    input_batch = next(iter(dataset))

    if not _validate_data_gpu_compatible(input_batch):
        logging.warning(
            "The dataloader generates INT32 tensors. Prefetch to "
            "GPU not supported"
        )
        return dataset

    try:
        ds_device = dataset._variant_tensor_attr.device.lower()
    except AttributeError as e:
        logging.error(
            f"Impossible to find the device from the dataset.\n"
            f"Error: {e}."
        )
        return dataset

    if device.lower() not in ds_device:
        logging.info(f"Adding prefetch to device `{device}` to the dataset.")
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                device=device, buffer_size=tf.data.AUTOTUNE
            )
        )
        return dataset

    return dataset


def get_dequeue_batch_fn(ds_iter, use_xla=False, use_synthetic_data=False):

    @force_gpu_resync
    def dequeue_batch_fn():
        """This function should not use tf.function().
        It would create two unwanted effects:
            - The dataset does not stop when it reaches the end
            - A very large overhead is added: 5X slower
        """
        return next(ds_iter)

    return dequeue_batch_fn


def get_force_data_on_gpu_fn(
    device="/gpu:0", use_xla=False, use_synthetic_data=False
):

    @auto_tf_func_tuner(use_xla=use_xla, use_synthetic_data=use_synthetic_data)
    def force_data_on_gpu_fn(data):
        with tf.device(device):
            if isinstance(data, (list, tuple)):
                return tf.identity_n(data)
            elif isinstance(data, dict):
                return dict(
                    zip(data.keys(), tf.identity_n(list(data.values())))
                )
            else:
                return tf.identity(data)

    return force_data_on_gpu_fn


def patch_dali_dataset(dataset):
    import nvidia.dali.plugin.tf as dali_tf

    if not isinstance(dataset, dali_tf.DALIDataset):
        raise TypeError(
            "Dataset supplied should be an instance of `DALIDataset`."
            f"Received: `{type(dataset)}`"
        )

    def take(self, limit):

        class _Dataset(self.__class__):

            def __init__(self, _ds, _limit):
                self._ds = _ds
                self._limit = _limit

            def __iter__(self):
                ds_iter = iter(self._ds)
                for idx in tf.range(self._limit):
                    yield next(ds_iter)

        return _Dataset(self, limit)

    # Monkey Patch
    dataset.__class__.take = take

    return dataset



def _get_signature_def_map(saved_model_dir, tag_set):
  """Gets SignatureDef map from a MetaGraphDef in a SavedModel.
  Returns the SignatureDef map for the given tag-set in the SavedModel
  directory.
  Args:
    saved_model_dir: Directory containing the SavedModel to inspect or execute.
    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
        string format, separated by ','. For tag-set contains multiple tags, all
        tags must be passed in.
  Returns:
    A SignatureDef map that maps from string keys to SignatureDefs.
  """
  meta_graph = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
  return meta_graph.signature_def


def _extract_dtype(tensor_info):
    return tf.dtypes.DType(tensor_info.dtype)


def _tensor_shape_to_list(tensor_info):
    tensor_shape = tensor_info.tensor_shape
    if tensor_shape.unknown_rank:
        raise ValueError()
    else:
        return [dim.size for dim in tensor_shape.dim]


def model_input_data(saved_model_dir, tag_set, input_signature_key):
    signature_def = _get_signature_def_map(saved_model_dir, tag_set)
    func_inputs = signature_def[input_signature_key].inputs

    input_data = list()

    for key, tensor_info in func_inputs.items():
        input_data.append({
            "key": key,
            "name": tensor_info.name.split(":")[0],
            "shape": _tensor_shape_to_list(tensor_info),
            "dtype": _extract_dtype(tensor_info),
        })

    return input_data


def get_random_tensor(shape, dtype):
    if dtype == tf.uint8:
        _dtype = tf.int32
        maxval = 255
    elif dtype == tf.bool:
        _dtype = tf.int32
        maxval = 2
    elif dtype in [tf.int32, tf.int64]:
        _dtype = dtype
        maxval = 255
    else:
        _dtype = dtype
        maxval = None

    shape[0] = 1 if shape[0] == -1 else shape[0]
    if any([d == -1 for d in shape]):
        raise ValueError("Unknown dimension found in the inputs")

    t = tf.random.uniform(shape=shape, dtype=_dtype, maxval=maxval)

    if dtype != _dtype:
        return tf.cast(t, dtype)
    return t
