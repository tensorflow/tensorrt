#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import json
import time

import numpy as np
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from benchmark_logger import logging


def force_gpu_resync(func):

    func_name = func.__name__
    try:
        if hasattr(tf.experimental, "sync_devices"):
            sync_device_fn = tf.experimental.sync_devices

        elif hasattr(tf.test.experimental, "sync_devices"):
            sync_device_fn = tf.test.experimental.sync_devices
            
        else:
            raise AttributeError("`sync_devices` API does not exist in this "
                                 "version of TensorFlow")

        logging.debug(
            f"Using API `tf.{sync_device_fn._tf_api_names[0]}` to resync GPUs "
            f"on function: {func_name}."
        )

    except AttributeError:
        logging.warning(
            "TensorFlow API to force GPU resync is not available in your "
            "version of TensorFlow. Please upgrade to TensorFlow >= `2.12.0`. "
            "Non negligeable overhead might be present on function: "
            f"{func_name}."
        )

        p = tf.constant(0.)  # Create small tensor to force GPU resync
        sync_device_fn = lambda: (p + 1.).numpy()  # Sync the GPU

    def wrapper(*args, **kwargs):
        rslt = func(*args, **kwargs)
        sync_device_fn()
        return rslt

    return wrapper


def print_dict(input_dict, prefix='\t', postfix='', redirect_to_str=False):
    rslt_str = ""
    for key, val in sorted(input_dict.items()):
        val = f"{val:.1f}" if isinstance(val, float) else val
        tmp_str = f"{prefix}- {key}: {val}{postfix}"
        if not redirect_to_str:
            logging.info(tmp_str)
        else:
            rslt_str += f"{tmp_str}\n"

    if redirect_to_str:
        return rslt_str.rstrip()


@contextmanager
def timed_section(msg, activate=True, start_end_mode=True):
    if activate:

        if start_end_mode:
            logging.info(f"[START] {msg} ...")

        start_time = time.perf_counter()
        yield
        total_time = time.perf_counter() - start_time

        if start_end_mode:
            logging.info(f"[END] {msg} - Duration: {total_time:.1f}s")
            logging.info("="*80 + "\n")
        else:
            logging.info(f"{msg:18s}: {total_time:.4f}s")

    else:
        yield


def _format_output_tensors(predictions, expected, batch_size):

    def dictionarize(data):
        tmp_preds = dict()
        if isinstance(data, (tuple, list)):
            for idx, pred_i in enumerate(data):
                tmp_preds[f"data_{idx:03d}"] = pred_i
        elif not isinstance(data, dict):
            tmp_preds["data"] = data
        else:
            tmp_preds = data
        return tmp_preds

    def format(data):

        def _format(tensor):
            if tensor.shape[0] != batch_size:
                tensor = np.expand_dims(tensor, 0)
            elif len(tensor.shape) == 1:
                tensor = np.expand_dims(tensor, 1)
            return tensor

        for key, val in data.items():
            data[key] = _format(val)

        return data

    predictions = format(dictionarize(predictions))
    expected = format(dictionarize(expected))

    return predictions, expected


def generate_json_metrics(metrics, args):
    metric_dict = {"results": metrics, "runtime_arguments": args}

    json_string = json.dumps(
        metric_dict, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )
    return json_string


class DataAggregator(object):

    def __init__(self, postprocess_model_outputs_fn, args):

        self._args = args

        self._predicted = dict()
        self._expected = dict()

        self._total_samples_processed = 0

        self._postprocess_model_outputs_fn = postprocess_model_outputs_fn

    def _calc_step_batchsize(self, data_arr):
        if isinstance(data_arr, (list, tuple)):
            return data_arr[0].shape[0]
        elif isinstance(data_arr, dict):
            return list(data_arr.values())[0].shape[0]
        else:  # TF.Tensor or TF.EagerTensor
            return data_arr.shape[0]

    @property
    def predicted_dict(self):
        tmp_data = dict()
        for key, val in self._predicted.items():
            tmp_data[key] = val[:self._total_samples_processed]
        return tmp_data

    @property
    def expected_dict(self):
        tmp_data = dict()
        for key, val in self._expected.items():
            tmp_data[key] = val[:self._total_samples_processed]
        return tmp_data

    @property
    def total_samples_processed(self):
        return self._total_samples_processed

    def aggregate_data(self, y_pred, y):

        with timed_section("Processing Time",
                           activate=self._args.debug_performance,
                           start_end_mode=False):

            step_batch_size = self._calc_step_batchsize(y_pred)

            y_pred, y = self._postprocess_model_outputs_fn(
                predictions=y_pred, expected=y
            )

            y_pred, y = _format_output_tensors(
                y_pred, y, batch_size=step_batch_size
            )

            if not self._predicted:  # First call
                for key, val in y_pred.items():
                    self._predicted[key] = np.empty(
                        [self._args.total_max_samples] + list(val.shape[1:]),
                        dtype=val.dtype
                    )

            if not self._expected:  # First call
                for key, val in y.items():
                    self._expected[key] = np.empty(
                        [self._args.total_max_samples] + list(val.shape[1:]),
                        dtype=val.dtype
                    )

            idx_start = self._total_samples_processed

            self._total_samples_processed += step_batch_size
            idx_stop = self._total_samples_processed

            if self._args.debug_data_aggregation:
                logging.debug(
                    f"Start: {idx_start} - Stop: {idx_stop} - "
                    f"Size: {idx_stop-idx_start} - Step Batch Size: {step_batch_size}"
                )

            with timed_section("Numpy Copy Time",
                               activate=self._args.debug_performance,
                               start_end_mode=False):
                for key, val in self._predicted.items():
                    if self._args.debug_data_aggregation:
                        logging.debug(
                            f"\t-Key: {key} - "
                            f"Storage Shape: {self._predicted[key][idx_start:idx_stop].shape} - "
                            f"Preds: {y_pred[key].shape}"
                        )
                    self._predicted[key][idx_start:idx_stop] = y_pred[key]
                for key, val in self._expected.items():
                    if self._args.debug_data_aggregation:
                        logging.debug(
                            f"\t-Key: {key} - "
                            f"Storage Shape: {self._expected[key][idx_start:idx_stop].shape} - "
                            f"Expected: {y[key].shape}"
                        )
                    self._expected[key][idx_start:idx_stop] = y[key]


def load_model_from_disk(
    path,
    tags=[tag_constants.SERVING],
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    precision="FP32"
):

    tf.config.optimizer.set_experimental_options({
        "disable_model_pruning": False,
        "debug_stripper": True,
        "auto_mixed_precision": precision != "FP32",
        "layout_optimizer": True,
        "dependency_optimization": True,
        "min_graph_nodes": -1  # do not skip small graphs
    })

    saved_model_loaded = tf.saved_model.load(export_dir=path, tags=tags)

    graph_func = saved_model_loaded.signatures[signature_key]

    # Known TF Issue: https://github.com/tensorflow/tensorflow/issues/37615#issuecomment-767804930
    # it looks like if the original trackable object is released by
    # the Python garbage collector once it goes out of scope, and
    # the signature returned by the function does not maintain a
    # back-reference to the original loaded object.
    graph_func._backref_to_saved_model = saved_model_loaded

    return graph_func
