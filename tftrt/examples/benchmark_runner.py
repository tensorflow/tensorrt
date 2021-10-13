#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import abc
import argparse
import copy
import logging
import time

from contextlib import contextmanager
from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt.trt_convert import \
    DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_constants import \
    DEFAULT_SERVING_SIGNATURE_DEF_KEY

import utils as tftrt_utils


class BaseBenchmarkRunner(object, metaclass=abc.ABCMeta):

    ACCURACY_METRIC_NAME = None

    ############################################################################
    # Methods expected to be overwritten by the subclasses
    ############################################################################

    def before_benchmark(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_accuracy_metric(self, batch_size, steps_executed, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def process_model_output(self, outputs, batch_y, **kwargs):
        raise NotImplementedError()

    ############################################################################
    # Common methods for all the benchmarks
    ############################################################################

    def __init__(
        self,
        input_saved_model_dir,
        output_saved_model_dir,
        allow_build_at_runtime=False,
        calibration_input_fn=None,
        gpu_mem_cap=None,
        input_signature_key=DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
        minimum_segment_size=5,
        num_calib_inputs=None,
        optimize_offline=False,
        optimize_offline_input_fn=None,
        precision_mode=None,
        use_dynamic_shape=False,
        use_tftrt=False):

        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.disable(logging.WARNING)

        tftrt_utils.config_gpu_memory(gpu_mem_cap)

        calibration_input_fn = (
            None
            if precision_mode != 'INT8' else
            calibration_input_fn
        )

        optimize_offline_input_fn = (
            None
            if not optimize_offline and not use_dynamic_shape else
            optimize_offline_input_fn
        )

        self._graph_func = tftrt_utils.get_graph_func(
            input_saved_model_dir=input_saved_model_dir,
            output_saved_model_dir=output_saved_model_dir,
            allow_build_at_runtime=allow_build_at_runtime,
            calibration_input_fn=calibration_input_fn,
            input_signature_key=input_signature_key,
            max_workspace_size_bytes=max_workspace_size_bytes,
            minimum_segment_size=minimum_segment_size,
            num_calib_inputs=num_calib_inputs,
            optimize_offline=optimize_offline,
            optimize_offline_input_fn=optimize_offline_input_fn,
            precision_mode=precision_mode,
            use_dynamic_shape=use_dynamic_shape,
            use_tftrt=use_tftrt
        )

    def execute_benchmark(
        self,
        batch_size,
        display_every,
        get_benchmark_input_fn,
        num_iterations,
        num_warmup_iterations,
        skip_accuracy_testing,
        use_synthetic_data,
        **kwargs):
        """Run the given graph_func on the data files provided.
        It consumes TFRecords with labels and reports accuracy.
        """

        self.before_benchmark(**kwargs)

        results = {}
        iter_times = []
        steps_executed = 0

        dataset = get_benchmark_input_fn(
            batch_size=batch_size,
            use_synthetic_data=use_synthetic_data,
        )

        @tf.function
        def infer_step(_batch_x):
          return self._graph_func(_batch_x)

        print("\nStart inference ...")
        for i, data_batch in enumerate(dataset):

            if isinstance(data_batch, (list, tuple)):
                if len(data_batch) == 1:
                    batch_x, batch_y = (data_batch, None)
                elif len(data_batch) == 2:
                    batch_x, batch_y = data_batch
                else:
                    raise RuntimeError("Error: The dataset function returned "
                                       "%d elements." % len(data_batch))
            # TF Tensor
            else:
                batch_x, batch_y = (data_batch, None)

            start_time = time.time()
            batch_preds = infer_step(batch_x)
            iter_times.append(time.time() - start_time)

            steps_executed += 1

            if (i + 1) % display_every == 0 or (i + 1) == num_iterations:
                print("  step %04d/%04d, iter_time(ms)=%.0f" % (
                    i + 1,
                    num_iterations,
                    np.mean(iter_times[-display_every:]) * 1000
                ))

            if not skip_accuracy_testing:
                self.process_model_output(
                    outputs=batch_preds,
                    batch_y=batch_y
                )

            if (i + 1) >= num_iterations:
                break

        if not skip_accuracy_testing:
            results['accuracy_metric'] = self.compute_accuracy_metric(
                batch_size=batch_size,
                steps_executed=steps_executed
            )

        iter_times = np.array(iter_times)
        run_times = iter_times[num_warmup_iterations:]

        results['total_time(s)'] = int(np.sum(iter_times))
        results['samples/sec'] = int(np.mean(batch_size / run_times))
        results['99th_percentile(ms)'] = np.percentile(
            run_times, q=99, interpolation='lower'
        ) * 1000
        results['latency_mean(ms)'] = np.mean(run_times) * 1000
        results['latency_median(ms)'] = np.median(run_times) * 1000
        results['latency_min(ms)'] = np.min(run_times) * 1000
        results['latency_max(ms)'] = np.max(run_times) * 1000

        print('\n=============================================\n')
        print('Results:\n')

        if "accuracy_metric" in results:
            print('  {}: {:.2f}'.format(
                self.ACCURACY_METRIC_NAME, results['accuracy_metric'] * 100))
            del results['accuracy_metric']

        for key, val in sorted(results.items()):
            if isinstance(val, float):
                print("  {}: {:.2f}".format(key, val))
            else:
                print("  {}: {}".format(key, val))
