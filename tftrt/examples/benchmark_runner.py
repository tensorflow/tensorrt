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


def _print_dict(input_dict, prefix='  ', postfix=''):
    for k, v in sorted(input_dict.items()):
        print('{prefix}{arg_name}: {value}{postfix}'.format(
            prefix=prefix,
            arg_name=k,
            value='%.1f' % v if isinstance(v, float) else v,
            postfix=postfix
        ))


@contextmanager
def _timed_section(msg):
    print('\n[START] {}'.format(msg))
    start_time = time.time()
    yield
    print("[END] Duration: {:.1f}s".format(time.time() - start_time))
    print("=" * 80, "\n")


def _force_gpu_resync(func):
    p = tf.constant(0.)  # Create small tensor to force GPU resync
    def wrapper(*args, **kwargs):
        rslt = func(*args, **kwargs)
        (p + 1.).numpy()  # Sync the GPU
        return rslt
    return wrapper


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
        use_tftrt=False
    ):

        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.disable(logging.WARNING)

        # TensorFlow can execute operations synchronously or asynchronously.
        # If asynchronous execution is enabled, operations may return
        # "non-ready" handles.
        tf.config.experimental.set_synchronous_execution(True)

        self._config_gpu_memory(gpu_mem_cap)

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

        self._graph_func = self._get_graph_func(
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

    def _config_gpu_memory(self, gpu_mem_cap):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if not gpus:
            raise RuntimeError("No GPUs has been found.")

        print('Found the following GPUs:')
        for gpu in gpus:
            print(' ', gpu)

        for gpu in gpus:
            try:
                if not gpu_mem_cap:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_mem_cap)])
            except RuntimeError as e:
                print('Can not set GPU memory config', e)

    def _get_graph_func(
        self,
        input_saved_model_dir,
        output_saved_model_dir,
        allow_build_at_runtime=False,
        calibration_input_fn=None,
        input_signature_key=DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
        minimum_segment_size=5,
        num_calib_inputs=None,
        optimize_offline=False,
        optimize_offline_input_fn=None,
        precision_mode=None,
        use_dynamic_shape=False,
        use_tftrt=False):
        """Retreives a frozen SavedModel and applies TF-TRT
        use_tftrt: bool, if true use TensorRT
        precision: str, floating point precision (FP32, FP16, or INT8)
        returns: TF function that is ready to run for inference
        """

        if not use_tftrt:

            with _timed_section('Loading TensorFlow native model...'):
                saved_model_loaded = tf.saved_model.load(
                    input_saved_model_dir, tags=[tag_constants.SERVING]
                )

                graph_func = saved_model_loaded.signatures[input_signature_key]
                graph_func = convert_to_constants.convert_variables_to_constants_v2(
                    graph_func
                )

        else:

            def get_trt_conversion_params(
                allow_build_at_runtime,
                max_workspace_size_bytes,
                precision_mode,
                minimum_segment_size):

                params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)

                def get_trt_precision():
                    if precision_mode == "FP32":
                        return trt.TrtPrecisionMode.FP32
                    elif precision_mode == "FP16":
                        return trt.TrtPrecisionMode.FP16
                    elif precision_mode == "INT8":
                        return trt.TrtPrecisionMode.INT8
                    else:
                        raise RuntimeError("Unknown precision received: `{}`. Expected: "
                                           "FP32, FP16 or INT8".format(precision))

                params = params._replace(
                    allow_build_at_runtime=allow_build_at_runtime,
                    max_workspace_size_bytes=max_workspace_size_bytes,
                    minimum_segment_size=minimum_segment_size,
                    precision_mode=get_trt_precision(),
                    use_calibration=precision_mode == "INT8"
                )

                print('\nTensorRT Conversion Params:')
                _print_dict(dict(params._asdict()))

                return params

            conversion_params = get_trt_conversion_params(
                allow_build_at_runtime=allow_build_at_runtime,
                max_workspace_size_bytes=max_workspace_size_bytes,
                precision_mode=precision_mode,
                minimum_segment_size=minimum_segment_size
            )

            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=input_saved_model_dir,
                conversion_params=conversion_params,
                input_saved_model_signature_key=input_signature_key,
                use_dynamic_shape=use_dynamic_shape
            )

            def _check_input_fn(func, name):
                if func is None:
                    raise ValueError("The function `{}` is None.".format(name))

                if not callable(func):
                    raise ValueError("The argument `{}` is not a function.".format(
                        name))

            if conversion_params.precision_mode == 'INT8':

                _check_input_fn(calibration_input_fn, "calibration_input_fn")

                with _timed_section('TF-TRT graph conversion and INT8 '
                                   'calibration ...'):
                    graph_func = converter.convert(
                        calibration_input_fn=tf.autograph.experimental.do_not_convert(
                            calibration_input_fn
                        )
                    )

            else:
                with _timed_section('TF-TRT graph conversion ...'):
                    graph_func = converter.convert()

            try:
                try:
                    line_length = max(160, os.get_terminal_size().columns)
                except OSError:
                    line_length = 160
                converter.summary(line_length=line_length, detailed=True)
            except AttributeError:
                pass

            if optimize_offline or use_dynamic_shape:

                _check_input_fn(
                    optimize_offline_input_fn,
                    "optimize_offline_input_fn"
                )

                with _timed_section('Building TensorRT engines...'):
                    converter.build(input_fn=tf.autograph.experimental.do_not_convert(
                        optimize_offline_input_fn
                    ))

            if output_saved_model_dir is not None:

                with _timed_section('Saving converted graph with TF-TRT ...'):
                    converter.save(output_saved_model_dir)
                    print("Converted graph saved to `{}`".format(
                        output_saved_model_dir))

        return graph_func

    def execute_benchmark(
        self,
        batch_size,
        display_every,
        get_benchmark_input_fn,
        num_iterations,
        num_warmup_iterations,
        skip_accuracy_testing,
        use_synthetic_data,
        use_xla,
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

        @_force_gpu_resync
        @tf.function(jit_compile=use_xla)
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
                    batch_y=batch_y,
                    **kwargs
                )

            if (i + 1) >= num_iterations:
                break

        if not skip_accuracy_testing:
            results['accuracy_metric'] = self.compute_accuracy_metric(
                batch_size=batch_size,
                steps_executed=steps_executed,
                **kwargs
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
