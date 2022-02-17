#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import os

import abc
import logging
import sys
import time

from distutils.util import strtobool

from benchmark_utils import DataAggregator
from benchmark_utils import force_gpu_resync
from benchmark_utils import print_dict
from benchmark_utils import timed_section

import numpy as np
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

__all__ = ["BaseBenchmarkRunner"]


class BaseBenchmarkRunner(object, metaclass=abc.ABCMeta):

    ############################################################################
    # Methods expected to be overwritten by the subclasses
    ############################################################################

    @abc.abstractmethod
    def get_dataset_batches(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def preprocess_model_inputs(self, data_batch):
        raise NotImplementedError()

    @abc.abstractmethod
    def postprocess_model_outputs(self, predictions, expected):
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        raise NotImplementedError()

    ############################################################################
    # Common methods for all the benchmarks
    ############################################################################

    def __init__(self, args):
        self._args = args

        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.disable(logging.WARNING)

        # TensorFlow can execute operations synchronously or asynchronously.
        # If asynchronous execution is enabled, operations may return
        # "non-ready" handles.
        tf.config.experimental.set_synchronous_execution(True)

        self._config_gpu_memory(self._args.gpu_mem_cap)

    def _config_gpu_memory(self, gpu_mem_cap):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if not gpus:
            raise RuntimeError("No GPUs has been found.")

        self._debug_print('Found the following GPUs:')
        for gpu in gpus:
            self._debug_print(f"\t- {gpu}")

        for gpu in gpus:
            try:
                if not gpu_mem_cap:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu, [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=gpu_mem_cap
                            )
                        ]
                    )
            except RuntimeError as e:
                print('Can not set GPU memory config', e)

    def _debug_print(self, msg):
        if self._args.debug:
            print(f"[DEBUG] {msg}")

    def _get_graph_func(self):
        """Retreives a frozen SavedModel and applies TF-TRT
        use_tftrt: bool, if true use TensorRT
        precision: str, floating point precision (FP32, FP16, or INT8)
        returns: TF function that is ready to run for inference
        """

        if not self._args.use_tftrt:

            with timed_section('Loading TensorFlow native model...'):

                saved_model_loaded = tf.saved_model.load(
                    export_dir=self._args.input_saved_model_dir,
                    tags=self._args.model_tag.split(",")
                )

                graph_func = saved_model_loaded.signatures[
                    self._args.input_signature_key]
                # graph_func = convert_to_constants.convert_variables_to_constants_v2(
                #     graph_func
                # )

        else:

            def get_trt_precision(precision):
                if precision == "FP32":
                    return trt.TrtPrecisionMode.FP32
                elif precision == "FP16":
                    return trt.TrtPrecisionMode.FP16
                elif precision == "INT8":
                    return trt.TrtPrecisionMode.INT8
                else:
                    raise RuntimeError(
                        f"Unknown precision received: `{precision}`. "
                        f"Expected: FP32, FP16 or INT8"
                    )

            tftrt_precision = get_trt_precision(self._args.precision)

            trt_converter_params = dict(
                allow_build_at_runtime=self._args.allow_build_at_runtime,
                enable_sparse_compute=True,
                input_saved_model_dir=self._args.input_saved_model_dir,
                input_saved_model_signature_key=self._args.input_signature_key,
                input_saved_model_tags=self._args.model_tag.split(","),
                max_workspace_size_bytes=self._args.max_workspace_size,
                maximum_cached_engines=1,
                minimum_segment_size=self._args.minimum_segment_size,
                precision_mode=tftrt_precision,
                use_calibration=(tftrt_precision == trt.TrtPrecisionMode.INT8),
                use_dynamic_shape=self._args.use_dynamic_shape,
            )

            print("\n[*] TF-TRT Converter Parameters:")
            print_dict(trt_converter_params)

            converter = trt.TrtGraphConverterV2(**trt_converter_params)

            def engine_build_input_fn(num_batches, model_phase):
                dataset, _ = self.get_dataset_batches()

                for idx, data_batch in enumerate(dataset):
                    print(
                        f"* [{model_phase}] "
                        f"- step {(idx+1):04d}/{num_batches:04d}"
                    )
                    x, _ = self.preprocess_model_inputs(data_batch)  # x, y

                    if not isinstance(x, (tuple, list, dict)):
                        x = [x]

                    yield x

                    if (idx + 1) >= num_batches:
                        break

            if tftrt_precision == trt.TrtPrecisionMode.INT8:

                calibration_input_fn = lambda: engine_build_input_fn(
                    num_batches=self._args.num_calib_batches,
                    model_phase="Calibration"
                )

                with timed_section(
                        "TF-TRT graph conversion and INT8 calibration ..."):
                    graph_func = converter.convert(
                        calibration_input_fn=(
                            tf.autograph.experimental.
                            do_not_convert(calibration_input_fn)
                        )
                    )

            else:
                with timed_section("TF-TRT graph conversion ..."):
                    graph_func = converter.convert()

            try:
                try:
                    line_length = max(160, os.get_terminal_size().columns)
                except OSError:
                    line_length = 160
                converter.summary(line_length=line_length, detailed=True)
            except AttributeError:
                pass

            if strtobool(os.environ.get("TF_TRT_BENCHMARK_QUIT_AFTER_SUMMARY",
                                        "0")):
                sys.exit(0)

            if self._args.optimize_offline or self._args.use_dynamic_shape:

                offline_opt_input_fn = lambda: engine_build_input_fn(
                    num_batches=1, model_phase="Building"
                )

                with timed_section('Building TensorRT engines...'):
                    converter.build(
                        input_fn=tf.autograph.experimental.
                        do_not_convert(offline_opt_input_fn)
                    )
                pass

            if self._args.output_saved_model_dir is not None:

                with timed_section('Saving converted graph with TF-TRT ...'):
                    converter.save(self._args.output_saved_model_dir)
                    print(
                        f"Converted graph saved to "
                        f"`{self._args.output_saved_model_dir}`"
                    )

        savedmodel_outputs = print_dict(
            graph_func.structured_outputs,
            redirect_to_str=True
        )

        chosen_outputs = "\n  - ".join(
            sorted(self._args.output_tensors_name.split(","))
        )

        self._debug_print(f"Available Output Tensors:\n{savedmodel_outputs}")
        print()  # visual spacing
        self._debug_print(f"Chosen Output Tensor:\n  - {chosen_outputs}")

        return graph_func

    def execute_benchmark(self):
        """Run the given graph_func on the data files provided.
        It consumes TFRecords with labels and reports accuracy.
        """

        print("\nStart loading model ...")

        graph_func = self._get_graph_func()

        dataset, bypass_data_to_eval = self.get_dataset_batches()

        print("\nStart inference ...")

        @force_gpu_resync
        @tf.function(jit_compile=self._args.use_xla)
        def infer_batch(x):
            if isinstance(x, (tuple, list)):
                model_out = graph_func(*x)
            elif isinstance(x, dict):
                model_out = graph_func(**x)
            else:
                model_out = graph_func(x)

            if self._args.output_tensors_name is not None:
                output_tensors_name = self._args.output_tensors_name.split(",")
                if len(output_tensors_name) == 1:
                    return model_out[self._args.output_tensors_name]
                else:
                    return {key: model_out[key] for key in output_tensors_name}

            return model_out

        data_aggregator = DataAggregator(
            self.postprocess_model_outputs, args=self._args
        )

        iter_times = []

        def log_step(step_idx, display_every, iter_time):
            if step_idx % display_every == 0:
                print("  step %04d, iter_time(ms)=%.3f" % (step_idx, iter_time))

        data_start_t = time.time()
        for step_idx, data_batch in enumerate(dataset):
            x, y = self.preprocess_model_inputs(data_batch)

            if self._args.debug:
                print("Step:", step_idx + 1)
                print("Data Loading Time:", time.time() - data_start_t)

            start_time = time.time()
            y_pred = infer_batch(x)
            iter_times.append(time.time() - start_time)

            if not self._args.debug:
                log_step(
                    step_idx + 1, self._args.display_every,
                    np.mean(iter_times[-self._args.display_every:]) * 1000
                )
            else:
                print("GPU Iteration Time:", iter_times[-1])

            data_aggregator.aggregate_data(y_pred, y)

            if (self._args.num_iterations is not None and
                    step_idx + 1 >= self._args.num_iterations):
                break

            if self._args.debug:
                print("===============")
                data_start_t = time.time()

        if step_idx % self._args.display_every != 0:  # avoids double printing
            log_step(
                step_idx + 1,
                display_every=1,  # force print
                iter_time=np.mean(iter_times[-self._args.display_every:]) * 1000
            )

        print("\nEnd of inference. Computing metrics ...\n")

        metric, metric_units = self.evaluate_model(
            data_aggregator.predicted_dict, data_aggregator.expected_dict,
            bypass_data_to_eval
        )
        print(f"- {metric_units:35s}: {metric:.2f}")

        metrics = dict()

        metrics["Total Samples Processed"] = (
            data_aggregator.total_samples_processed
        )

        # Skipping last batch. Might have different batch_size
        run_times = np.array(iter_times)[self._args.num_warmup_iterations:-1]

        metrics['Total GPU Time (s)'] = int(np.ceil(np.sum(iter_times)))
        metrics['Throughput (samples/sec)'] = np.mean(
            self._args.batch_size / run_times
        )
        metrics['99th_percentile (ms)'] = np.percentile(
            run_times, q=99, interpolation='lower'
        ) * 1000
        metrics['GPU Latency Mean (ms)'] = np.mean(run_times) * 1000
        metrics['GPU Latency Median (ms)'] = np.median(run_times) * 1000
        metrics['GPU Latency Min (ms)'] = np.min(run_times) * 1000
        metrics['GPU Latency Max (ms)'] = np.max(run_times) * 1000

        for key, val in sorted(metrics.items()):
            if isinstance(val, int):
                print(f"- {key:35s}: {val}")
            else:
                print(f"- {key:35s}: {val:.2f}")
