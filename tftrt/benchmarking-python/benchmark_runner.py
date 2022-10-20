#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import abc
import contextlib
import copy
import csv
import functools
import json
import os
import requests
import sys
import time

from distutils.util import strtobool

import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from absl import logging as absl_logging

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from benchmark_autotuner import auto_tf_func_tuner

from benchmark_info import __version__
from benchmark_info import get_commit_id

from benchmark_logger import logging

from benchmark_profiling import ProfilingCTX
from benchmark_profiling import time_and_trace_ctx

from benchmark_utils import DataAggregator
from benchmark_utils import generate_json_metrics
from benchmark_utils import print_dict
from benchmark_utils import timed_section

from dataloading_utils import SyntheticDataset
from dataloading_utils import ensure_dataset_on_gpu
from dataloading_utils import get_dequeue_batch_fn
from dataloading_utils import get_force_data_on_gpu_fn

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

        # Temporary fix to re-enable NHWC layout.
        # os.environ["TF_ENABLE_LAYOUT_NHWC"] = "1"

        if args.use_xla_auto_jit:
            logging.info("[Benchmark] - Activating XLA JIT Auto Clustering")
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
            os.environ["TF_XLA_FLAGS"] += " --tf_xla_cpu_global_jit"

        if args.no_tf32:
            logging.info("[Benchmark] - Deactivating the use of TF32 format")
            os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

        # Hide unnecessary TensorFlow DEBUG Python Logs
        tf_logger = tf_logging.get_logger()
        tf_logger.setLevel(tf_logging.INFO)
        tf_logger.propagate = False

        # disable TF warnings
        tf_logging.get_logger().warning = lambda *a, **kw: None
        tf_logging.get_logger().warn = lambda *a, **kw: None
        old_log = tf_logging.get_logger().log
        tf_logging.get_logger().log = lambda level, msg, *a, **kw: (
            old_log(level, msg, *a, **kw) if level != tf_logging.WARN else None
        )

        # Set ABSL verbosity to Err level
        absl_logging.set_verbosity(absl_logging.ERROR)

        # TensorFlow can execute operations synchronously or asynchronously.
        # If asynchronous execution is enabled, operations may return
        # "non-ready" handles.
        tf.config.experimental.set_synchronous_execution(True)

        self._config_gpu_memory(self._args.gpu_mem_cap)

    def _config_gpu_memory(self, gpu_mem_cap):
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except AttributeError:
            gpus = tf.config.experimental.list_physical_devices("GPU")

        if not gpus:
            raise RuntimeError("No GPUs has been found.")

        print()  # visual spacing
        logging.debug("Found the following GPUs:")
        for gpu in gpus:
            logging.debug(f"\t- {gpu}")

        for gpu in gpus:
            try:
                if not gpu_mem_cap:
                    try:
                        tf.config.set_memory_growth(gpu, True)
                    except AttributeError:
                        tf.config.experimental.set_memory_growth(gpu, True)

                else:
                    try:
                        set_virtual_device_configuration = tf.config.set_virtual_device_configuration
                        device_config = tf.config.LogicalDeviceConfiguration(
                            memory_limit=gpu_mem_cap
                        )
                    except AttributeError:
                        set_virtual_device_configuration = tf.config.experimental.set_virtual_device_configuration
                        device_config = tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_mem_cap
                        )

                    set_virtual_device_configuration(gpu, [device_config])
            except RuntimeError as e:
                logging.error(f"Can not set GPU memory config: {e}")

        print()  # visual spacing

    def _export_runtime_metrics_to_json(self, metric_dict):

        try:

            file_path = self._args.export_metrics_json_path
            if file_path is None:
                return

            json_string = generate_json_metrics(
                metrics=metric_dict,
                args=vars(self._args),
            )

            with open(file_path, "w") as json_f:
                print(json_string, file=json_f)

        except Exception as e:
            logging.error(f"An exception occured during export to JSON: {e}")

    def _export_runtime_metrics_to_csv(self, metric_dict):

        try:

            file_path = self._args.export_metrics_csv_path
            if file_path is None:
                return

            data = {f"metric_{k}": v for k, v in metric_dict.items()}

            # yapf: disable
            args_to_save = [
                "batch_size",
                "input_saved_model_dir",
                "minimum_segment_size",
                "no_tf32",
                "precision",
                "use_dynamic_shape",
                "use_synthetic_data",
                "use_tftrt",
                "use_xla",
                "use_xla_auto_jit"
            ]
            # yapf: enable

            runtime_arguments = vars(self._args)
            for key in args_to_save:
                data[f"arg_{key}"] = str(runtime_arguments[key]).split("/")[-1]

            fieldnames = sorted(data.keys())

            if not os.path.isfile(file_path):
                with open(file_path, "w") as outcsv:
                    writer = csv.DictWriter(
                        outcsv, fieldnames=fieldnames, delimiter=","
                    )
                    writer.writeheader()

            with open(file_path, "a") as outcsv:
                writer = csv.DictWriter(
                    outcsv, fieldnames=fieldnames, delimiter=","
                )
                writer.writerow(data)

        except Exception as e:
            logging.error(f"An exception occured during export to CSV: {e}")

    def _upload_metrics_to_endpoint(self, metric_dict):

        try:

            if self._args.upload_metrics_endpoint is None:
                return

            json_string = generate_json_metrics(
                metrics=metric_dict,
                args=vars(self._args),
            )

            headers = {"Content-Type": "application/json"}

            response = requests.put(
                self._args.upload_metrics_endpoint,
                data=json_string,
                headers=headers
            )
            response.raise_for_status()

            logging.info(
                "Metrics Uploaded to endpoint: "
                f"`{self._args.upload_metrics_endpoint}` with experiment name: "
                f"`{self._args.experiment_name}`."
            )

        except Exception as e:
            logging.error(f"An exception occured during export to JSON: {e}")

    def _get_graph_func(self):
        """Retreives a frozen SavedModel and applies TF-TRT
        use_tftrt: bool, if true use TensorRT
        precision: str, floating point precision (FP32, FP16, or INT8)
        returns: TF function that is ready to run for inference
        """

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

        if not self._args.use_tftrt:

            with time_and_trace_ctx("Loading TensorFlow native model"):
                graph_func = load_model_from_disk(
                    path=self._args.input_saved_model_dir,
                    tags=self._args.model_tag.split(","),
                    signature_key=self._args.input_signature_key,
                    precision=self._args.precision
                )

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

            logging.info("[*] TF-TRT Converter Parameters:")
            print_dict(trt_converter_params)

            with timed_section("TF-TRT Conversion and Build"):

                with timed_section("TrtGraphConverterV2 creation"):

                    try:
                        converter = trt.TrtGraphConverterV2(
                            **trt_converter_params
                        )
                    except TypeError:
                        del trt_converter_params["enable_sparse_compute"]
                        converter = trt.TrtGraphConverterV2(
                            **trt_converter_params
                        )

                def engine_build_input_fn(num_batches, model_phase):
                    dataset, _ = self.get_dataset_batches()

                    for idx, data_batch in enumerate(dataset):
                        logging.info(
                            f"* [{model_phase}] "
                            f"- step {(idx+1):04d}/{num_batches:04d}"
                        )
                        x, _ = self.preprocess_model_inputs(data_batch)  # x, y

                        if not isinstance(x, (tuple, list, dict)):
                            x = [x]

                        yield x

                        if (idx + 1) >= num_batches:
                            break

                with ProfilingCTX(self._args.tftrt_convert_profile_export_path,
                                  verbose=self._args.tf_profile_verbose,
                                  delay_ms=0):

                    with time_and_trace_ctx("TF-TRT Model Conversion",
                                            step_num=0, _r=0):
                        if tftrt_precision == trt.TrtPrecisionMode.INT8:

                            calibration_input_fn = lambda: engine_build_input_fn(
                                num_batches=self._args.num_calib_batches,
                                model_phase="Calibration"
                            )

                            graph_func = converter.convert(
                                calibration_input_fn=(
                                    tf.autograph.experimental.
                                    do_not_convert(calibration_input_fn)
                                )
                            )

                        else:
                            graph_func = converter.convert()

                    with time_and_trace_ctx("TF-TRT Model Summary", step_num=0,
                                            _r=0):
                        try:
                            try:
                                line_length = max(
                                    160,
                                    os.get_terminal_size().columns
                                )
                            except OSError:
                                line_length = 160

                            converter.summary(
                                line_length=line_length,
                                detailed=self._args.detailed_conversion_summary
                            )
                        except AttributeError:
                            pass

                    if strtobool(os.environ.get("TF_TRT_BENCHMARK_EARLY_QUIT",
                                                "0")):
                        with time_and_trace_ctx("TF-TRT Model Saving",
                                                step_num=0, _r=0):
                            # Save the result if needed
                            if self._args.output_saved_model_dir is not None:
                                converter.save(
                                    self._args.output_saved_model_dir
                                )

                            logging.info(
                                f"Converted graph saved to "
                                f"`{self._args.output_saved_model_dir}`"
                            )
                            try:
                                # Stop profiling and export if started
                                profiler.stop()
                            except tf.errors.UnavailableError:
                                pass
                            sys.exit(0)

                if self._args.optimize_offline:

                    with ProfilingCTX(
                            self._args.tftrt_build_profile_export_path,
                            verbose=self._args.tf_profile_verbose, delay_ms=0):

                        if tftrt_precision == trt.TrtPrecisionMode.INT8:
                            message = "TF-TRT Engines Building and Calibrating"
                        else:
                            message = "TF-TRT Engines Building"

                        with time_and_trace_ctx(message, step_num=0, _r=0):

                            offline_opt_input_fn = lambda: engine_build_input_fn(
                                num_batches=self._args.num_build_batches,
                                model_phase="Building"
                            )

                            converter.build(
                                input_fn=tf.autograph.experimental.
                                do_not_convert(offline_opt_input_fn)
                            )

                if self._args.output_saved_model_dir is not None:

                    with time_and_trace_ctx("TF-TRT Model Saving", step_num=0,
                                            _r=0):
                        converter.save(self._args.output_saved_model_dir)
                        logging.info(
                            f"Converted graph saved to "
                            f"`{self._args.output_saved_model_dir}`"
                        )

                    with time_and_trace_ctx("TF-TRT Model Reloading",
                                            step_num=0, _r=0):
                        # Engine cache is cleared while saving, we have to reload.
                        # Failing to do so, would force TF-TRT to rebuild
                        del converter
                        del graph_func
                        graph_func = load_model_from_disk(
                            self._args.output_saved_model_dir,
                            tags=self._args.model_tag.split(","),
                            signature_key=self._args.input_signature_key
                        )

        if isinstance(graph_func.structured_outputs, (tuple, list)):
            savedmodel_outputs = "\n\t- ".join([
                str(t) for t in graph_func.structured_outputs
            ])
            if savedmodel_outputs:
                savedmodel_outputs = f"\t- {savedmodel_outputs}"

            savedmodel_outputs = f"\t- {savedmodel_outputs}"
        else:
            savedmodel_outputs = print_dict(
                graph_func.structured_outputs, redirect_to_str=True
            )

        logging.debug(f"Available Output Tensors:")
        for _str in savedmodel_outputs.split("\n"):
            logging.debug(_str)
        print()  # visual spacing

        chosen_outputs = "\n\t- ".join(
            sorted(self._args.output_tensors_name.split(","))
        )
        if chosen_outputs:
            chosen_outputs = f"\t- {chosen_outputs}"

        logging.debug(f"Chosen Output Tensor:")
        for _str in chosen_outputs.split("\n"):
            logging.debug(_str)
        print()  # visual spacing

        return graph_func

    def execute_benchmark(self):
        """Run the given graph_func on the data files provided.
        It consumes TFRecords with labels and reports accuracy.
        """

        with timed_section("Model Loading"):
            graph_func = self._get_graph_func()

        with timed_section("Model Inference"):
            dataset, bypass_data_to_eval = self.get_dataset_batches()

            if self._args.use_synthetic_data:
                try:
                    dataset = SyntheticDataset(dataset, device="/gpu:0")
                    logging.debug(
                        "Model dataset has been replaced by a synthetic data "
                        "loader to minimize data loading jitter."
                    )

                except Exception as e:
                    logging.error(
                        f"Impossible to transform the dataset into a "
                        f"synthetic dataset. Performance numbers will be "
                        f"impacted.\nError: {str(e)}."
                    )
            else:
                dataset = ensure_dataset_on_gpu(dataset, device="GPU:0")

            @auto_tf_func_tuner(
                use_xla=self._args.use_xla,
                use_synthetic_data=self._args.use_synthetic_data
            )
            def infer_batch(x):
                if isinstance(x, (tuple, list)):
                    model_out = graph_func(*x)
                elif isinstance(x, dict):
                    model_out = graph_func(**x)
                else:
                    model_out = graph_func(x)

                if self._args.output_tensors_name is not None:
                    output_ts_name = self._args.output_tensors_name.split(",")
                    if len(output_ts_name) == 1:
                        return model_out[self._args.output_tensors_name]
                    else:
                        return {key: model_out[key] for key in output_ts_name}

                return model_out

            if not self._args.use_synthetic_data:
                data_aggregator = DataAggregator(
                    self.postprocess_model_outputs, args=self._args
                )

            iter_times = []
            memcopy_times = []
            dequeue_times = []

            def log_step(
                step_idx, display_every, iter_time, memcpyHtoD_time,
                dequeue_time
            ):
                if step_idx % display_every == 0:
                    logging.info(
                        f"step {step_idx:04d}, "
                        f"iter_time(ms)={iter_time:08.3f}, "
                        f"memcpyHtoD_time(ms)={memcpyHtoD_time:08.3f}, "
                        f"dequeue_time(ms)={dequeue_time:08.3f}"
                    )

            step_idx = 0
            ds_iter = iter(dataset)

            dequeue_batch_fn = get_dequeue_batch_fn(
                ds_iter,
                use_xla=self._args.use_xla,
                use_synthetic_data=self._args.use_synthetic_data
            )

            force_data_on_gpu_fn = get_force_data_on_gpu_fn(
                device="/gpu:0",
                use_xla=self._args.use_xla,
                use_synthetic_data=self._args.use_synthetic_data
            )

            profiler = ProfilingCTX(
                export_dir=self._args.inference_loop_profile_export_path,
                verbose=self._args.tf_profile_verbose,
                delay_ms=0
            )

            while True:

                step_idx += 1

                if step_idx == self._args.num_warmup_iterations - 5:
                    profiler.start()

                if (self._args.num_iterations is not None and
                        step_idx > self._args.num_iterations):
                    break

                with tf.profiler.experimental.Trace("Step ", step_num=step_idx,
                                                    _r=1):

                    with tf.profiler.experimental.Trace("Input Dequeueing"):
                        try:
                            start_time = time.perf_counter()
                            data_batch = dequeue_batch_fn()
                            dequeue_times.append(
                                time.perf_counter() - start_time
                            )
                        except (StopIteration, OutOfRangeError):
                            logging.info("[Exiting] Reached end of dataset ...")
                            break

                    with tf.profiler.experimental.Trace("Inputs Preprocessing"):
                        x, y = self.preprocess_model_inputs(data_batch)

                    with tf.profiler.experimental.Trace("Inputs MemcpyHtoD"):
                        start_time = time.perf_counter()
                        x = force_data_on_gpu_fn(x)
                        memcopy_times.append(time.perf_counter() - start_time)

                    with tf.profiler.experimental.Trace("GPU Inference"):
                        start_time = time.perf_counter()
                        y_pred = infer_batch(x)
                        iter_times.append(time.perf_counter() - start_time)

                if not self._args.debug_performance:
                    log_step(
                        step_idx,
                        display_every=self._args.display_every,
                        iter_time=np.mean(
                            iter_times[-self._args.display_every:]
                        ) * 1000,
                        memcpyHtoD_time=np.mean(
                            memcopy_times[-self._args.display_every:]
                        ) * 1000,
                        dequeue_time=np.mean(
                            dequeue_times[-self._args.display_every:]
                        ) * 1000
                    )
                else:
                    logging.info(
                        f"{'GPU Iteration Time':18s}: {iter_times[-1]:08.4f}s"
                    )
                    logging.info(
                        f"{'Data MemCopyHtoD Time':18s}: {memcpyHtoD_time[-1]:08.4f}s"
                    )
                    logging.info(
                        f"{'Data Dequeue Time':18s}: {dequeue_times[-1]:08.4f}s"
                    )

                if not self._args.use_synthetic_data:
                    data_aggregator.aggregate_data(y_pred, y)

            # yapf: disable
            if (
                not self._args.debug_performance and
                # avoids double printing
                step_idx % self._args.display_every != 0
            ):
                log_step(
                    step_idx,
                    display_every=1,  # force print
                    iter_time=np.mean(iter_times[-self._args.display_every:]) *
                    1000,
                    memcpyHtoD_time=np.mean(
                        memcopy_times[-self._args.display_every:]
                    ) * 1000,
                    dequeue_time=np.mean(
                        dequeue_times[-self._args.display_every:]
                    ) * 1000
                )
            # yapf: enable

            if step_idx >= 100:
                profiler.stop()

        with time_and_trace_ctx("Metric Computation"):

            metrics = dict()

            if not self._args.use_synthetic_data:
                metric, metric_units = self.evaluate_model(
                    data_aggregator.predicted_dict,
                    data_aggregator.expected_dict, bypass_data_to_eval
                )
                metrics["Metric"] = {metric_units: metric}

                metrics["Total Samples Processed"] = (
                    data_aggregator.total_samples_processed
                )

            # Skipping last batch. Might have different batch_size
            iter_times = np.array(iter_times)
            iter_times = iter_times[self._args.num_warmup_iterations:-1]

            memcopy_times = np.array(memcopy_times)
            memcopy_times = memcopy_times[self._args.num_warmup_iterations:-1]

            dequeue_times = np.array(dequeue_times)
            dequeue_times = dequeue_times[self._args.num_warmup_iterations:-1]

            metrics["Total GPU Time (s)"] = int(np.ceil(np.sum(iter_times)))

            metrics["__commit__"] = get_commit_id()
            metrics["__version__"] = __version__

            metrics["Throughput (samples/sec)"] = (
                self._args.batch_size /
                sp.stats.trim_mean(iter_times, self._args.trim_mean_percentage)
            )

            def timing_metrics(time_arr, log_prefix):
                data = dict()
                data[
                    f"{log_prefix} Trim Mean [{self._args.trim_mean_percentage * 100}%] (ms)"
                ] = (
                    sp.stats.
                    trim_mean(time_arr, self._args.trim_mean_percentage) * 1000
                )
                data[f"{log_prefix} 99th_percentile (ms)"] = np.percentile(
                    time_arr, q=99, interpolation="lower"
                ) * 1000
                data[f"{log_prefix} Mean (ms)"] = np.mean(time_arr) * 1000
                data[f"{log_prefix} Median (ms)"] = np.median(time_arr) * 1000
                data[f"{log_prefix} Min (ms)"] = np.min(time_arr) * 1000
                data[f"{log_prefix} Max (ms)"] = np.max(time_arr) * 1000
                return data

            metrics.update(timing_metrics(iter_times, "GPU Latency"))
            metrics.update(
                timing_metrics(dequeue_times, "Data Batch Dequeue Time")
            )
            metrics.update(
                timing_metrics(memcopy_times, "Data MemCopyHtoD Time")
            )

            self._export_runtime_metrics_to_json(metrics)
            self._export_runtime_metrics_to_csv(metrics)
            self._upload_metrics_to_endpoint(metrics)

            def log_value(key, val):
                if isinstance(val, (int, str)):
                    logging.info(f"- {key:50s}: {val}")
                else:
                    logging.info(f"- {key:50s}: {val:.2f}")

            for key, val in sorted(metrics.items()):
                if isinstance(val, dict):
                    log_value(*list(val.items())[0])
                else:
                    log_value(key, val)

        print()  # visual spacing
