#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from tensorflow.python.compiler.tensorrt.trt_convert import \
    DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_constants import \
    DEFAULT_SERVING_SIGNATURE_DEF_KEY

from benchmark_logger import logging
from benchmark_utils import print_dict


class BaseCommandLineAPI(object):

    ALLOWED_PRECISION_MODES = ["FP32", "FP16", "INT8"]
    SAMPLES_IN_VALIDATION_SET = None

    def __init__(self):

        self._parser = argparse.ArgumentParser(
            description="TF-TRT Inference Benchmark"
        )

        # ======================= SavedModel Directories ===================== #

        self._parser.add_argument(
            "--input_saved_model_dir",
            type=str,
            default=None,
            help="Directory containing the input saved model."
        )

        self._parser.add_argument(
            "--output_saved_model_dir",
            type=str,
            default=None,
            help="Directory in which the converted model will be saved"
        )

        # ======================== Dataset Directories ======================= #

        self._parser.add_argument(
            "--calib_data_dir",
            type=str,
            help="Directory containing the dataset used for INT8 calibration."
        )

        self._parser.add_argument(
            "--data_dir",
            type=str,
            default=None,
            help="Directory containing the dataset used for model validation."
        )

        # ======================= Generic Runtime Flags ====================== #

        self._parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Number of images per batch."
        )

        self._parser.add_argument(
            "--display_every",
            type=int,
            default=50,
            help="Number of iterations executed between two consecutive "
            "displays of metrics"
        )

        self._parser.add_argument(
            "--gpu_mem_cap",
            type=int,
            default=0,
            help="Upper bound for GPU memory in MB. Default is 0 which means "
            "allow_growth will be used."
        )

        default_sign_key = DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self._parser.add_argument(
            "--input_signature_key",
            type=str,
            default=default_sign_key,
            help=f"SavedModel signature to use for inference, defaults to: "
            f"`{default_sign_key}`"
        )

        default_tag = tag_constants.SERVING
        self._parser.add_argument(
            "--model_tag",
            type=str,
            default=default_tag,
            help=f"SavedModel inference tag to use, defaults to: "
            f"{default_tag}"
        )

        self._parser.add_argument(
            "--output_tensors_name",
            type=str,
            default=None,
            help="Output tensors' name, defaults to all tensors available if "
            "not set. Will only work with `--use_tftrt`."
        )

        self._parser.add_argument(
            "--num_iterations",
            type=int,
            default=None,
            help="How many iterations(batches) to evaluate. If not supplied, "
            "the whole set will be evaluated."
        )

        self._parser.add_argument(
            "--num_warmup_iterations",
            type=int,
            default=200,
            help="Number of initial iterations skipped from timing."
        )

        self._parser.add_argument(
            "--trim_mean_percentage",
            type=float,
            default=0.1,
            required=False,
            help="Percentage used to trim step timing distribution from both "
            "tails (fastest and slowest steps). 0.1 (default value) means that "
            "10% of the fastest and slowest iteration will be removed for "
            "model throughput computation."
        )

        self._parser.add_argument(
            "--total_max_samples",
            type=int,
            default=None,
            required=True,
            help="Preallocated size of the result numpy arrays. Shall be at "
            "least as large as the number of samples in the dataset."
        )

        self._add_bool_argument(
            name="no_tf32",
            default=False,
            required=False,
            help="If set to True, the benchmark will force not using TF32."
        )

        self._add_bool_argument(
            name="use_xla",
            default=False,
            required=False,
            help="If set to True, the benchmark will use XLA JIT Compilation."
        )

        self._add_bool_argument(
            name="use_xla_auto_jit",
            default=False,
            required=False,
            help="If set to True, the benchmark will use XLA JIT Auto "
            "Clustering Compilation."
        )

        self._add_bool_argument(
            name="use_synthetic_data",
            default=False,
            required=False,
            help="If set to True, one unique batch of random batch of data is "
            "generated and used at every iteration."
        )

        self._parser.add_argument(
            "--precision",
            type=str,
            choices=self.ALLOWED_PRECISION_MODES,
            default="FP32",
            help="Precision mode to use. FP16 and INT8 modes only works if "
            "--use_tftrt is used."
        )

        # =========================== TF-TRT Flags ========================== #

        self._add_bool_argument(
            name="use_tftrt",
            default=False,
            required=False,
            help="If set to True, the inference graph will be converted using "
            "TF-TRT graph converter."
        )

        self._add_bool_argument(
            name="allow_build_at_runtime",
            default=False,
            required=False,
            help="Whether to build TensorRT engines during runtime."
        )

        self._add_bool_argument(
            name="detailed_conversion_summary",
            default=False,
            required=False,
            help="Whether to use TF-TRT detailled conversion summary."
        )

        self._parser.add_argument(
            "--max_workspace_size",
            type=int,
            # Ensuring default of minimum 8GB
            default=max(1 << 33, DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES),
            help="The maximum GPU temporary memory which the TRT engine can "
            "use at execution time."
        )

        self._parser.add_argument(
            "--minimum_segment_size",
            type=int,
            default=5,
            help="Minimum number of TensorFlow ops in a TRT engine."
        )

        self._parser.add_argument(
            "--num_build_batches",
            type=int,
            default=1,
            help="How many iterations(batches) to use to build the TF-TRT "
            "engines. If not supplied, only one batch will be used. This "
            "parameter has only an effect if `--optimize_offline=True`"
        )

        self._parser.add_argument(
            "--num_calib_batches",
            type=int,
            default=10,
            help="Number of batches used for INT8 calibration (only useful if "
            "--use_tftrt is set with --precision=INT8"
        )

        self._add_bool_argument(
            name="optimize_offline",
            default=True,
            required=False,
            help="If set to True, TensorRT engines are built before runtime."
        )

        self._add_bool_argument(
            name="use_dynamic_shape",
            default=False,
            required=False,
            help="Whether to use implicit batch mode or dynamic shape mode."
        )

        # =========================== Metric Flags ========================== #

        self._parser.add_argument(
            "--experiment_name",
            type=str,
            default=None,
            help="Name of the experiment being run, only used for archiving "
            "objectives: exports in JSON or CSV."
        )

        self._parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            default=None,
            help="Name of the model being benchmarked."
        )

        self._parser.add_argument(
            "--model_source",
            type=str,
            required=True,
            default=None,
            help="Source of the model where it was originally published."
        )

        self._parser.add_argument(
            "--export_metrics_json_path",
            type=str,
            default=None,
            help="If set, the script will export runtime metrics and arguments "
            "to the set location in JSON format for further processing."
        )

        self._parser.add_argument(
            "--export_metrics_csv_path",
            type=str,
            default=None,
            help="If set, the script will export runtime metrics and arguments "
            "to the set location in CSV format for further processing."
        )

        self._parser.add_argument(
            "--upload_metrics_endpoint",
            type=str,
            default=None,
            help="If set, the benchmark will upload the metrics in JSON format "
            "to the set endpoint using a PUT requests."
        )

        # =========================== TF Profiling =========================== #

        self._parser.add_argument(
            "--tftrt_build_profile_export_path",
            type=str,
            default=None,
            help="If set, the script will export tf.profile files for further "
            "performance analysis."
        )

        self._parser.add_argument(
            "--tftrt_convert_profile_export_path",
            type=str,
            default=None,
            help="If set, the script will export tf.profile files for further "
            "performance analysis."
        )

        self._parser.add_argument(
            "--inference_loop_profile_export_path",
            type=str,
            default=None,
            help="If set, the script will export tf.profile files for further "
            "performance analysis."
        )

        self._add_bool_argument(
            name="tf_profile_verbose",
            default=False,
            required=False,
            help="If set to True, will add extra information to the TF Profile."
        )

        # ============================ Debug Flags =========================== #

        self._add_bool_argument(
            name="debug",
            default=False,
            required=False,
            help="If set to True, will print additional information."
        )

        self._add_bool_argument(
            name="debug_performance",
            default=False,
            required=False,
            help="If set to True, will print additional information."
        )

        self._add_bool_argument(
            name="debug_data_aggregation",
            default=False,
            required=False,
            help="If set to True, will print additional information related to "
            "data aggregation."
        )

    def _add_bool_argument(
        self, name=None, default=False, required=False, help=None
    ):
        if not isinstance(default, bool):
            raise ValueError()

        feature_parser = self._parser.add_mutually_exclusive_group(\
            required=required
        )

        feature_parser.add_argument(
            "--" + name,
            dest=name,
            action="store_true",
            help=help,
            default=default
        )

        feature_parser.add_argument(
            "--no" + name, dest=name, action="store_false"
        )

        feature_parser.set_defaults(name=default)

    def _validate_args(self, args):

        if args.data_dir is None:
            raise ValueError("--data_dir is required")

        elif not os.path.isdir(args.data_dir):
            raise RuntimeError(
                f"The path --data_dir=`{args.data_dir}` doesn't exist or is "
                "not a directory"
            )

        if args.use_synthetic_data and args.num_iterations is None:
            raise ValueError(
                "The use of --use_synthetic_data requires to "
                "specify the number of iterations using "
                "--num_iterations=X."
            )

        if (args.num_iterations is not None and
                args.num_iterations <= args.num_warmup_iterations):
            raise ValueError(
                "--num_iterations must be larger than --num_warmup_iterations "
                f"({args.num_iterations} <= {args.num_warmup_iterations})"
            )

        if (args.tf_profile_verbose and args.tf_profile_export_path is None):
            raise ValueError(
                "`--tf_profile_verbose` can only be set if "
                "`--tf_profile_export_path=/path/to/export` is defined."
            )

        if not args.use_tftrt:
            if args.use_dynamic_shape:
                raise ValueError(
                    "TensorRT must be enabled for Dynamic Shape support to be "
                    "enabled (--use_tftrt)."
                )

            if args.precision == "INT8":
                raise ValueError(
                    "TensorRT must be enabled for INT8 precision mode: "
                    "`--use_tftrt`."
                )

        else:
            if args.use_xla:
                raise ValueError("--use_xla flag is not supported with TF-TRT.")

            if args.precision == "INT8":

                if not args.calib_data_dir:
                    raise ValueError(
                        "--calib_data_dir is required for INT8 precision mode"
                    )

                elif not os.path.isdir(args.calib_data_dir):
                    raise RuntimeError(
                        f"The path --calib_data_dir=`{args.calib_data_dir}` "
                        "doesn't exist or is not a directory"
                    )

        # yapf: disable
        if (
            args.upload_metrics_endpoint is not None and
            args.experiment_name is None
        ):
            raise NotImplementedError(
                "--experiment_name must be specified if "
                "--upload_metrics_endpoint is set."
            )
        # yapf: enable

    def _post_process_args(self, args):
        if args.use_synthetic_data:
            # This variable is not used in synthetic data mode.
            # Let's fix it to 1 to save memory.
            args.total_max_samples = 1

        if args.debug or args.debug_data_aggregation or args.debug_performance:
            logging.set_verbosity(logging.DEBUG)

        if (args.inference_loop_profile_export_path or
                args.tftrt_build_profile_export_path or
                args.tftrt_convert_profile_export_path):
            """Warm-up the profiler session.
            The profiler session will set up profiling context, including loading CUPTI
            library for GPU profiling. This is used for improving the accuracy of
            the profiling results.
            """
            from tensorflow.python.profiler.profiler_v2 import warmup
            logging.info("[PROFILER] Warming Up ...")
            warmup()

        return args

    def parse_args(self):
        args = self._parser.parse_args()

        args = self._post_process_args(args)
        self._validate_args(args)

        logging.info("Benchmark arguments:")
        print_dict(vars(args))
        print()

        return args
