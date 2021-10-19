#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt.trt_convert import \
    DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_constants import \
    DEFAULT_SERVING_SIGNATURE_DEF_KEY

from benchmark_runner import _print_dict


class BaseCommandLineAPI(object):

    ALLOWED_TFTRT_PRECISION_MODES = ['FP32', 'FP16', 'INT8']
    SAMPLES_IN_VALIDATION_SET = None

    def __init__(self):
        self._parser = argparse.ArgumentParser(description='tftrt_benchmark')

        # ======================= SavedModel Directories ===================== #

        self._parser.add_argument('--input_saved_model_dir', type=str,
                                  default=None,
                                  help='Directory containing the input saved '
                                       'model.')

        self._parser.add_argument('--output_saved_model_dir', type=str,
                                  default=None,
                                  help='Directory in which the converted model '
                                       'will be saved')

        # ======================== Dataset Directories ======================= #

        self._parser.add_argument('--calib_data_dir', type=str,
                                  help='Directory containing the dataset used '
                                       'for INT8 calibration.')

        self._parser.add_argument('--data_dir', type=str, default=None,
                                  help='Directory containing the dataset used '
                                       'for model validation.')

        # ======================= Generic Runtime Flags ====================== #

        self._parser.add_argument('--batch_size', type=int, default=8,
                                  help='Number of images per batch.')

        self._parser.add_argument('--display_every', type=int, default=50,
                                  help='Number of iterations executed between'
                                       'two consecutive display of metrics')

        self._parser.add_argument('--gpu_mem_cap', type=int, default=0,
                                  help='Upper bound for GPU memory in MB. '
                                        'Default is 0 which means allow_growth '
                                        'will be used.')

        default_sign_key = DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self._parser.add_argument('--input_signature_key', type=str,
                                  default=default_sign_key,
                                  help='SavedModel signature to use for '
                                  'inference, defaults to: %s' % (
                                    default_sign_key
                                  ))

        self._parser.add_argument('--num_iterations', type=int, default=None,
                                  help='How many iterations(batches) to '
                                       'evaluate. If not supplied, the whole '
                                       'set will be evaluated.')

        self._parser.add_argument('--num_warmup_iterations', type=int,
                                  default=100,
                                  help='Number of initial iterations skipped '
                                       'from timing')

        self._add_bool_argument(
            name="use_xla",
            default=False,
            required=False,
            help='If set to True, the benchmark will use XLA JIT Compilation'
        )

        self._add_bool_argument(
            name="skip_accuracy_testing",
            default=False,
            required=False,
            help='If set to True, accuracy calculation will be skipped.'
        )

        self._add_bool_argument(
           name="use_synthetic_data",
           default=False,
           required=False,
           help='If set to True, one unique batch of random batch of data is '
                'generated and used at every iteration.'
        )

        # =========================== TF-TRT Flags ========================== #

        self._add_bool_argument(
            name="use_tftrt",
            default=False,
            required=False,
            help='If set to True, the inference graph will be converted using '
                 'TF-TRT graph converter.'
        )

        self._add_bool_argument(
            name="allow_build_at_runtime",
            default=False,
            required=False,
            help="Whether to build TensorRT engines during runtime."
        )

        self._parser.add_argument('--max_workspace_size', type=int,
                                 default=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
                                 help='The maximum GPU temporary memory which '
                                      'the TRT engine can use at execution '
                                      'time.')

        self._parser.add_argument('--minimum_segment_size', type=int, default=5,
                                  help='Minimum number of TensorFlow ops in a '
                                       'TRT engine.')

        self._parser.add_argument('--num_calib_inputs', type=int, default=500,
                                  help='Number of inputs (e.g. images) used '
                                       'for calibration (last batch is skipped '
                                       'in case it is not full)')

        self._add_bool_argument(
            name="optimize_offline",
            default=True,
            required=False,
            help='If set to True, TensorRT engines are built before runtime.'
        )

        self._parser.add_argument('--precision', type=str,
                                  choices=self.ALLOWED_TFTRT_PRECISION_MODES,
                                  default='FP32',
                                  help='Precision mode to use. FP16 and INT8 '
                                       'modes only works if --use_tftrt is '
                                       'used.')

        self._add_bool_argument(
            name="use_dynamic_shape",
            default=False,
            required=False,
            help='Whether to use implicit batch mode or dynamic shape mode.'
        )

    def _add_bool_argument(self, name=None, default=False, required=False, help=None):
            if not isinstance(default, bool):
                raise ValueError()

            feature_parser = self._parser.add_mutually_exclusive_group(\
                required=required
            )

            feature_parser.add_argument('--' + name, dest=name,
                                        action='store_true',
                                        help=help,
                                        default=default)

            feature_parser.add_argument('--no' + name, dest=name,
                                        action='store_false')

            feature_parser.set_defaults(name=default)

    def _validate_args(self, args):

        if args.data_dir is None:
            raise ValueError("--data_dir is required")

        elif not os.path.isdir(args.data_dir):
            raise RuntimeError("The path --data_dir=`{}` doesn't exist or is "
                               "not a directory".format(args.data_dir))

        if (
            args.num_iterations is not None and
            args.num_iterations <= args.num_warmup_iterations
        ):
            raise ValueError(
                '--num_iterations must be larger than --num_warmup_iterations '
                '({} <= {})'.format(args.num_iterations,
                                    args.num_warmup_iterations))

        if not args.use_tftrt:
            if args.use_dynamic_shape:
                raise ValueError('TensorRT must be enabled for Dynamic Shape '
                                 'support to be enabled (--use_tftrt).')

            if args.precision != 'FP32':
                raise ValueError('TensorRT must be enabled for FP16'
                                 'or INT8 modes (--use_tftrt).')

        else:
            if args.use_xla:
                raise ValueError("--use_xla flag is not supported with TF-TRT.")
                
            if args.precision not in self.ALLOWED_TFTRT_PRECISION_MODES:
                raise ValueError("The received --precision={} is not supported."
                                 " Allowed: {}".format(
                                    args.precision,
                                    self.ALLOWED_TFTRT_PRECISION_MODES
                ))

            if args.precision == 'INT8':

                if not args.calib_data_dir:
                    raise ValueError('--calib_data_dir is required for INT8 '
                                     'precision mode')

                elif not os.path.isdir(args.calib_data_dir):
                    raise RuntimeError("The path --calib_data_dir=`{}` doesn't "
                                       "exist or is not a directory".format(
                                            args.calib_data_dir))

                if args.use_dynamic_shape:
                    raise ValueError('TF-TRT does not support dynamic shape '
                                     'mode with INT8 calibration.')

                if args.num_calib_inputs <= args.batch_size:
                    raise ValueError(
                        '--num_calib_inputs must not be smaller than '
                        '--batch_size ({} <= {})'.format(
                        args.num_calib_inputs, args.batch_size))

    def _post_process_args(self, args):

        if args.num_iterations is None:
            args.num_iterations = (
                self.SAMPLES_IN_VALIDATION_SET // args.batch_size
            )

        return args

    def parse_args(self):
        args = self._parser.parse_args()
        args = self._post_process_args(args)
        self._validate_args(args)

        print('\nBenchmark arguments:')
        _print_dict(vars(args))

        return args
