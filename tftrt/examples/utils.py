#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import os
import time

from contextlib import contextmanager

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
def timed_section(msg):
    print('\n[START] {}'.format(msg))
    start_time = time.time()
    yield
    print("[END] Duration: {:.1f}s".format(time.time() - start_time))
    print("=" * 80, "\n")


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


def config_gpu_memory(gpu_mem_cap):
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


def get_graph_func(
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

        with timed_section('Loading TensorFlow native model...'):
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

            with timed_section('TF-TRT graph conversion and INT8 '
                               'calibration ...'):
                graph_func = converter.convert(
                    calibration_input_fn=tf.autograph.experimental.do_not_convert(
                        calibration_input_fn
                    )
                )

        else:
            with timed_section('TF-TRT graph conversion ...'):
                graph_func = converter.convert()

        if optimize_offline or use_dynamic_shape:

            _check_input_fn(
                optimize_offline_input_fn,
                "optimize_offline_input_fn"
            )

            with timed_section('Building TensorRT engines...'):
                converter.build(input_fn=tf.autograph.experimental.do_not_convert(
                    optimize_offline_input_fn
                ))

        if output_saved_model_dir is not None:

            with timed_section('Saving converted graph with TF-TRT ...'):
                converter.save(output_saved_model_dir)
                print("Converted graph saved to `{}`".format(
                    output_saved_model_dir))

    return graph_func
