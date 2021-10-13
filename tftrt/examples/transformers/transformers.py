#!# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import sys

import logging
import multiprocessing
import time

from functools import partial

import numpy as np
import tensorflow as tf

from statistics import mean

# Import of Top Level file `utils.py`
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import utils as tftrt_utils
from benchmark_runner import BaseBenchmarkRunner


class CommandLineAPI(tftrt_utils.BaseCommandLineAPI):

    # SAMPLES_IN_VALIDATION_SET = 50000

    ALLOWED_VOCAB_SIZES = [
        30522,  # BERT Uncased
        28996,  # BERT Cased
        50265,  # BART
    ]

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument('--sequence_length', type=int, default=128,
                            help='Directory containing the input saved model.')

        self._parser.add_argument('--vocab_size', type=int, required=True,
                                  choices=self.ALLOWED_VOCAB_SIZES,
                                  help='Size of the vocabulory used for '
                                       'training. Refer to huggingface '
                                       'documentation.')

        self._parser.add_argument('--validate_output', action='store_true',
                            help='Validates that the model returns the correct '
                            'value. This only works with batch_size =32.')


    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        if args.validate_output and args.batch_size != 32:
            raise ValueError("Output validation only supports batch size 32.")

        # TODO: Remove when proper dataloading is implemented
        if args.num_iterations is None:
            raise ValueError("This benchmark does not currently support "
                             "--num_iterations=None")

    # TODO: Remove when proper dataloading is implemented
    def _post_process_args(self, args):
        return args


class BenchmarkRunner(BaseBenchmarkRunner):

    ACCURACY_METRIC_NAME = "mAP"

    def before_benchmark(self, **kwargs):
        pass

    def compute_accuracy_metric(self, batch_size, steps_executed, **kwargs):
        pass

    def process_model_output(self, outputs, batch_y, **kwargs):
        pass

# def validate_model_artifacts(infer_func, model_dir, use_tftrt, precision):
#     numpy_asset_dir = os.path.join(model_dir, "numpy_assets")
#
#     input_data = np.load(os.path.join(numpy_asset_dir, 'input_data.npy'))
#     input_data = tf.constant(input_data, dtype=tf.int32)
#
#     output = infer_func(input_ids=input_data)
#
#     if use_tftrt:
#         if precision == "fp16":
#             rtol=1e-2
#             atol=2e-1
#         else:
#             rtol=1e-2
#             atol=5e-2
#     else:
#         rtol=1e-5
#         atol=1e-8
#
#     for key in output.keys():
#         target = np.load(os.path.join(numpy_asset_dir, '%s.npy' % key))
#         np.testing.assert_allclose(
#             target, output[key].numpy(), rtol=rtol, atol=atol
#         )
#     print("\n*****************************************************************")
#     print("Model was validated with success ...")
#     print("*****************************************************************\n")


def get_dataset(batch_size, seq_len, vocab_size, use_synthetic_data):

    if not use_synthetic_data:
        raise NotImplementedError()

    tf.random.set_seed(10)
    input_data = tf.random.uniform(shape=(1, seq_len), maxval=vocab_size,
                                   dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(count=1)  # loop over 1 batch
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    def _input_fn(build_steps, model_phase):

        dataset = dataloader_fn(
            batch_size=args.batch_size,
            seq_len=args.sequence_length,
            vocab_size=args.vocab_size
        )

        for i, (input_batch) in enumerate(dataset):
            if i >= build_steps:
                break

            print("* [%s] - step %04d/%04d" % (
                model_phase, i + 1, build_steps
            ))
            yield input_batch,

    calibration_input_fn = partial(
        _input_fn,
        build_steps=args.num_calib_inputs // args.batch_size,
        model_phase="Calibration"
    )
    optimize_offline_input_fn = partial(
        _input_fn,
        build_steps=1,
        model_phase="Building"
    )

    runner = BenchmarkRunner(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        allow_build_at_runtime=args.allow_build_at_runtime,
        calibration_input_fn=calibration_input_fn,
        gpu_mem_cap=args.gpu_mem_cap,
        input_signature_key=args.input_signature_key,
        max_workspace_size_bytes=args.max_workspace_size,
        minimum_segment_size=args.minimum_segment_size,
        num_calib_inputs=args.num_calib_inputs,
        optimize_offline=args.optimize_offline,
        optimize_offline_input_fn=optimize_offline_input_fn,
        precision_mode=args.precision,
        use_dynamic_shape=args.use_dynamic_shape,
        use_tftrt=args.use_tftrt)

    # if args.validate_output:
    #     # artifacts only generated for BS == 32
    #     validate_model_artifacts(
    #         graph_func,
    #         args.input_saved_model_dir,
    #         args.use_tftrt,
    #         args.precision.lower()
    #     )

    get_benchmark_input_fn = partial(
        get_dataset,
        seq_len=args.sequence_length,
        vocab_size=args.vocab_size
    )

    runner.execute_benchmark(
        batch_size=args.batch_size,
        display_every=args.display_every,
        get_benchmark_input_fn=get_benchmark_input_fn,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        skip_accuracy_testing=(
            args.use_synthetic_data or args.skip_accuracy_testing
        ),
        use_synthetic_data=args.use_synthetic_data,
    )
