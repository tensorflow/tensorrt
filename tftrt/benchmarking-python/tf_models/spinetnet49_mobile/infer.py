#!# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import math
import os
import sys

import numpy as np
import tensorflow as tf

# Allow import of top level python files
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)

sys.path.insert(0, parentdir)

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner


class CommandLineAPI(BaseCommandLineAPI):

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            '--input_size',
            type=int,
            default=384,
            help='Size of input images expected by the model'
        )

    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        # TODO: Remove when proper dataloading is implemented
        if not args.use_synthetic_data:
            raise ValueError(
                "This benchmark does not currently support non-synthetic data "
                "--use_synthetic_data"
            )
        # This model requires that the batch size is 1
        if args.batch_size != 1:
            raise ValueError(
                "This benchmark does not currently support "
                "--batch_size != 1"
            )


class BenchmarkRunner(BaseBenchmarkRunner):

    def get_dataset_batches(self):
        """Returns a list of batches of input samples.

        Each batch should be in the form [x, y], where
        x is a numpy array of the input samples for the batch, and
        y is a numpy array of the expected model outputs for the batch

        Returns:
        - dataset: a TF Dataset object
        - bypass_data_to_eval: any object type that will be passed unmodified to
                            `evaluate_result()`. If not necessary: `None`

        Note: script arguments can be accessed using `self._args.attr`
        """

        tf.random.set_seed(10)

        inputs = tf.random.uniform(
            shape=(1, self._args.input_size, self._args.input_size, 3),
            maxval=255,
            dtype=tf.int32
        )

        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        dataset = dataset.map(
            lambda x: {"inputs": tf.cast(x, tf.uint8)},
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.repeat()
        dataset = dataset.batch(self._args.batch_size)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, None

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr` """

        return data_batch, None

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        # NOTE : DO NOT MODIFY FOR NOW => We do not measure accuracy right now

        return predictions.numpy(), expected.numpy()

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """
        return None, "Raw Pitch Accuracy"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)
    runner.execute_benchmark()
