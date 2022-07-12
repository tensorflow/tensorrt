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

import numpy as np

import tensorflow as tf

# Allow import of top level python files
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

benchmark_base_dir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, benchmark_base_dir)

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner


class CommandLineAPI(BaseCommandLineAPI):

    ALLOWED_VOCAB_SIZES = [
        30522,  # BERT Uncased
        28996,  # BERT Cased
        50265,  # BART
    ]

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            "--sequence_length",
            type=int,
            default=128,
            help="Input data sequence length."
        )

        self._parser.add_argument(
            "--vocab_size",
            type=int,
            required=True,
            choices=self.ALLOWED_VOCAB_SIZES,
            help="Size of the vocabulory used for training. Refer to "
            "huggingface documentation."
        )

        # self._parser.add_argument(
        #     "--validate_output",
        #     action="store_true",
        #     help="Validates that the model returns the correct value. This "
        #     "only works with batch_size =32."
        # )

    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        # if args.validate_output and args.batch_size != 32:
        #     raise ValueError("Output validation only supports batch size 32.")

        # TODO: Remove when proper dataloading is implemented
        if args.num_iterations is None:
            raise ValueError(
                "This benchmark does not currently support "
                "--num_iterations=None"
            )

    def _post_process_args(self, args):
        args = super(CommandLineAPI, self)._post_process_args(args)

        return args


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%% IMPLEMENT MODEL-SPECIFIC FUNCTIONS HERE %%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


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

        if not self._args.use_synthetic_data:
            raise NotImplementedError()

        tf.random.set_seed(10)

        input_data = tf.random.uniform(
            shape=(1, self._args.sequence_length),
            maxval=self._args.vocab_size,
            dtype=tf.int32
        )

        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._args.batch_size)
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset, None

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """

        x = data_batch
        return x, None

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        return predictions.numpy(), expected.numpy()

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        return None, "Top-1 Accuracy %"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
