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

benchmark_base_dir = os.path.dirname(currentdir)
sys.path.insert(0, benchmark_base_dir)

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner


class CommandLineAPI(BaseCommandLineAPI):

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        # self._parser.add_argument(
        #     "--sequence_length",
        #     type=int,
        #     default=128,
        #     help="Input data sequence length."
        # )


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

        # seq = generate_a_sequence(self._args.sequence_length)

        # - https://www.tensorflow.org/guide/data_performance
        # - https://www.tensorflow.org/guide/data
        # dataset = tf.data....

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

        # NOTE : DO NOT MODIFY FOR NOW => We do not measure accuracy right now

        return predictions.numpy(), expected.numpy()

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        # NOTE: PLEASE ONLY MODIFY THE NAME OF THE ACCURACY METRIC

        return None, "<ACCURACY METRIC NAME>"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)
    runner.execute_benchmark()

################ TO BE REMOVED - HIGH LEVEL CONCEPT #####################

import time

model_fn = load_my_model("/path/to/my/model")

dataset, _ = get_dataset_batches()  # dataset, None

ds_iter = iter(dataset)

for idx, batch in enumerate(ds_iter):
    print(f"Batch ID: {idx + 1} - Data: {batch}")

    # - IF NEEDED - This transforms the inputs - Most cases it doesn't do anything
    # let's say transforming a list into a dict() or reverse
    batch = preprocess_model_inputs(batch)

    start_t = time.time()
    outputs = model_fn(batch)
    print(f"Inference Time: {(time.time() - start_t)*1000:.1f}ms")  # 0.001

    ## post my outputs to "measure accuracy"
    ## note: we skip that

print("Success")
sys.exit(0)
