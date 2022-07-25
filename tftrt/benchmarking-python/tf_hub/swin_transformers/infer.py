# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
sys.path.insert(0, os.path.join(benchmark_base_dir, "image_classification"))

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner

from image_classification.dataloading import get_dataloader
from image_classification import preprocessing


class CommandLineAPI(BaseCommandLineAPI):

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            '--input_size',
            type=int,
            default=224,
            help='Size of input images expected by the '
            'model'
        )

        self._parser.add_argument(
            '--num_classes',
            type=int,
            default=1000,
            help='Number of classes used when training '
            'the model'
        )

        self._parser.add_argument(
            '--preprocess_method',
            type=str,
            choices=['swin_transformer'],
            default='swin_transformer',
            help='The image preprocessing method used in dataloading.'
        )

    def _post_process_args(self, args):
        args = super(CommandLineAPI, self)._post_process_args(args)
        args.labels_shift = 1 if args.num_classes == 1001 else 0

        return args

    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        if args.num_classes != 1000:
            raise ValueError(
                "The argument --num_classes must be equal to 1000 for this model."
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

        dataset = get_dataloader(self._args)

        return dataset, None

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """

        x, y = data_batch
        return x, y

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        predictions = predictions.numpy()

        if len(predictions.shape) != 1:
            predictions = tf.math.argmax(predictions, axis=1)
            predictions = predictions.numpy().reshape(-1)

        predictions - self._args.labels_shift

        return predictions - self._args.labels_shift, expected.numpy()

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        return (
            np.mean(predictions["data"] == expected["data"]) * 100.0,
            "Top-1 Accuracy %"
        )


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
