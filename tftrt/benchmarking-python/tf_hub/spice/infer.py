<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
#!# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
=======
#!# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
>>>>>>> [Benchmarking-Py]  Add google/spice
=======
#!# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
>>>>>>> [Benchmarking-Py] spice - add run_inference.sh
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

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
import math
=======
>>>>>>> [Benchmarking-Py]  Add google/spice
=======
import math
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
=======
import math
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
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

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
=======
from dataloading import get_dataset_cola

>>>>>>> [Benchmarking-Py]  Add google/spice

=======
>>>>>>> [Benchmarking-Py] spice - add run_inference.sh
=======

>>>>>>> yapf cleaning
class CommandLineAPI(BaseCommandLineAPI):

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            "--samples_per_input",
            type=int,
            default=128,
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
            help="Input number of samples per input to generate random wave data."
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
=======
            help="Input data sequence length."
=======
            help="Input number of samples per input to generate input random wave data."
>>>>>>> [Benchmarking-Py] spice - add run_inference.sh
=======
            help=
<<<<<<< refs/remotes/origin/master
            "Input number of samples per input to generate input random wave data."
>>>>>>> yapf cleaning
=======
            "Input number of samples per input to generate random wave data."
>>>>>>> Address review comments
        )

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
>>>>>>> [Benchmarking-Py]  Add google/spice
=======
=======
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
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
<<<<<<< refs/remotes/origin/master
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
=======
>>>>>>> [Benchmarking-Py] Spice - run_all.sh


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

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
        # A single wave, 128 samples (8ms at 16kHz) long.
        wave = np.array(
            np.sin(np.linspace(-np.pi, np.pi, self._args.samples_per_input)),
            dtype=np.float32
        )

        # tile to 2048 samples. The model resizes the input to (2048,) automatically
        tile_factor = math.ceil(2048 / wave.shape[0])
        waves = np.expand_dims(np.tile(wave, tile_factor), axis=0)

        dataset = tf.data.Dataset.from_tensor_slices(waves)
<<<<<<< refs/remotes/origin/master
=======

        tf.random.set_seed(10)


        input = tf.random.uniform(
            shape=(1, self._args.samples_per_input),
            maxval=self._args.vocab_size,
            dtype=tf.int32
        )
        dataset = dataset.batch(self._args.batch_size)
=======
        # A single wave, 128 samples (8ms at 16kHz) long.
        wave = np.array(
            np.sin(np.linspace(-np.pi, np.pi, self._args.samples_per_input)),
            dtype=np.float32
        )

        # tile to 2048 samples. The model resizes the input to (2048,) automatically
        tile_factor = math.ceil(2048 / wave.shape[0])
        waves = np.expand_dims(np.tile(wave, tile_factor), axis=0)

        dataset = tf.data.Dataset.from_tensor_slices(waves)
>>>>>>> [Benchmarking-Py] spice - add run_inference.sh
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
>>>>>>> [Benchmarking-Py]  Add google/spice
=======
>>>>>>> Address review comments
        dataset = dataset.repeat()

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
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
        return None, "Raw Pitch Accuracy"
=======

        # NOTE: PLEASE ONLY MODIFY THE NAME OF THE ACCURACY METRIC

<<<<<<< refs/remotes/origin/master
        return None, "GLUE Score"
>>>>>>> [Benchmarking-Py]  Add google/spice
=======
=======
>>>>>>> Address review comments
        return None, "Raw Pitch Accuracy"
>>>>>>> [Benchmarking-Py] spice - add run_inference.sh


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)
    runner.execute_benchmark()
