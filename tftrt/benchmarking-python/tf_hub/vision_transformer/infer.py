# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
=======
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
>>>>>>> Fix preprocessing for vit
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
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master

=======
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======

>>>>>>> Fix preprocessing for vit
from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner

from image_classification.dataloading import get_dataloader
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
from image_classification import preprocessing
=======
import image_classification.preprocessing as preprocessing
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
from image_classification import preprocessing
>>>>>>> Fix preprocessing for vit


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
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
            default=1000,
=======
            default=1001,
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
            default=1000,
>>>>>>> Fix preprocessing for vit
            help='Number of classes used when training '
            'the model'
        )

        self._parser.add_argument(
            '--preprocess_method',
            type=str,
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
            choices=['vision_transformer'],
            default='vision_transformer',
=======
            choices=['vgg', 'inception', 'resnet50_v1_5_tf1_ngc'],
            default='vgg',
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
            choices=['vision_transformer'],
            default='vision_transformer',
>>>>>>> Fix preprocessing for vit
            help='The image preprocessing method used in dataloading.'
        )

    def _post_process_args(self, args):
        args = super(CommandLineAPI, self)._post_process_args(args)
        args.labels_shift = 1 if args.num_classes == 1001 else 0

        return args
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master

    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        if args.input_size != 224:
            raise ValueError(
                "The argument --input_size must be equal to 224 for this model."
            )

        if args.num_classes != 1000:
            raise ValueError(
                "The argument --num_classes must be equal to 1000 for this model."
            )
=======
    
=======

>>>>>>> Clean up to pass pytest
    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        if args.input_size != 224:
            raise ValueError(
                "The argument --input_size must be equal to 224 for this model."
            )

        if args.num_classes != 1000:
            raise ValueError(
                "The argument --num_classes must be equal to 1000 for this model."
            )
<<<<<<< refs/remotes/origin/master



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%% IMPLEMENT MODEL-SPECIFIC FUNCTIONS HERE %%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
>>>>>>> Init vit scripts. Needs to tune args in scripts

=======
>>>>>>> Fix preprocessing for vit


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

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
=======
>>>>>>> Fix preprocessing for vit
        predictions = predictions.numpy()

        if len(predictions.shape) != 1:
            predictions = tf.math.argmax(predictions, axis=1)
            predictions = predictions.numpy().reshape(-1)

        predictions - self._args.labels_shift

        return predictions - self._args.labels_shift, expected.numpy()
<<<<<<< refs/remotes/origin/master
=======
        return predictions.numpy(), expected.numpy()
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
>>>>>>> Fix preprocessing for vit

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
=======
>>>>>>> Fix preprocessing for vit
        return (
            np.mean(predictions["data"] == expected["data"]) * 100.0,
            "Top-1 Accuracy %"
        )
<<<<<<< refs/remotes/origin/master
=======
        return None, ""
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
>>>>>>> Fix preprocessing for vit


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
