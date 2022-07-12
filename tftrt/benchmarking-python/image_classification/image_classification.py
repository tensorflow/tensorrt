# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import preprocessing

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
            default=1001,
            help='Number of classes used when training '
            'the model'
        )

        self._parser.add_argument(
            '--preprocess_method',
            type=str,
            choices=['vgg', 'inception', 'resnet50_v1_5_tf1_ngc_preprocess'],
            default='vgg',
            help='The image preprocessing method used in '
            'dataloading.'
        )

    def _post_process_args(self, args):
        args = super(CommandLineAPI, self)._post_process_args(args)
        args.labels_shift = 1 if args.num_classes == 1001 else 0

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

        def get_files(data_dir, filename_pattern):
            if data_dir is None:
                return []

            files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))

            if not files:
                raise ValueError(
                    'Can not find any files in {} with '
                    'pattern "{}"'.format(data_dir, filename_pattern)
                )
            return files

        def deserialize_image_record(record):
            feature_map = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1)
            }
            with tf.compat.v1.name_scope('deserialize_image_record'):
                obj = tf.io.parse_single_example(
                    serialized=record, features=feature_map
                )
                imgdata = obj['image/encoded']
                label = tf.cast(obj['image/class/label'], tf.int32)
            return imgdata, label

        def get_preprocess_fn(preprocess_method, input_size):
            """Creates a function to parse and process a TFRecord
            input_size: int
            returns: function, the preprocessing function for a record
            """
            if preprocess_method == 'vgg':
                preprocess_fn = preprocessing.vgg_preprocess
            elif preprocess_method == 'inception':
                preprocess_fn = preprocessing.inception_preprocess
            elif preprocess_method == 'resnet50_v1_5_tf1_ngc_preprocess':
                preprocess_fn = preprocessing.resnet50_v1_5_tf1_ngc_preprocess
            else:
                raise ValueError(
                    'Invalid preprocessing method {}'.format(preprocess_method)
                )

            def preprocess_sample_fn(record):
                # Parse TFRecord
                imgdata, label = deserialize_image_record(record)
                label -= 1  # Change to 0-based (don't use background class)
                try:
                    image = tf.image.decode_jpeg(
                        imgdata,
                        channels=3,
                        fancy_upscaling=False,
                        dct_method='INTEGER_FAST'
                    )
                except:
                    image = tf.image.decode_png(imgdata, channels=3)
                # Use model's preprocessing function
                image = preprocess_fn(image, input_size, input_size)
                return image, label

            return preprocess_sample_fn

        data_files = get_files(self._args.data_dir, 'validation*')
        dataset = tf.data.Dataset.from_tensor_slices(data_files)

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.experimental.AUTOTUNE,
            block_length=max(self._args.batch_size, 32)
        )

        # preprocess function for input data
        preprocess_fn = get_preprocess_fn(
            preprocess_method=self._args.preprocess_method,
            input_size=self._args.input_size
        )

        dataset = dataset.map(
            map_func=preprocess_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(self._args.batch_size, drop_remainder=False)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
