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

from functools import partial

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

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._add_bool_argument(
            name="amp",
            default=False,
            required=False,
            help="Whether the model was trained using mixed-precision"
        )

        self._parser.add_argument(
            "--test_filename",
            type=str,
            default="test.tfrecord",
            help="Name of the output tensor, see `analysis.txt`"
        )

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

        feature_spec = {
            "uid": tf.io.FixedLenFeature([], tf.int64),
            "item_id": tf.io.FixedLenFeature([], tf.int64),
            "cate_id": tf.io.FixedLenFeature([], tf.int64),
            "long_hist_item": tf.io.FixedLenFeature([90], tf.int64),
            "long_hist_cate": tf.io.FixedLenFeature([90], tf.int64),
            "short_hist_item": tf.io.FixedLenFeature([10], tf.int64),
            "short_hist_cate": tf.io.FixedLenFeature([10], tf.int64),
            "short_neg_hist_item": tf.io.FixedLenFeature([10], tf.int64),
            "short_neg_hist_cate": tf.io.FixedLenFeature([10], tf.int64),
            "long_sequence_mask": tf.io.FixedLenFeature([90], tf.float32),
            "short_sequence_mask": tf.io.FixedLenFeature([10], tf.float32),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        def _remap_values(sample, out_type):
            sample["short_sequence_mask"] = tf.cast(
                sample["short_sequence_mask"], dtype=out_type
            )
            label = sample.pop("label")
            return sample, label

        dataset = tf.data.TFRecordDataset([
            os.path.join(self._args.data_dir, self._args.test_filename)
        ])

        dataset = dataset.batch(self._args.batch_size, drop_remainder=False)

        dataset = dataset.map(
            map_func=partial(tf.io.parse_example, features=feature_spec),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        out_type = tf.float16 if self._args.amp else tf.float32
        dataset = dataset.map(
            map_func=partial(_remap_values, out_type=out_type),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

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

        predictions = {key: val.numpy() for key, val in predictions.items()}
        expected = {"sim_model_1": expected.numpy()}

        return predictions, expected


    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """
        auc = tf.keras.metrics.AUC(from_logits=True, num_thresholds=8000)

        logit_diff = (
            predictions["sim_model_1"][:, 0] - predictions["sim_model_1"][:, 1]
        )
        auc_score = auc(expected["sim_model_1"], logit_diff).numpy()


        return auc_score * 100, "ROC AUC"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
