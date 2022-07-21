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

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            "--tokenizer_model_dir",
            type=str,
            required=True,
            help="Directory containing the tokenizer model from HuggingFace."
        )

        self._parser.add_argument(
            "--sequence_length",
            type=int,
            default=1024,
            help="Input data sequence length."
        )

        self._parser.add_argument(
            "--vocab_size",
            type=int,
            default=50257,
            help="Size of the vocabulory used for training. Refer to "
            "huggingface documentation."
        )

    def _validate_args(self, args):
        super(CommandLineAPI, self)._validate_args(args)

        if not args.use_synthetic_data:
            raise ValueError(
                "The use of --use_synthetic_data is necessary for this model."
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

        tf.random.set_seed(12345)

        input_ids = tf.random.uniform(
            shape=(1, self._args.sequence_length),
            maxval=self._args.vocab_size,
            dtype=tf.int32
        )
        ds_input_ids = tf.data.Dataset.from_tensor_slices(input_ids)

        attention_mask = tf.ones([1, self._args.sequence_length],
                                 dtype=tf.int32)
        ds_attention_mask = tf.data.Dataset.from_tensor_slices(attention_mask)

        token_type_ids = tf.random.uniform(
            shape=(1, self._args.sequence_length), maxval=2, dtype=tf.int32
        )
        ds_token_type_ids = tf.data.Dataset.from_tensor_slices(token_type_ids)

        dataset = tf.data.Dataset.zip((
            ds_attention_mask,
            ds_input_ids,
            ds_token_type_ids,
        ))

        def map_dict_fn(attention_mask, input_ids, token_type_ids):
            return {
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids
            }

        dataset = dataset.map(map_dict_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, None

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """

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

        return None, "Perplexity (PPL)"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)
    runner.execute_benchmark()
