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
import csv
import tensorflow_datasets as tfds


def get_dataset(sequence_length=128, batch_size=32, vocab_size=32000):
    tf.random.set_seed(10)

    input_mask = tf.random.uniform(
        shape=(1, sequence_length),
        maxval=vocab_size,
        dtype=tf.int32
    )
    input_type_ids = tf.random.uniform(
        shape=(1, sequence_length),
        maxval=vocab_size,
        dtype=tf.int32
    )
    input_word_ids = tf.random.uniform(
        shape=(1, sequence_length),
        maxval=vocab_size,
        dtype=tf.int32
    )

    ds_mask = tf.data.Dataset.from_tensor_slices(input_mask)
    ds_typeids = tf.data.Dataset.from_tensor_slices(input_type_ids)
    ds_wordids = tf.data.Dataset.from_tensor_slices(input_word_ids)
    dataset = tf.data.Dataset.zip((ds_mask, ds_typeids, ds_wordids))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(count=1)  # loop over 1 batch
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))


def read_tsv(input_file):

    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, 'r') as f:
      reader = csv.reader(f, delimiter="\t", quotechar='|')
      lines = []
      for line in reader:
        lines.append(line)
      return lines


def get_cola_labels():
    return ["0","1"]

def get_dataset_cola(sequence_length=128, batch_size=32,
        vocab_size=32000, full_path_to_file='test_cola.tsv'):
     """Loads the data from the specified tsv file."""
     from transformers import BertTokenizer

     tz = BertTokenizer.from_pretrained("bert-base-cased")
     lines = read_tsv(full_path_to_file)
     guid_arr = []
     text_arr = []
     set_type = 'test'
     for (i,line) in enumerate(lines):
        # print("Working on line: {} with length {} \n".format(i,len(line)))
        print(line)
        guid = i # (set_type, i)
        text_a = convert_to_unicode(line[1])
        encoded = tz.encode_plus(
            text=text_a,  # the sentence to be encoded
            add_special_tokens=False,  # Add [CLS] and [SEP]
            max_length=sequence_length,  # maximum length of a sentence
            padding='max_length',
            return_tensors='tf',  # ask the function to return PyTorch tensors
            return_attention_mask = False  # Generate the attention mask
            )
        input_id =  encoded['input_ids'];

        # text_a = tz.convert_tokens_to_ids(tz.tokenize(text_a))
        label = "0"
        guid_arr.append(guid)
        text_arr.append(input_id)

     input_mask = tf.random.uniform(
         shape=(1, sequence_length),
         maxval=vocab_size,
         dtype=tf.int32
     )

     ds_mask = tf.data.Dataset.from_tensor_slices(input_mask)
     ds_typeids = tf.data.Dataset.from_tensor_slices(guid_arr)
     ds_wordids = tf.data.Dataset.from_tensor_slices(text_arr)
     dataset = tf.data.Dataset.zip((ds_mask, ds_typeids, ds_wordids))

     dataset = dataset.repeat()
     dataset = dataset.batch(batch_size)
     dataset = dataset.take(count=1)  # loop over 1 batch
     dataset = dataset.cache()
     dataset = dataset.repeat()
     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

     return dataset


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
            "--sequence_length",
            type=int,
            default=128,
            help="Input data sequence length."
        )

        self._parser.add_argument(
            "--vocab_size",
            type=int,
            required=True,
            help="Size of the vocabulory used for training. Refer to "
            "huggingface documentation."
        )

        # self._parser.add_argument(
        #     "--validate_output",
        #     action="store_true",
        #     help="Validates that the model returns the correct value. This "
        #     "only works with batch_size =32."
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

        #if not self._args.use_synthetic_data:
        if True:
            fullpath = "".join([self._args.data_dir, 'test_cola.tsv'])
            dataset = get_dataset_cola(sequence_length=self._args.sequence_length, batch_size=self._args.batch_size,
                    vocab_size=self._args.vocab_size, full_path_to_file=self._args.data_dir)
        else:
            tf.random.set_seed(10)

            input_mask = tf.random.uniform(
                shape=(1, self._args.sequence_length),
                maxval=self._args.vocab_size,
                dtype=tf.int32
            )
            input_type_ids = tf.random.uniform(
                shape=(1, self._args.sequence_length),
                maxval=self._args.vocab_size,
                dtype=tf.int32
            )
            input_word_ids = tf.random.uniform(
                shape=(1, self._args.sequence_length),
                maxval=self._args.vocab_size,
                dtype=tf.int32
            )

            ds_mask = tf.data.Dataset.from_tensor_slices(input_mask)
            ds_typeids = tf.data.Dataset.from_tensor_slices(input_type_ids)
            ds_wordids = tf.data.Dataset.from_tensor_slices(input_word_ids)
            dataset = tf.data.Dataset.zip((ds_mask, ds_typeids, ds_wordids))
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

        input_mask, input_type_ids, input_word_ids = data_batch
        x =  {
            "input_mask":input_mask,
            "input_type_ids":input_type_ids,
            "input_word_ids":input_word_ids
        }
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
