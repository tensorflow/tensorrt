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

import json
import os
import shlex
import sys
import subprocess

import numpy as np

import tensorflow as tf

import tokenization
import squad_lib

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
            "--max_seq_length",
            type=int,
            default=384,
            help="The maximum total input sequence length "
            "after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and "
            "sequences shorter than this will be "
            "padded."
        )

        self._parser.add_argument(
            "--doc_stride",
            type=int,
            default=128,
            help="When splitting up a long document into "
            "chunks, how much stride to take between "
            "chunks."
        )

        self._parser.add_argument(
            "--max_query_length",
            type=int,
            default=64,
            help="The maximum number of tokens for the "
            "question. Questions longer than this will "
            "be truncated to this length."
        )

        self._add_bool_argument(
            name="version_2_with_negative",
            default=False,
            required=False,
            help="If true, the SQuAD examples contain some that do not have an "
            "answer."
        )

        self._parser.add_argument(
            "--output_dir",
            type=str,
            default="/tmp",
            help="The output directory where the model "
            "checkpoints and predictions will be "
            "written."
        )

        self._parser.add_argument(
            "--n_best_size",
            type=int,
            default=20,
            help="The total number of n-best predictions to "
            "generate in the nbest_predictions.json "
            "output file."
        )

        self._parser.add_argument(
            "--max_answer_length",
            type=int,
            default=30,
            help="The maximum length of an answer that can "
            "be generated. This is needed because the "
            "start and end predictions are not "
            "conditioned on one another."
        )

        self._add_bool_argument(
            name="do_lower_case",
            default=True,
            required=False,
            help="Whether to lower case the input text. True for uncased "
            "models, False for cased models."
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

        predict_file = os.path.join(
            self._args.data_dir,
            "squad/v1.1/dev-v1.1.json"
        )

        vocab_file = os.path.join(
            self._args.data_dir,
            "google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
        )

        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=self._args.do_lower_case
        )

        eval_examples = squad_lib.read_squad_examples(
            input_file=predict_file,
            is_training=False,
            version_2_with_negative=self._args.version_2_with_negative
        )

        eval_features = []

        def append_feature(feature, is_padding):
            eval_features.append(feature)

        squad_lib.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=self._args.max_seq_length,
            doc_stride=self._args.doc_stride,
            max_query_length=self._args.max_query_length,
            is_training=False,
            output_fn=append_feature,
            batch_size=1
        )

        all_unique_ids = tf.convert_to_tensor(
            [f.unique_id for f in eval_features], dtype=tf.int32)
        all_input_ids = tf.convert_to_tensor(
            [f.input_ids for f in eval_features], dtype=tf.int32)
        all_input_mask = tf.convert_to_tensor(
            [f.input_mask for f in eval_features], dtype=tf.int32)
        all_segment_ids = tf.convert_to_tensor(
            [f.segment_ids for f in eval_features], dtype=tf.int32)

        dataset = tf.data.Dataset.from_tensor_slices(
            (all_unique_ids, all_input_ids, all_input_mask, all_segment_ids)
        )

        dataset = dataset.batch(self._args.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        bypass_data_to_eval = {
            "eval_features": eval_features,
            "eval_examples": eval_examples
        }

        return dataset, bypass_data_to_eval

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """

        unique_ids, input_ids, input_mask, segment_ids = data_batch

        return {
            "input_word_ids": input_ids,
            "input_type_ids": segment_ids,
            "input_mask": input_mask
        }, unique_ids

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        for key, val in predictions.items():
            predictions[key] = val.numpy()

        expected = expected.numpy()

        return predictions, expected

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        eval_examples = bypass_data_to_eval["eval_examples"]
        eval_features = bypass_data_to_eval["eval_features"]

        results = []
        for i, unique_id in enumerate(np.squeeze(expected["data"])):
            start_logits = predictions["start_positions"][i].tolist()
            end_logits = predictions["end_positions"][i].tolist()
            results.append(
                squad_lib.RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits
                )
            )

        output_dir = self._args.output_dir

        output_prediction_file = os.path.join(output_dir, "predictions.json")
        output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")

        squad_lib.write_predictions(
            all_examples=eval_examples,
            all_features=eval_features,
            all_results=results,
            n_best_size=self._args.n_best_size,
            max_answer_length=self._args.max_answer_length,
            do_lower_case=self._args.do_lower_case,
            output_prediction_file=output_prediction_file,
            output_nbest_file=output_nbest_file,
            output_null_log_odds_file=output_null_log_odds_file,
            version_2_with_negative=self._args.version_2_with_negative,
            null_score_diff_threshold=0.0,
            verbose=self._args.debug
        )

        if self._args.version_2_with_negative:
            dev_file = "dev-v2.0.json"
            eval_file = "evaluate-v2.0.py"
        else:
            dev_file = "dev-v1.1.json"
            eval_file = "evaluate-v1.1.py"

        squad_dir = os.path.join(self._args.data_dir, "squad", "v1.1")
        command_str = (
            f"{sys.executable} {os.path.join(squad_dir, eval_file)} "
            f"{os.path.join(squad_dir, dev_file)} {output_prediction_file}"
        )
        if self._args.debug:
            print(f"\nExecuting: `{command_str}`\n")

        eval_out = subprocess.check_output(shlex.split(command_str))

        # scores: {'exact_match': 84.91958372753075, 'f1': 91.43193117076082}
        scores = json.loads(eval_out.decode("UTF-8").strip())

        metric_units = "f1"

        return scores[metric_units], metric_units


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
