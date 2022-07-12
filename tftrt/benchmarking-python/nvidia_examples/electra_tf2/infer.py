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

import pickle
import json
import os
import shlex
import sys
import subprocess

import numpy as np

import tensorflow as tf

from tokenization import ElectraTokenizer
from squad_utils import (
    SquadResult, SquadV1Processor, SquadV2Processor,
    squad_convert_examples_to_features, get_answers
)

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
            "--electra_model",
            type=str,
            default="google/electra-base-discriminator",
            help="Model selected in the list"
        )

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

        self._add_bool_argument(
            name="joint_head",
            default=True,
            required=False,
            help="Prevent jointly predict the start and end positions"
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
            "--beam_size",
            type=int,
            default=4,
            help="Beam size when doing joint predictions"
        )

        self._parser.add_argument(
            "--null_score_diff_threshold",
            type=float,
            default=-5.6,
            help="If null_score - best_non_null is greater "
            "than the threshold predict null."
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
            default=False,
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

        def get_dataset_from_features(features, batch_size):

            all_unique_ids = tf.convert_to_tensor(
                [f.unique_id for f in features], dtype=tf.int64)
            all_input_ids = tf.convert_to_tensor(
                [f.input_ids for f in features], dtype=tf.int64)
            all_input_mask = tf.convert_to_tensor(
                [f.attention_mask for f in features], dtype=tf.int64)
            all_segment_ids = tf.convert_to_tensor(
                [f.token_type_ids for f in features], dtype=tf.int64)
            all_start_pos = tf.convert_to_tensor(
                [f.start_position for f in features], dtype=tf.int64)
            all_end_pos = tf.convert_to_tensor(
                [f.end_position for f in features], dtype=tf.int64)
            all_cls_index = tf.convert_to_tensor(
                [f.cls_index for f in features], dtype=tf.int64)
            all_p_mask = tf.convert_to_tensor(
                [f.p_mask for f in features], dtype=tf.float32)
            all_is_impossible = tf.convert_to_tensor(
                [f.is_impossible for f in features], dtype=tf.float32)

            dataset = tf.data.Dataset.from_tensor_slices((
                all_unique_ids, all_input_ids, all_input_mask, all_segment_ids,
                all_start_pos, all_end_pos, all_cls_index, all_p_mask,
                all_is_impossible
            ))

            dataset = dataset.batch(batch_size, drop_remainder=False)
            dataset = dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            )

            return dataset

        tokenizer = ElectraTokenizer.from_pretrained(
            self._args.electra_model, cache_dir=None
        )
        processor = (
            SquadV2Processor()
            if self._args.version_2_with_negative else SquadV1Processor()
        )

        dev_examples = processor.get_dev_examples(self._args.data_dir)

        print("***** Loading features *****")
        # Load cached features
        squad_version = '2.0' if self._args.version_2_with_negative else '1.1'
        cache_dir = self._args.data_dir

        cached_dev_features_file = os.path.join(
            cache_dir.rstrip('/'), f"TF2_dev-v{squad_version}.json_"
            f"{self._args.electra_model.split('/')[1]}_"
            f"{self._args.max_seq_length}_"
            f"{self._args.doc_stride}_"
            f"{self._args.max_query_length}"
        )

        try:
            with open(cached_dev_features_file, "rb") as reader:
                dev_features = pickle.load(reader)
        except:
            dev_features = (
                squad_convert_examples_to_features(
                    examples=dev_examples,
                    tokenizer=tokenizer,
                    max_seq_length=self._args.max_seq_length,
                    doc_stride=self._args.doc_stride,
                    max_query_length=self._args.max_query_length,
                    is_training=False,
                    return_dataset="",
                )
            )

            print(f"**** Building Cache Files: {cached_dev_features_file} ****")
            with open(cached_dev_features_file, "wb") as writer:
                pickle.dump(dev_features, writer)

        dev_dataset = get_dataset_from_features(
            features=dev_features, batch_size=self._args.batch_size
        )

        bypass_data_to_eval = {
            "dev_features": dev_features,
            "dev_examples": dev_examples
        }

        return dev_dataset, bypass_data_to_eval

    def preprocess_model_inputs(self, data_batch):
        """This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """

        unique_ids, input_ids, input_mask, segment_ids, start_positions, \
        end_positions, cls_index, p_mask, is_impossible = data_batch

        x = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "cls_index": cls_index,
            "p_mask": p_mask
        }

        y = {
            "unique_ids": unique_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "is_impossible": is_impossible,
        }

        return x, y

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        for key, val in predictions.items():
            predictions[key] = val.numpy()

        for key, val in expected.items():
            expected[key] = val.numpy()

        return predictions, expected


    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        result = []
        for i, unique_id in enumerate(np.squeeze(expected["unique_ids"])):
            start_logits = predictions['tf_electra_for_question_answering'][i]
            start_top_index = predictions['tf_electra_for_question_answering_1'][i]
            end_logits = predictions['tf_electra_for_question_answering_2'][i]
            end_top_index = predictions['tf_electra_for_question_answering_3'][i]
            cls_logits = predictions['tf_electra_for_question_answering_4'][i]

            result.append(
                SquadResult(
                    unique_id,
                    start_logits.tolist(),
                    end_logits.tolist(),
                    start_top_index=start_top_index.tolist(),
                    end_top_index=end_top_index.tolist(),
                    cls_logits=cls_logits.tolist(),
                )
            )

        dev_features = bypass_data_to_eval["dev_features"]
        dev_examples = bypass_data_to_eval["dev_examples"]

        answers, nbest_answers = get_answers(
            dev_examples, dev_features, result, self._args
        )

        output_prediction_file = os.path.join(
            self._args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            self._args.output_dir, "nbest_predictions.json")

        with open(output_prediction_file, "w") as f:
            f.write(json.dumps(answers, indent=4) + "\n")
        with open(output_nbest_file, "w") as f:
            f.write(json.dumps(nbest_answers, indent=4) + "\n")

        if self._args.version_2_with_negative:
            dev_file = "dev-v2.0.json"
            eval_file = "evaluate-v2.0.py"
        else:
            dev_file = "dev-v1.1.json"
            eval_file = "evaluate-v1.1.py"

        command_str = (
            f"{sys.executable} {os.path.join(self._args.data_dir, eval_file)} "
            f"{os.path.join(self._args.data_dir, dev_file)} "
            f"{output_prediction_file}"
        )
        if self._args.debug:
            print(f"\nExecuting: `{command_str}`\n")

        eval_out = subprocess.check_output(shlex.split(command_str))

        # scores: {'exact_match': 87.06717123935667, 'f1': 92.78048326711645}
        scores = json.loads(eval_out.decode("UTF-8").strip())

        if self._args.debug:
            print("scores:", scores)

        metric_units = "f1"

        return scores[metric_units], metric_units


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
