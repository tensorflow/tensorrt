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

import glob
import multiprocessing
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

from utils import (
    create_coco_format_dataset,
    dataset_parser,
    extract_coco_groundtruth,
    process_predictions,
    EvaluationMetric,
)


class CommandLineAPI(BaseCommandLineAPI):

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            "--max_num_instances",
            type=int,
            default=100,
            required=False,
            help="Maximum Number of Instances"
        )

        self._parser.add_argument(
            "--image_size_h",
            type=int,
            default=832,
            required=False,
            help="Input Image Height"
        )

        self._parser.add_argument(
            "--image_size_w",
            type=int,
            default=1344,
            required=False,
            help="Input Image Width"
        )

        self._add_bool_argument(
            name="use_category",
            default=True,
            required=False,
            help="If set to True, MRCNN will use categories."
        )

        self._parser.add_argument(
            "--min_level",
            type=int,
            default=2,
            required=False,
            help="MRCNN Min Level"
        )

        self._parser.add_argument(
            "--max_level",
            type=int,
            default=6,
            required=False,
            help="MRCNN Max Level"
        )

        self._parser.add_argument(
            "--num_anchors",
            type=int,
            default=3,
            required=False,
            help="MRCNN Number of Anchors"
        )

        self._parser.add_argument(
            "--gt_mask_size",
            type=int,
            # size specified in the config is 112, however
            # there is hard-coded +4 in the code
            default=116,
            required=False,
            help="MRCNN GT Mask Size"
        )

        self._parser.add_argument(
            "--eval_split_pattern",
            type=str,
            default="val*.tfrecord",
            required=False,
            help="Evaluation dataset split pattern"
        )

    def _post_process_args(self, args):
        args = super(CommandLineAPI, self)._post_process_args(args)
        args.image_size = (args.image_size_h, args.image_size_w)

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

        eval_files = glob.glob(
            os.path.join(args.data_dir, self._args.eval_split_pattern)
        )

        dataset = tf.data.Dataset.from_tensor_slices(eval_files)

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=min(8, multiprocessing.cpu_count()),
            block_length=max(args.batch_size, 32)
        )

        dataset = dataset.map(
            lambda x: dataset_parser(value=x, args=self._args),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.batch(batch_size=args.batch_size, drop_remainder=False)

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

        # x = {
        #     "source_ids": (8, 1),
        #     "images": (8, 832, 1344, 3),
        #     "image_info": (8, 5),
        #     "cropped_gt_masks": (8, 100, 116, 116),
        #     "gt_boxes": (8, 100, 4),
        #     "gt_classes": (8, 100, 1),
        #     "score_targets":_2 (8, 208, 336, 3),
        #     "box_targets_2": (8, 208, 336, 12),
        #     "score_targets_3": (8, 104, 168, 3),
        #     "box_targets_3": (8, 104, 168, 12),
        #     "score_targets_4": (8, 52, 84, 3),
        #     "box_targets_4": (8, 52, 84, 12),
        #     "score_targets_5": (8, 26, 42, 3),
        #     "box_targets_5": (8, 26, 42, 12),
        #     "score_targets_6": (8, 13, 21, 3),
        #     "box_targets_6": (8, 13, 21, 12),
        # }

        # y = {
        #     "width": (8,),
        #     "height": (8,),
        #     "groundtruth_boxes": (8, 100, 4),
        #     "groundtruth_classes": (8, 100, 1),
        #     "num_groundtruth_labels": (8,),
        #     "groundtruth_is_crowd": (8, 100, 1),
        #     "source_ids": (8, 1)
        # }

        return x, y

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        predictions_np = {k: v.numpy() for k, v in predictions.items()}
        expected_np = {k: v.numpy() for k, v in expected.items()}

        return predictions_np, expected_np


    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        # convert from [y1, x1, y2, x2] to [x1, y1, w, h] * scale
        predictions = process_predictions(predictions)

        # create evaluation metric
        eval_metric = EvaluationMetric()

        # eval using the file or groundtruth from features
        images, annotations = extract_coco_groundtruth(expected)
        coco_dataset = create_coco_format_dataset(images, annotations)

        metric = eval_metric.predict_metric_fn(
            predictions, groundtruth_data=coco_dataset
        )
        return metric * 100, 'AP BBOX'


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
