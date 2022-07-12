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
import shutil
import sys

import numpy as np
import ujson as json

import tensorflow as tf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
            default=640,
            help='Size of input images expected by the '
            'model'
        )

        self._parser.add_argument(
            '--annotation_path',
            type=str,
            help='Path that contains COCO annotations'
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

        coco_api = COCO(annotation_file=self._args.annotation_path)
        image_ids = coco_api.getImgIds()

        image_paths = []
        for image_id in image_ids:
            coco_img = coco_api.imgs[image_id]
            image_paths.append(
                os.path.join(self._args.data_dir, coco_img['file_name'])
            )

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        def load_image_op(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)

            return tf.data.Dataset.from_tensor_slices([image])

        dataset = dataset.interleave(
            load_image_op,
            cycle_length=tf.data.experimental.AUTOTUNE,
            block_length=8,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        def preprocess_fn(image):
            if self._args.input_size is not None:
                image = tf.image.resize(
                    image,
                    size=(self._args.input_size, self._args.input_size)
                )
                image = tf.cast(image, tf.uint8)
            return image

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

        return data_batch, np.array([])

    def postprocess_model_outputs(self, predictions, expected):
        """Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        predictions = {k: t.numpy() for k, t in predictions.items()}

        return predictions, expected

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """
        coco_api = COCO(annotation_file=self._args.annotation_path)
        image_ids = coco_api.getImgIds()

        coco_detections = []
        for i, image_id in enumerate(image_ids):
            coco_img = coco_api.imgs[image_id]
            image_width = coco_img['width']
            image_height = coco_img['height']

            for j in range(int(predictions['num_detections'][i])):
                bbox = predictions['boxes'][i][j]
                y1, x1, y2, x2 = list(bbox)
                bbox_coco_fmt = [
                    x1 * image_width,  # x0
                    y1 * image_height,  # x1
                    (x2-x1) * image_width,  # width
                    (y2-y1) * image_height,  # height
                ]
                coco_detection = {
                    'image_id': image_id,
                    'category_id': int(predictions['classes'][i][j]),
                    'bbox': [int(coord) for coord in bbox_coco_fmt],
                    'score': float(predictions['scores'][i][j])
                }
                coco_detections.append(coco_detection)

        # write coco detections to file
        tmp_dir = "/tmp/tmp_detection_results"

        try:
            shutil.rmtree(tmp_dir)
        except FileNotFoundError:
            pass

        os.makedirs(tmp_dir)

        coco_detections_path = os.path.join(tmp_dir, 'coco_detections.json')
        with open(coco_detections_path, 'w') as f:
            json.dump(coco_detections, f)

        cocoDt = coco_api.loadRes(coco_detections_path)

        shutil.rmtree(tmp_dir)

        # compute coco metrics
        eval = COCOeval(coco_api, cocoDt, 'bbox')
        eval.params.imgIds = image_ids

        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        return eval.stats[0] * 100, "mAP %"


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
