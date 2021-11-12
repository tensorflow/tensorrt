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

import logging
import time
import shutil

from functools import partial
import ujson as json

import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Allow import of top level python files
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner


class CommandLineAPI(BaseCommandLineAPI):

    SAMPLES_IN_VALIDATION_SET = 5000

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument('--input_size', type=int, default=640,
                                  help='Size of input images expected by the '
                                       'model')

        self._parser.add_argument('--annotation_path', type=str,
                                  help='Path that contains COCO annotations')


class BenchmarkRunner(BaseBenchmarkRunner):

    ACCURACY_METRIC_NAME = "mAP"

    def before_benchmark(self, **kwargs):
        self._output_name_map = (
            # <tf.Tensor 'detection_boxes:0' shape=(8, None, None) dtype=float32>
            (0, 'boxes'),
            # <tf.Tensor 'detection_classes:0' shape=(8, None) dtype=float32>
            (1, 'classes'),
            # <tf.Tensor 'num_detections:0' shape=(8,) dtype=float32>
            (2, 'num_detections'),
            # <tf.Tensor 'detection_scores:0' shape=(8, None) dtype=float32>
            (3, 'scores'),
        )

    def compute_accuracy_metric(self, predictions, expected, **kwargs):
        return self._eval_model(
            predictions=predictions,
            image_ids=kwargs["image_ids"],
            annotation_path=kwargs["annotation_path"]
        )

    def _eval_model(self, predictions, image_ids, annotation_path):

        # for key in predictions:
        #     predictions[key] = np.vstack(predictions[key])
        #     if key == 'num_detections':
        #         predictions[key] = predictions[key].ravel()

        coco = COCO(annotation_file=annotation_path)
        coco_detections = []
        for i, image_id in enumerate(image_ids):
            coco_img = coco.imgs[image_id]
            image_width = coco_img['width']
            image_height = coco_img['height']

            for j in range(int(predictions['num_detections'][i])):
                bbox = predictions['boxes'][i][j]
                y1, x1, y2, x2 = list(bbox)
                bbox_coco_fmt = [
                    x1 * image_width,  # x0
                    y1 * image_height,  # x1
                    (x2 - x1) * image_width,  # width
                    (y2 - y1) * image_height,  # height
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
        os.makedirs(tmp_dir)

        coco_detections_path = os.path.join(tmp_dir, 'coco_detections.json')
        with open(coco_detections_path, 'w') as f:
            json.dump(coco_detections, f)
        cocoDt = coco.loadRes(coco_detections_path)

        shutil.rmtree(tmp_dir)

        # compute coco metrics
        eval = COCOeval(coco, cocoDt, 'bbox')
        eval.params.imgIds = image_ids

        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        return eval.stats[0]

    def process_model_output(self, outputs, **kwargs):
        # outputs = graph_func(batch_images)
        if isinstance(outputs, dict):
            outputs = {k:t.numpy() for k, t in outputs.items()}
        else:
            outputs = {
                name: outputs[idx].numpy()
                for idx, name in self._output_name_map
            }

        return outputs


def get_dataset(batch_size,
                images_dir,
                image_ids,
                input_size,
                use_synthetic_data):

    image_paths = []

    for image_id in image_ids:
        coco_img = coco.imgs[image_id]
        image_paths.append(os.path.join(images_dir, coco_img['file_name']))

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_image_op(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)

        return tf.data.Dataset.from_tensor_slices([image])

    dataset = dataset.interleave(
        lambda path: load_image_op(path),
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    def preprocess_fn(image):
        if input_size is not None:
            image = tf.image.resize(image, size=(input_size, input_size))
            image = tf.cast(image, tf.uint8)
        return image

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=preprocess_fn,
            batch_size=batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True
        )
    )

    if use_synthetic_data:
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    coco = COCO(annotation_file=args.annotation_path)
    image_ids = coco.getImgIds()

    def _input_fn(input_data_dir, build_steps, model_phase):

        dataset = get_dataset(
            batch_size=args.batch_size,
            images_dir=input_data_dir,
            image_ids=image_ids,
            input_size=args.input_size,
            # even when using synthetic data, we need to
            # build and/or calibrate using real training data
            # to be in a realistic scenario
            use_synthetic_data=False,
        )

        for i, batch_images in enumerate(dataset):
            if i >= build_steps:
                break

            print("* [%s] - step %04d/%04d" % (
                model_phase, i + 1, build_steps
            ))
            yield batch_images,

    calibration_input_fn = partial(
        _input_fn,
        input_data_dir=args.calib_data_dir,
        build_steps=args.num_calib_inputs // args.batch_size,
        model_phase="Calibration"
    )

    optimize_offline_input_fn = partial(
        _input_fn,
        input_data_dir=args.data_dir,
        build_steps=1,
        model_phase="Building"
    )

    runner = BenchmarkRunner(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        allow_build_at_runtime=args.allow_build_at_runtime,
        calibration_input_fn=calibration_input_fn,
        debug=args.debug,
        gpu_mem_cap=args.gpu_mem_cap,
        input_signature_key=args.input_signature_key,
        max_workspace_size_bytes=args.max_workspace_size,
        minimum_segment_size=args.minimum_segment_size,
        num_calib_inputs=args.num_calib_inputs,
        optimize_offline=args.optimize_offline,
        optimize_offline_input_fn=optimize_offline_input_fn,
        output_tensor_indices=args.output_tensor_indices,
        output_tensor_names=args.output_tensor_names,
        precision_mode=args.precision,
        use_dynamic_shape=args.use_dynamic_shape,
        use_tftrt=args.use_tftrt)

    get_benchmark_input_fn = partial(
        get_dataset,
        images_dir=args.data_dir,
        image_ids=image_ids,
        input_size=args.input_size
    )

    runner.execute_benchmark(
        batch_size=args.batch_size,
        display_every=args.display_every,
        get_benchmark_input_fn=get_benchmark_input_fn,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        skip_accuracy_testing=(
            args.use_synthetic_data or args.skip_accuracy_testing
        ),
        use_synthetic_data=args.use_synthetic_data,
        use_xla=args.use_xla,
        ########### Additional Settings ############
        image_ids=image_ids,
        annotation_path=args.annotation_path
    )
