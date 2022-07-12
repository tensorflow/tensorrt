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
            "--input_size",
            type=int,
            default=None,
            required=True,
            help="Size of input images expected by the model"
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

        def decode_and_center_crop(
            image_bytes: tf.Tensor, image_size, crop_padding=32
        ) -> tf.Tensor:
            """Crops to center of image with padding then scales image_size.

            Args:
            image_bytes: `Tensor` representing an image binary of arbitrary size.
            image_size: image height/width dimension.
            crop_padding: the padding size to use when centering the crop.

            Returns:
            A decoded and cropped image `Tensor`.
            """
            decoded = image_bytes.dtype != tf.string
            shape = (
                tf.shape(image_bytes)
                if decoded else tf.image.extract_jpeg_shape(image_bytes)
            )
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(
                ((image_size / (image_size+crop_padding)) *
                tf.cast(tf.minimum(image_height, image_width), tf.float32)),
                tf.int32
            )

            offset_height = ((image_height-padded_center_crop_size) + 1) // 2
            offset_width = ((image_width-padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([
                offset_height, offset_width, padded_center_crop_size,
                padded_center_crop_size
            ])
            if decoded:
                image = tf.image.crop_to_bounding_box(
                    image_bytes,
                    offset_height=offset_height,
                    offset_width=offset_width,
                    target_height=padded_center_crop_size,
                    target_width=padded_center_crop_size
                )
            else:
                image = tf.image.decode_and_crop_jpeg(
                    image_bytes, crop_window, channels=3
                )

            image = tf.compat.v1.image.resize(
                image, [image_size, image_size],
                method=tf.image.ResizeMethod.BILINEAR,
                align_corners=False
            )

            return image

        def preprocess(
            image_bytes,
            label,
            image_size,
            num_channels=3,
            num_classes=1000,
            dtype=tf.float32,
        ) -> tf.Tensor:
            """Preprocesses the given image for evaluation.

            Args:
            image_bytes: `Tensor` representing an image binary of arbitrary size.
            image_size: image height/width dimension.
            num_channels: number of image input channels.
            mean_subtract: whether or not to apply mean subtraction.
            standardize: whether or not to apply standardization.
            dtype: the dtype to convert the images to. Set to `None` to skip conversion.

            Returns:
            A preprocessed and normalized image `Tensor`.
            """
            images = decode_and_center_crop(
                image_bytes, image_size=image_size, crop_padding=32
            )
            images = tf.reshape(images, [image_size, image_size, num_channels])

            if dtype is not None:
                images = tf.image.convert_image_dtype(images, dtype=dtype)
            label = tf.one_hot(label, num_classes)
            label = tf.reshape(label, [num_classes])
            return images, label

        def parse_record(record, image_size):
            """Parse an ImageNet record from a serialized string Tensor."""
            keys_to_features = {
                'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
                'image/format': tf.io.FixedLenFeature((), tf.string, 'jpeg'),
                'image/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
                'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64),
            }

            parsed = tf.io.parse_single_example(record, keys_to_features)

            label = tf.reshape(parsed['image/class/label'], shape=[1])
            label = tf.cast(label, dtype=tf.int32)

            # Subtract one so that labels are in [0, 1000)
            label -= 1

            image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
            image, label = preprocess(image_bytes, label, image_size)
            # populate features and labels dict
            features = dict()
            features['image'] = image
            features['is_tr_split'] = [False]
            features['cutmix_mask'] = tf.zeros((image_size, image_size, 1))
            features['mixup_weight'] = tf.ones((1, 1, 1))
            return features, label

        file_pattern = os.path.join(self._args.data_dir, 'validation*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        parse_record_fn = lambda record: parse_record(
            record=record, image_size=self._args.input_size
        )
        dataset = dataset.map(
            parse_record_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
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

        return predictions.numpy(), expected.numpy()

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """

        return (
            np.mean(
                np.argmax(
                    predictions["data"], 1) == np.argmax(expected["data"],
                    axis=1
                )
            ) * 100.0,
            "Top-1 Accuracy %"
        )



if __name__ == '__main__':

    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()
