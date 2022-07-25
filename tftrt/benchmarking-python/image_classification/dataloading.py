# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf

import preprocessing

__all__ = ["get_dataloader"]


def get_dataloader(args):

    def get_files(data_dir, filename_pattern):
        if data_dir is None:
            return []

        files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))

        if not files:
            raise ValueError(
                f'Can not find any files in {data_dir} with '
                f'pattern "{filename_pattern}"'
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
        elif preprocess_method == 'resnet50_v1_5_tf1_ngc':
            preprocess_fn = preprocessing.resnet50_v1_5_tf1_ngc_preprocess
            preprocess_fn = preprocessing.inception_preprocess
        elif preprocess_method == 'vision_transformer':
            preprocess_fn = preprocessing.vision_transformer_preprocess
        elif preprocess_method == 'swin_transformer':
            preprocess_fn = preprocessing.swin_transformer_preprocess
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

    data_files = get_files(args.data_dir, 'validation*')
    dataset = tf.data.Dataset.from_tensor_slices(data_files)

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        block_length=max(args.batch_size, 32)
    )

    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(
        preprocess_method=args.preprocess_method, input_size=args.input_size
    )

    dataset = dataset.map(
        map_func=preprocess_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(args.batch_size, drop_remainder=False)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
