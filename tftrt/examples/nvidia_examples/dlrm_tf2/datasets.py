# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#
# author: Tomasz Grel (tgrel@nvidia.com)

import tensorflow as tf
import os
import json
import numpy as np
from collections import namedtuple

np_type_to_tf_type = {np.int8: tf.int8, np.int16: tf.int16, np.int32: tf.int32}


def get_categorical_feature_type(size):
    types = (np.int8, np.int16, np.int32)
    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(
        f'Categorical feature of size {size} is too large for defined types'
    )


def create_reader(filename, bytes_per_batch):
    fd = os.open(filename, os.O_RDONLY)
    file_len = os.fstat(fd).st_size
    os.close(fd)
    num_batches = int(file_len / bytes_per_batch)
    file_len_patched = num_batches * bytes_per_batch
    footer_bytes = file_len - file_len_patched

    reader = tf.data.FixedLengthRecordDataset(
        filenames=[filename],
        record_bytes=bytes_per_batch,
        footer_bytes=footer_bytes
    )
    return reader, num_batches


DatasetMetadata = namedtuple(
    'DatasetMetadata', ['num_numerical_features', 'categorical_cardinalities']
)


class DummyDataset:

    def __init__(
        self, batch_size, num_numerical_features, num_categorical_features,
        num_batches
    ):
        self.num_batches = num_batches
        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = num_categorical_features
        self.batch_size = batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/GPU:0'):
            factor = tf.random.uniform(
                shape=[1], minval=0, maxval=1, dtype=tf.int32
            )
            if self.num_numerical_features > 0:
                numerical = tf.ones(
                    shape=[self.batch_size, self.num_numerical_features],
                    dtype=tf.float16
                )
                numerical = numerical * tf.cast(factor, tf.float16)
            else:
                numerical = None

            categorical = []
            for _ in range(self.num_categorical_features):
                f = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32)
                f = f * factor
                categorical.append(f)

            labels = tf.ones(shape=[self.batch_size, 1], dtype=tf.int8)
            labels = labels * tf.cast(factor, tf.int8)

            return (numerical, categorical), labels

    def get_next(self):
        return self.__next__()

    def op(self):
        return self

    @staticmethod
    def get_metadata(FLAGS):
        cardinalities = [int(d) for d in FLAGS.synthetic_dataset_cardinalities]
        metadata = DatasetMetadata(
            num_numerical_features=FLAGS.num_numerical_features,
            categorical_cardinalities=cardinalities
        )
        return metadata


class TfRawBinaryDataset:
    """Dataset for reading labels, numerical and categorical features from
    a set of binary files. Internally uses TensorFlow's FixedLengthRecordDataset
    and decode_raw for best performance.

    Args:
        data_path (str): Full path to split binary file of dataset. It must contain numerical.bin, label.bin and
            cat_0 ~ cat_25.bin
        batch_size (int):
        numerical_features(boolean): Number of numerical features to load, default=0 (don't load any)
        categorical_features (list or None): categorical features used by the rank (IDs of the features)
        prefetch_depth (int): How many samples to prefetch. Default 10.
    """

    def __init__(
        self,
        data_path,
        batch_size=1,
        numerical_features=0,
        categorical_features=None,
        prefetch_depth=10
    ):

        self._batch_size = batch_size
        self._data_path = data_path
        self._prefetch_depth = prefetch_depth
        self._numerical_features = numerical_features
        self._categorical_ids = categorical_features

        self._initialize_label_reader()
        self._initialize_numerical_reader()
        self._initialize_categorical_reader()

    @classmethod
    def get_metadata(cls, path, num_numerical_features):
        with open(os.path.join(path, 'model_size.json'), 'r') as f:
            global_table_sizes = json.load(f)

        global_table_sizes = list(global_table_sizes.values())
        global_table_sizes = [s + 1 for s in global_table_sizes]

        metadata = DatasetMetadata(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=global_table_sizes
        )
        return metadata

    def __len__(self):
        return self.num_batches

    def _initialize_label_reader(self):
        bytes_per_sample = np.dtype(np.bool).itemsize
        label_filename = os.path.join(self._data_path, f'label.bin')
        self._label, self.num_batches = create_reader(
            label_filename, bytes_per_sample * self._batch_size
        )

    def _initialize_numerical_reader(self):
        bytes_per_sample = self._numerical_features * np.dtype(
            np.float16
        ).itemsize

        if self._numerical_features > 0:
            num_filename = os.path.join(self._data_path, 'numerical.bin')
            self._numerical, batches = create_reader(
                num_filename, bytes_per_sample * self._batch_size
            )

            if batches != self.num_batches:
                raise ValueError(
                    f'Size mismatch. Expected: {self.num_batches}, got: {batches}'
                )
        else:
            self._numerical = tuple()

    def _load_feature_sizes(self):
        sizes_path = os.path.join(self._data_path, '../model_size.json')
        with open(sizes_path) as f:
            all_table_sizes = json.load(f)

        all_table_sizes = list(all_table_sizes.values())
        all_table_sizes = [s + 1 for s in all_table_sizes]
        self._categorical_sizes = [
            all_table_sizes[cat_id] for cat_id in self._categorical_ids
        ]

    def _initialize_categorical_reader(self):
        self._load_feature_sizes()
        categorical_types = [
            get_categorical_feature_type(size)
            for size in self._categorical_sizes
        ]

        self._categorical = []
        for cat_id, cat_type in zip(self._categorical_ids, categorical_types):
            path = os.path.join(self._data_path, f'cat_{cat_id}.bin')
            bytes_per_sample = np.dtype(cat_type).itemsize
            reader, batches = create_reader(
                path, bytes_per_sample * self._batch_size
            )

            if batches != self.num_batches:
                raise ValueError(
                    f'Size mismatch. Expected: {self.num_batches}, got: {batches}'
                )
            self._categorical.append(reader)

        self._categorical = tuple(self._categorical)

        # memorize for decoding
        self._categorical_types = [
            np_type_to_tf_type[np_type] for np_type in categorical_types
        ]
        self._categorical_types_numpy = categorical_types

    def op(self):
        pipeline = tf.data.Dataset.zip(
            (self._label, self._numerical, self._categorical)
        )
        pipeline = pipeline.map(
            self.decode_batch, num_parallel_calls=tf.data.AUTOTUNE
        )
        pipeline = pipeline.batch(batch_size=1)
        pipeline = pipeline.apply(
            tf.data.experimental.prefetch_to_device(f'/gpu:0')
        )
        pipeline = pipeline.unbatch()
        return pipeline

    @tf.function
    def decode_batch(self, labels, numerical_features, categorical_features):
        #labels, numerical_features, categorical_features = batch
        labels = tf.io.decode_raw(labels, out_type=tf.int8)
        if self._numerical_features > 0:
            numerical_features = tf.io.decode_raw(
                numerical_features, out_type=tf.float16
            )
            numerical_features = tf.reshape(
                numerical_features, shape=[-1, self._numerical_features]
            )

        if self._categorical_ids:
            temp = []
            for dtype, feature in zip(self._categorical_types,
                                      categorical_features):
                feature = tf.io.decode_raw(feature, out_type=dtype)
                feature = tf.cast(feature, dtype=tf.int32)
                feature = tf.expand_dims(feature, axis=1)
                temp.append(feature)
            categorical_features = tf.concat(temp, axis=1)

        return (numerical_features, categorical_features), labels
