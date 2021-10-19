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
import multiprocessing
import time

from functools import partial

import numpy as np
import tensorflow as tf

import preprocessing

# Allow import of top level python files
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner


class CommandLineAPI(BaseCommandLineAPI):

    SAMPLES_IN_VALIDATION_SET = 50000

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument('--input_size', type=int, default=224,
                                  help='Size of input images expected by the '
                                       'model')

        self._parser.add_argument('--num_classes', type=int, default=1001,
                                  help='Number of classes used when training '
                                       'the model')

        self._parser.add_argument('--preprocess_method', type=str,
                                  choices=['vgg', 'inception'], default='vgg',
                                  help='The image preprocessing method used in '
                                       'dataloading.')


class BenchmarkRunner(BaseBenchmarkRunner):

    ACCURACY_METRIC_NAME = "accuracy"

    def before_benchmark(self, **kwargs):
        self._adjust = 1 if kwargs["num_classes"] == 1001 else 0
        self._corrects = 0

        try:
            self._output_tensorname = list(
                self._graph_func.structured_outputs.keys()
            )[0]
        except AttributeError:
            # Output tensor doesn't have a name, index 0
            self._output_tensorname = 0

    def compute_accuracy_metric(self, batch_size, steps_executed, **kwargs):
        return self._corrects / (batch_size * steps_executed)

    def _eval_fn(self, preds, labels, adjust):
        """Measures number of correct predicted labels in a batch.
        Assumes preds and labels are numpy arrays.
        """
        preds = np.argmax(preds, axis=1).reshape(-1) - adjust
        return np.sum((labels.reshape(-1) == preds).astype(np.float32))

    def process_model_output(self, outputs, batch_y, **kwargs):
        self._corrects += self._eval_fn(
            preds=outputs[self._output_tensorname].numpy(),
            labels=batch_y.numpy(),
            adjust=self._adjust
        )


def get_dataset(data_files, batch_size, use_synthetic_data, preprocess_method, input_size):

    def deserialize_image_record(record):
        feature_map = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
            'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
        }
        with tf.compat.v1.name_scope('deserialize_image_record'):
            obj = tf.io.parse_single_example(serialized=record,
                                             features=feature_map)
            imgdata = obj['image/encoded']
            label = tf.cast(obj['image/class/label'], tf.int32)
        return imgdata, label

    def get_preprocess_fn(preprocess_method, input_size):
        """Creates a function to parse and process a TFRecord

        preprocess_method: string
        input_size: int
        returns: function, the preprocessing function for a record
        """
        if preprocess_method == 'vgg':
            preprocess_fn = preprocessing.vgg_preprocess
        elif preprocess_method == 'inception':
            preprocess_fn = preprocessing.inception_preprocess
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

    dataset = tf.data.Dataset.from_tensor_slices(data_files)

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=min(8, multiprocessing.cpu_count()),
        block_length=max(batch_size, 32)
    )

    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(
        preprocess_method=preprocess_method,
        input_size=input_size
    )

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=preprocess_fn,
            batch_size=batch_size,
            num_parallel_calls=min(8, multiprocessing.cpu_count()),
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

    def get_files(data_dir, filename_pattern):
        if data_dir is None:
            return []

        files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))

        if not files:
            raise ValueError('Can not find any files in {} with '
                             'pattern "{}"'.format(data_dir, filename_pattern))
        return files

    data_files = get_files(args.data_dir, 'validation*')

    calib_files = (
        []
        if args.precision != 'INT8' else
        get_files(args.calib_data_dir, 'train*')
    )

    def _input_fn(input_files, build_steps, model_phase):

        dataset = get_dataset(
            data_files=input_files,
            batch_size=args.batch_size,
            # even when using synthetic data, we need to
            # build and/or calibrate using real training data
            # to be in a realistic scenario
            use_synthetic_data=False,
            preprocess_method=args.preprocess_method,
            input_size=args.input_size
        )

        for i, (batch_images, _) in enumerate(dataset):
            if i >= build_steps:
                break

            print("* [%s] - step %04d/%04d" % (
                model_phase, i + 1, build_steps
            ))
            yield batch_images,

    calibration_input_fn = partial(
        _input_fn,
        input_files=calib_files,
        build_steps=args.num_calib_inputs // args.batch_size,
        model_phase="Calibration"
    )

    optimize_offline_input_fn = partial(
        _input_fn,
        input_files=data_files,
        build_steps=1,
        model_phase="Building"
    )

    runner = BenchmarkRunner(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        allow_build_at_runtime=args.allow_build_at_runtime,
        calibration_input_fn=calibration_input_fn,
        gpu_mem_cap=args.gpu_mem_cap,
        input_signature_key=args.input_signature_key,
        max_workspace_size_bytes=args.max_workspace_size,
        minimum_segment_size=args.minimum_segment_size,
        num_calib_inputs=args.num_calib_inputs,
        optimize_offline=args.optimize_offline,
        optimize_offline_input_fn=optimize_offline_input_fn,
        precision_mode=args.precision,
        use_dynamic_shape=args.use_dynamic_shape,
        use_tftrt=args.use_tftrt)

    get_benchmark_input_fn = partial(
        get_dataset,
        data_files=data_files,
        input_size=args.input_size,
        preprocess_method=args.preprocess_method
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
        num_classes=args.num_classes,
    )
