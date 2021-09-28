# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import copy
import logging
import multiprocessing
import time

from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

import preprocessing

logging.disable(logging.WARNING)

SAMPLES_IN_VALIDATION_SET = 50000


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


def get_dataset(data_files,
                batch_size,
                use_synthetic_data,
                preprocess_method,
                input_size):

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


def get_func_from_saved_model(saved_model_dir, input_signature_key=None):

    if input_signature_key is None:
        input_signature_key = \
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])

    _graph_func = saved_model_loaded.signatures[input_signature_key]
    _graph_func = convert_to_constants.convert_variables_to_constants_v2(
        _graph_func
    )
    return _graph_func


def get_graph_func(input_saved_model_dir,
                   preprocess_method,
                   input_size,
                   output_saved_model_dir,
                   conversion_params=None,
                   use_trt=False,
                   calib_files=None,
                   num_calib_inputs=None,
                   use_synthetic_data=False,
                   batch_size=None,
                   optimize_offline=False,
                   use_dynamic_shape=False,
                   input_signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                   ):
    """Retreives a frozen SavedModel and applies TF-TRT
    use_trt: bool, if true use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: TF function that is ready to run for inference
    """
    start_time = time.time()
    graph_func = get_func_from_saved_model(
        input_saved_model_dir, input_signature_key
    )

    if use_trt:

        if conversion_params is None:
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params,
            input_saved_model_signature_key=input_signature_key,
            use_dynamic_shape=use_dynamic_shape
        )

        def input_fn(input_files, build_steps, model_phase):

            dataset = get_dataset(
                data_files=input_files,
                batch_size=batch_size,
                # even when using synthetic data, we need to
                # build and/or calibrate using real training data
                # to be in a realistic scenario
                use_synthetic_data=False,
                preprocess_method=preprocess_method,
                input_size=input_size
            )

            for i, (batch_images, _) in enumerate(dataset):
                if i >= build_steps:
                    break

                print("* [%s] - step %04d/%04d" % (
                    model_phase, i + 1, build_steps
                ))
                yield batch_images,

        if conversion_params.precision_mode == 'INT8':
            print('Graph conversion and INT8 calibration...')
            graph_func = converter.convert(
                calibration_input_fn=partial(
                    input_fn,
                    input_files=calib_files,
                    build_steps=num_calib_inputs // batch_size,
                    model_phase="Calibration"
                )
            )

        else:
            print('Graph conversion...')
            graph_func = converter.convert()

        if optimize_offline or use_dynamic_shape:
            print('Building TensorRT engines...')
            converter.build(
                input_fn=partial(input_fn, data_files, 1, "Building")
            )

        if output_saved_model_dir is not None:
            converter.save(output_saved_model_dir=output_saved_model_dir)

    return graph_func, time.time() - start_time


def eval_fn(preds, labels, adjust):
    """Measures number of correct predicted labels in a batch.
    Assumes preds and labels are numpy arrays.
    """
    preds = np.argmax(preds, axis=1).reshape(-1) - adjust
    return np.sum((labels.reshape(-1) == preds).astype(np.float32))


def run_inference(graph_func,
                  data_files,
                  batch_size,
                  preprocess_method,
                  input_size,
                  num_classes,
                  num_iterations,
                  num_warmup_iterations,
                  use_synthetic_data,
                  display_every=100,
                  skip_accuracy_testing=False):
    """Run the given graph_func on the data files provided.
    It consumes TFRecords with labels and reports accuracy.
    """
    results = {}
    corrects = 0
    iter_times = []
    adjust = 1 if num_classes == 1001 else 0

    dataset = get_dataset(
        data_files=data_files,
        batch_size=batch_size,
        use_synthetic_data=use_synthetic_data,
        input_size=input_size,
        preprocess_method=preprocess_method
    )

    steps_executed = 0
    total_steps = (
        SAMPLES_IN_VALIDATION_SET // batch_size
        if num_iterations is None else
        num_iterations
    )

    try:
        output_tensorname = list(graph_func.structured_outputs.keys())[0]
    except AttributeError:
        # Output tensor doesn't have a name, index 0
        output_tensorname = 0

    @tf.function
    def infer_step(batch_x):
      return graph_func(batch_x)[output_tensorname]

    print("\nStart inference ...")
    for i, (batch_images, batch_labels) in enumerate(dataset):

        start_time = time.time()
        batch_preds = infer_step(batch_images).numpy()
        iter_times.append(time.time() - start_time)

        steps_executed += 1

        if (i + 1) % display_every == 0 or (i + 1) == total_steps:
            print("  step %04d/%04d, iter_time(ms)=%.0f" % (
                i + 1,
                total_steps,
                np.mean(iter_times[-display_every:]) * 1000
            ))

        if not skip_accuracy_testing:
            corrects += eval_fn(
                preds=batch_preds,
                labels=batch_labels.numpy(),
                adjust=adjust
            )

        if (i + 1) >= total_steps:
            break

    if not skip_accuracy_testing:
        accuracy = corrects / (batch_size * steps_executed)
        results['accuracy'] = accuracy

    iter_times = np.array(iter_times)
    run_times = iter_times[num_warmup_iterations:]

    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / run_times)
    results['99th_percentile'] = np.percentile(
        run_times, q=99, interpolation='lower'
    ) * 1000
    results['latency_mean'] = np.mean(run_times) * 1000
    results['latency_median'] = np.median(run_times) * 1000
    results['latency_min'] = np.min(run_times) * 1000
    results['latency_max'] = np.max(run_times) * 1000

    return results


def config_gpu_memory(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if not gpus:
        raise RuntimeError("No GPUs has been found.")

    print('Found the following GPUs:')
    for gpu in gpus:
        print(' ', gpu)

    for gpu in gpus:
        try:
            if not gpu_mem_cap:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpu_mem_cap)])
        except RuntimeError as e:
            print('Can not set GPU memory config', e)


def get_trt_conversion_params(max_workspace_size_bytes,
                              precision_mode,
                              minimum_segment_size):

    conversion_params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    conversion_params = conversion_params._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=max_workspace_size_bytes,
        minimum_segment_size=minimum_segment_size,
        use_calibration=precision_mode == 'INT8'
    )
    return conversion_params


if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--input_saved_model_dir', type=str, default=None,
                        help='Directory containing the input saved model.')
    parser.add_argument('--output_saved_model_dir', type=str, default=None,
                        help='Directory in which the converted model is saved')
    parser.add_argument('--preprocess_method', type=str,
                        choices=['vgg', 'inception'], default='vgg',
                        help='The image preprocessing method')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Size of input images expected by the model')
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='Number of classes used when training the model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing validation set'
                             'TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
                        help='Directory containing TFRecord files for'
                             'calibrating INT8.')
    parser.add_argument('--use_trt', action='store_true',
                        help='If set, the graph will be converted to a'
                             'TensorRT graph.')
    parser.add_argument('--optimize_offline', action='store_true',
                        help='If set, TensorRT engines are built'
                             'before runtime.')
    parser.add_argument('--precision', type=str,
                        choices=['FP32', 'FP16', 'INT8'], default='FP32',
                        help='Precision mode to use. FP16 and INT8 only'
                             'work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images per batch.')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
                        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--num_iterations', type=int, default=None,
                        help='How many iterations(batches) to evaluate. If not '
                             'supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
                        help='Number of iterations executed between'
                             'two consecutive display of metrics')
    parser.add_argument('--use_dynamic_shape', action='store_true', help="Enable"
                      "dynamic shape mode")
    parser.add_argument('--use_synthetic_data', action='store_true',
                        help='If set, one batch of random data is'
                             'generated and used at every iteration.')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
                        help='Number of initial iterations skipped from timing')
    parser.add_argument('--num_calib_inputs', type=int, default=500,
                        help='Number of inputs (e.g. images) used for'
                             'calibration (last batch is skipped in case'
                             'it is not full)')
    parser.add_argument('--gpu_mem_cap', type=int, default=0,
                        help='Upper bound for GPU memory in MB. Default is 0 '
                             'which means allow_growth will be used.')
    parser.add_argument('--max_workspace_size', type=int, default=(1 << 30),
                        help='workspace size in bytes')
    parser.add_argument('--input_signature_key', type=str,
                        default=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                        help='Directory containing TFRecord files for')
    parser.add_argument('--skip_accuracy_testing', action='store_true',
                        help='If set, accuracy calculation will be skipped.')
    args = parser.parse_args()

    if args.use_dynamic_shape and not args.use_trt:
        raise ValueError('TensorRT must be enabled for Dynamic Shape support '
                         'to be enabled (--use_trt).')

    if args.use_dynamic_shape and args.precision == 'INT8':
        raise ValueError('TF-TRT does not support dynamic shape mode with INT8 '
                         'calibration.')

    if args.precision != 'FP32' and not args.use_trt:
        raise ValueError('TensorRT must be enabled for FP16'
                         'or INT8 modes (--use_trt).')

    if args.precision == 'INT8' and not args.calib_data_dir:
        raise ValueError('--calib_data_dir is required for INT8 mode')

    if args.use_synthetic_data:
        args.skip_accuracy_testing = True

        if args.num_iterations is None:
            args.num_iterations = SAMPLES_IN_VALIDATION_SET // args.batch_size

    if (
        args.num_iterations is not None and
        args.num_iterations <= args.num_warmup_iterations
    ):
        raise ValueError(
            '--num_iterations must be larger than --num_warmup_iterations '
            '({} <= {})'.format(args.num_iterations,
                                args.num_warmup_iterations))

    if args.precision == 'INT8' and args.num_calib_inputs <= args.batch_size:
        raise ValueError(
            '--num_calib_inputs must not be smaller than --batch_size'
            '({} <= {})'.format(args.num_calib_inputs, args.batch_size))

    if args.data_dir is None:
        raise ValueError("--data_dir is required")

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

    config_gpu_memory(args.gpu_mem_cap)

    params = get_trt_conversion_params(
        args.max_workspace_size,
        args.precision,
        args.minimum_segment_size
    )

    graph_func, convert_time = get_graph_func(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        preprocess_method=args.preprocess_method,
        input_size=args.input_size,
        conversion_params=params,
        use_trt=args.use_trt,
        calib_files=calib_files,
        batch_size=args.batch_size,
        num_calib_inputs=args.num_calib_inputs,
        use_synthetic_data=args.use_synthetic_data,
        optimize_offline=args.optimize_offline,
        use_dynamic_shape=args.use_dynamic_shape,
        input_signature_key=args.input_signature_key)

    def print_dict(input_dict, prefix='  ', postfix=''):
        for k, v in sorted(input_dict.items()):
            print('{prefix}{arg_name}: {value}{postfix}'.format(
                prefix=prefix,
                arg_name=k,
                value='%.1f' % v if isinstance(v, float) else v,
                postfix=postfix
            ))

    print('\nBenchmark arguments:')
    print_dict(vars(args))
    print('\nTensorRT Conversion Params:')
    print_dict(dict(params._asdict()))
    print('\nConversion time: %.1f' % convert_time)

    results = run_inference(
        graph_func,
        data_files=data_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        preprocess_method=args.preprocess_method,
        input_size=args.input_size,
        num_classes=args.num_classes,
        use_synthetic_data=args.use_synthetic_data,
        display_every=args.display_every,
        skip_accuracy_testing=args.skip_accuracy_testing
    )

    print('\n=============================================\n')
    print('Results:\n')

    if "accuracy" in results:
        print('  accuracy: %.2f' % (results['accuracy'] * 100))
    print('  images/sec: %d' % results['images_per_sec'])
    print('  99th_percentile(ms): %.2f' % results['99th_percentile'])
    print('  total_time(s): %.1f' % results['total_time'])
    print('  latency_mean(ms): %.2f' % results['latency_mean'])
    print('  latency_median(ms): %.2f' % results['latency_median'])
    print('  latency_min(ms): %.2f' % results['latency_min'])
    print('  latency_max(ms): %.2f' % results['latency_max'])
