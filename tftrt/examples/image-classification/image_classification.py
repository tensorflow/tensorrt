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

import argparse
import os
import logging
import time
import pprint
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
import preprocessing

def deserialize_image_record(record):
  feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                 'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                 'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                 'image/object/bbox/xmin': tf.io.VarLenFeature(
                     dtype=tf.float32),
                 'image/object/bbox/ymin': tf.io.VarLenFeature(
                     dtype=tf.float32),
                 'image/object/bbox/xmax': tf.io.VarLenFeature(
                     dtype=tf.float32),
                 'image/object/bbox/ymax': tf.io.VarLenFeature(
                     dtype=tf.float32)}
  with tf.compat.v1.name_scope('deserialize_image_record'):
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)
    return imgdata, label

def get_preprocess_fn(preprocess_method, input_size, mode='validation'):
  """Creates a function to parse and process a TFRecord

  preprocess_method: string
  input_size: int
  mode: string, which mode to use (validation or benchmark)
  returns: function, the preprocessing function for a record
  """
  if preprocess_method == 'vgg':
    preprocess_fn = preprocessing.vgg_preprocess
  elif preprocess_method == 'inception':
    preprocess_fn = preprocessing.inception_preprocess
  else:
    raise ValueError(
        'Invalid preprocessing method {}'.format(preprocess_method))

  def validation_process(record):
    # Parse TFRecord
    imgdata, label = deserialize_image_record(record)
    label -= 1 # Change to 0-based (don't use background class)
    try:
      image = tf.image.decode_jpeg(
          imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
    except:
      image = tf.image.decode_png(imgdata, channels=3)
    # Use model's preprocessing function
    image = preprocess_fn(image, input_size, input_size)
    return image, label

  def benchmark_process(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_fn(image, input_size, input_size)
    return image

  if mode == 'validation':
    return validation_process
  if mode == 'benchmark':
    return benchmark_process
  raise ValueError("Mode must be either 'validation' or 'benchmark'")


def get_dataset(data_files,
                batch_size,
                use_synthetic,
                preprocess_method,
                input_size,
                mode='validation'):
  if use_synthetic:
    features = np.random.normal(
        loc=112, scale=70,
        size=(batch_size, input_size, input_size, 3)).astype(np.float32)
    features = np.clip(features, 0.0, 255.0)
    features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
        "features", dtype=tf.float32, initializer=tf.constant(features)))
    dataset = tf.data.Dataset.from_tensor_slices([features])
    dataset = dataset.repeat()
  else:
    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(
        preprocess_method=preprocess_method,
        input_size=input_size,
        mode=mode)
    if mode == 'validation':
      dataset = tf.data.TFRecordDataset(data_files)
      dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
      dataset = dataset.batch(batch_size=batch_size)
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      dataset = dataset.repeat(count=1)
    elif mode == 'benchmark':
      dataset = tf.data.Dataset.from_tensor_slices(data_files)
      dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
      dataset = dataset.batch(batch_size=batch_size)
      dataset = dataset.repeat(count=1)
    else:
      raise ValueError("Mode must be either 'validation' or 'benchmark'")
  return dataset


def get_func_from_saved_model(saved_model_dir):
  saved_model_loaded = tf.saved_model.load(
      saved_model_dir, tags=[tag_constants.SERVING])
  graph_func = saved_model_loaded.signatures[
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
  return graph_func


def get_graph_func(input_saved_model_dir,
                   preprocess_method,
                   input_size,
                   output_saved_model_dir=None,
                   conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
                   use_trt=False,
                   calib_files=None,
                   num_calib_inputs=None,
                   use_synthetic=False,
                   batch_size=None,
                   optimize_offline=False):
  """Retreives a frozen SavedModel and applies TF-TRT
  use_trt: bool, if true use TensorRT
  precision: str, floating point precision (FP32, FP16, or INT8)
  batch_size: int, batch size for TensorRT optimizations
  returns: TF function that is ready to run for inference
  """
  start_time = time.time()
  graph_func = get_func_from_saved_model(input_saved_model_dir)
  if use_trt:
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params,
    )
    def input_fn(input_files, num_iterations):
      dataset = get_dataset(data_files=input_files,
                            batch_size=batch_size,
                            use_synthetic=False,
                            preprocess_method=preprocess_method,
                            input_size=input_size,
                            mode='validation')
      for i, (batch_images, _) in enumerate(dataset):
        if i >= num_iterations:
          break
        yield (batch_images,)
        print("  step %d/%d" % (i+1, num_iterations))
        i += 1
    if conversion_params.precision_mode != 'INT8':
      print('Graph convertion...')
      converter.convert()
      if optimize_offline:
        print('Building TensorRT engines...')
        converter.build(input_fn=partial(input_fn, data_files, 1))
      converter.save(output_saved_model_dir=output_saved_model_dir)
      graph_func = get_func_from_saved_model(output_saved_model_dir)
    else:
      print('Graph convertion and INT8 calibration...')
      converter.convert(calibration_input_fn=partial(
          input_fn, calib_files, num_calib_inputs//batch_size))
      if optimize_offline:
        print('Building TensorRT engines...')
        converter.build(input_fn=partial(input_fn, data_files, 1))
      converter.save(output_saved_model_dir=output_saved_model_dir)
      graph_func = get_func_from_saved_model(output_saved_model_dir)
  return graph_func, {'conversion': time.time() - start_time}

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
                  use_synthetic,
                  display_every=100,
                  mode='validation',
                  target_duration=None):
  """Run the given graph_func on the data files provided. In validation mode,
  it consumes TFRecords with labels and reports accuracy. In benchmark mode, it
  times inference on real data (.jpgs).
  """
  results = {}
  corrects = 0
  iter_times = []
  adjust = 1 if num_classes == 1001 else 0
  initial_time = time.time()
  dataset = get_dataset(data_files=data_files,
                        batch_size=batch_size,
                        use_synthetic=use_synthetic,
                        input_size=input_size,
                        preprocess_method=preprocess_method,
                        mode=mode)

  if mode == 'validation':
    for i, (batch_images, batch_labels) in enumerate(dataset):
      start_time = time.time()
      batch_preds = graph_func(batch_images)[0].numpy()
      end_time = time.time()
      iter_times.append(end_time - start_time)
      if i % display_every == 0:
        print("  step %d/%d, iter_time(ms)=%.0f" %
              (i+1, 50000//batch_size, iter_times[-1]*1000))
      corrects += eval_fn(
          batch_preds, batch_labels.numpy(), adjust)
      if i > 1 and target_duration is not None and \
        time.time() - initial_time > target_duration:
        break
    accuracy = corrects / (batch_size * i)
    results['accuracy'] = accuracy

  elif mode == 'benchmark':
    for i, batch_images in enumerate(dataset):
      if i >= num_warmup_iterations:
        start_time = time.time()
        batch_preds = list(graph_func(batch_images).values())[0].numpy()
        iter_times.append(time.time() - start_time)
        if i % display_every == 0:
          print("  step %d/%d, iter_time(ms)=%.0f" %
                (i+1, num_iterations, iter_times[-1]*1000))
      else:
        batch_preds = list(graph_func(batch_images).values())[0].numpy()
      if i > 0 and target_duration is not None and \
        time.time() - initial_time > target_duration:
        break
      if num_iterations is not None and i >= num_iterations:
        break

  if not iter_times:
    return results
  iter_times = np.array(iter_times)
  iter_times = iter_times[num_warmup_iterations:]
  results['total_time'] = np.sum(iter_times)
  results['images_per_sec'] = np.mean(batch_size / iter_times)
  results['99th_percentile'] = np.percentile(
      iter_times, q=99, interpolation='lower') * 1000
  results['latency_mean'] = np.mean(iter_times) * 1000
  results['latency_median'] = np.median(iter_times) * 1000
  results['latency_min'] = np.min(iter_times) * 1000
  return results


def get_trt_conversion_params(max_workspace_size_bytes,
                              precision_mode,
                              minimum_segment_size,
                              max_batch_size):
  conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
  conversion_params = conversion_params._replace(
      max_workspace_size_bytes=max_workspace_size_bytes)
  conversion_params = conversion_params._replace(precision_mode=precision_mode)
  conversion_params = conversion_params._replace(
      minimum_segment_size=minimum_segment_size)
  conversion_params = conversion_params._replace(
      use_calibration=precision_mode == 'INT8')
  conversion_params = conversion_params._replace(
      max_batch_size=max_batch_size)
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
  parser.add_argument('--num_iterations', type=int, default=2048,
                      help='How many iterations(batches) to evaluate.'
                      'If not supplied, the whole set will be evaluated.')
  parser.add_argument('--display_every', type=int, default=100,
                      help='Number of iterations executed between'
                      'two consecutive display of metrics')
  parser.add_argument('--use_synthetic', action='store_true',
                      help='If set, one batch of random data is'
                      'generated and used at every iteration.')
  parser.add_argument('--num_warmup_iterations', type=int, default=50,
                      help='Number of initial iterations skipped from timing')
  parser.add_argument('--num_calib_inputs', type=int, default=500,
                      help='Number of inputs (e.g. images) used for'
                      'calibration (last batch is skipped in case'
                      'it is not full)')
  parser.add_argument('--max_workspace_size', type=int, default=(1<<30),
                      help='workspace size in bytes')
  parser.add_argument('--mode', choices=['validation', 'benchmark'],
                      default='validation',
                      help='Which mode to use (validation or benchmark)')
  parser.add_argument('--target_duration', type=int, default=None,
                      help='If set, script will run for specified'
                      'number of seconds.')
  args = parser.parse_args()

  if args.precision != 'FP32' and not args.use_trt:
    raise ValueError('TensorRT must be enabled for FP16'
                     'or INT8 modes (--use_trt).')
  if (args.precision == 'INT8' and not args.calib_data_dir
      and not args.use_synthetic):
    raise ValueError('--calib_data_dir is required for INT8 mode')
  if (args.num_iterations is not None
      and args.num_iterations <= args.num_warmup_iterations):
    raise ValueError(
        '--num_iterations must be larger than --num_warmup_iterations '
        '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))
  if args.num_calib_inputs < args.batch_size:
    raise ValueError(
        '--num_calib_inputs must not be smaller than --batch_size'
        '({} <= {})'.format(args.num_calib_inputs, args.batch_size))
  if args.mode == 'validation' and args.use_synthetic:
    raise ValueError('Cannot use both validation mode and synthetic dataset')
  if args.data_dir is None and not args.use_synthetic:
    raise ValueError("--data_dir required if you are not using synthetic data")
  if args.use_synthetic and args.num_iterations is None:
    raise ValueError("--num_iterations is required for --use_synthetic")
  if args.use_trt and not args.output_saved_model_dir:
    raise ValueError("--output_saved_model_dir must be set if use_trt=True")

  calib_files = []
  data_files = []
  def get_files(data_dir, filename_pattern):
    if data_dir is None:
      return []
    files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))
    if files == []:
      raise ValueError('Can not find any files in {} with '
                       'pattern "{}"'.format(data_dir, filename_pattern))
    return files
  if not args.use_synthetic:
    if args.mode == "validation":
      data_files = get_files(args.data_dir, 'validation*')
    elif args.mode == "benchmark":
      data_files = [os.path.join(path, name) for path, _, files
                    in os.walk(args.data_dir) for name in files]
    else:
      raise ValueError("Mode must be either 'validation' or 'benchamark'")
    if args.precision == 'INT8':
      calib_files = get_files(args.calib_data_dir, 'train*')

  params = get_trt_conversion_params(
      args.max_workspace_size,
      args.precision,
      args.minimum_segment_size,
      args.batch_size,)
  graph_func, times = get_graph_func(
      input_saved_model_dir=args.input_saved_model_dir,
      output_saved_model_dir=args.output_saved_model_dir,
      preprocess_method=args.preprocess_method,
      input_size=args.input_size,
      conversion_params=params,
      use_trt=args.use_trt,
      calib_files=calib_files,
      batch_size=args.batch_size,
      num_calib_inputs=args.num_calib_inputs,
      use_synthetic=args.use_synthetic,
      optimize_offline=args.optimize_offline)

  def print_dict(input_dict, prefix='  ', postfix=''):
    for k, v in sorted(input_dict.items()):
      print('{}{}: {}{}'.format(prefix, k, '%.1f'%v if isinstance(v, float) else v, postfix))
  print('Benchmark arguments:')
  print_dict(vars(args))
  print('TensorRT Conversion Params:')
  print_dict(dict(params._asdict()))
  print('Conversion times:')
  print_dict(times, postfix='s')

  results = run_inference(graph_func,
                data_files=data_files,
                batch_size=args.batch_size,
                num_iterations=args.num_iterations,
                num_warmup_iterations=args.num_warmup_iterations,
                preprocess_method=args.preprocess_method,
                input_size=args.input_size,
                num_classes=args.num_classes,
                use_synthetic=args.use_synthetic,
                display_every=args.display_every,
                mode=args.mode,
                target_duration=args.target_duration)
  if args.mode == 'validation':
    print('  accuracy: %.2f' % (results['accuracy'] * 100))
  print('  images/sec: %d' % results['images_per_sec'])
  print('  99th_percentile(ms): %.2f' % results['99th_percentile'])
  print('  total_time(s): %.1f' % results['total_time'])
  print('  latency_mean(ms): %.2f' % results['latency_mean'])
  print('  latency_median(ms): %.2f' % results['latency_median'])
  print('  latency_min(ms): %.2f' % results['latency_min'])
