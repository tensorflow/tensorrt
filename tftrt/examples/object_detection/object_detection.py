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
from functools import partial
import json
import numpy as np
import subprocess
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.disable(logging.WARNING)


def get_dataset(images_dir,
                annotation_path,
                batch_size,
                use_synthetic,
                input_size):

    coco = COCO(annotation_file=annotation_path)
    image_ids = coco.getImgIds()
    image_paths = []

    num_steps = len(image_ids) // batch_size

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

    if use_synthetic:
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat(count=num_steps)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, image_ids, num_steps


def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func


def get_graph_func(input_saved_model_dir,
                   data_dir,
                   calib_data_dir,
                   annotation_path,
                   input_size,
                   output_saved_model_dir=None,
                   conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
                   use_trt=False,
                   num_calib_inputs=None,
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

        def input_fn(input_data_dir, num_iterations, model_phase):
            dataset, image_ids, _ = get_dataset(
                images_dir=input_data_dir,
                annotation_path=annotation_path,
                batch_size=batch_size,
                input_size=input_size,
                use_synthetic=False
            )

            for i, batch_images in enumerate(dataset):
                if i >= num_iterations:
                    break
                yield (batch_images,)
                print("* [%s] - step %02d/%02d" % (
                model_phase, i + 1, num_iterations))
                i += 1

        if conversion_params.precision_mode != 'INT8':
            print('Graph conversion...')
            converter.convert()
            if optimize_offline:
                print('Building TensorRT engines...')
                converter.build(
                    input_fn=partial(input_fn, data_dir, 1, "Building"))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
        else:
            print('Graph conversion and INT8 calibration...')
            converter.convert(calibration_input_fn=partial(
                input_fn, calib_data_dir, num_calib_inputs // batch_size,
                "Calibration"))
            if optimize_offline:
                print('Building TensorRT engines...')
                converter.build(
                    input_fn=partial(input_fn, data_dir, 1, "Building"))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
    return graph_func, {'conversion': time.time() - start_time}


def run_inference(graph_func,
                  data_dir,
                  annotation_path,
                  batch_size,
                  input_size,
                  num_iterations,
                  num_warmup_iterations,
                  use_synthetic,
                  display_every=100,
                  target_duration=None):
    """Run the given graph_func on the data files provided. It consumes
    TFRecords with labels and reports accuracy.
    """
    results = {}
    predictions = {}
    iter_times = []
    initial_time = time.time()

    dataset, image_ids, num_steps = get_dataset(
        images_dir=data_dir,
        annotation_path=annotation_path,
        batch_size=batch_size,
        use_synthetic=use_synthetic,
        input_size=input_size
    )

    for i, batch_images in enumerate(dataset):
        start_time = time.time()
        batch_preds = graph_func(batch_images)
        iter_times.append(time.time() - start_time)

        for key in batch_preds.keys():
            if key not in predictions:
                predictions[key] = [batch_preds[key]]
            else:
                predictions[key].append(batch_preds[key])

        if (i + 1) % display_every == 0:
            print("  step %03d/%03d, iter_time(ms)=%.0f" %
                  (i + 1, num_steps, iter_times[-1] * 1000))

        if (
            i > 1 and target_duration is not None and
            time.time() - initial_time > target_duration
        ):
            break

    if not iter_times:
        return results

    iter_times = np.array(iter_times)
    run_times = iter_times[num_warmup_iterations:]

    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = batch_size / np.mean(run_times)
    results['99th_percentile'] = np.percentile(
        run_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(run_times) * 1000
    results['latency_median'] = np.median(run_times) * 1000
    results['latency_min'] = np.min(run_times) * 1000
    results['latency_max'] = np.max(run_times) * 1000
    return results, predictions, image_ids


def eval_model(predictions, image_ids, annotation_path, output_saved_model_dir):
    name_map = {
        'output_0': 'boxes',
        'output_1': 'classes',
        'output_2': 'num_detections',
        'output_3': 'scores',
    }
    for old_key in list(predictions.keys()):
        if old_key in name_map:
            new_key = name_map[old_key]
            predictions[new_key] = predictions[old_key]
            del predictions[old_key]
    for key in predictions:
        predictions[key] = [t.numpy() for t in predictions[key]]
        predictions[key] = np.vstack(predictions[key])
        if key == 'num_detections':
            predictions[key] = predictions[key].ravel()

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
    tmp_dir = os.path.join(output_saved_model_dir, 'tmp_detection_results')
    os.makedirs(tmp_dir)

    coco_detections_path = os.path.join(tmp_dir, 'coco_detections.json')
    with open(coco_detections_path, 'w') as f:
        json.dump(coco_detections, f)
    cocoDt = coco.loadRes(coco_detections_path)

    subprocess.call(['rm', '-r', tmp_dir])

    # compute coco metrics
    eval = COCOeval(coco, cocoDt, 'bbox')
    eval.params.imgIds = image_ids

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return eval.stats[0]


def config_gpu_memory(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return
    print('Found the following GPUs:')
    for gpu in gpus:
        print('  ', gpu)
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
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=max_workspace_size_bytes)
    conversion_params = conversion_params._replace(
        precision_mode=precision_mode)
    conversion_params = conversion_params._replace(
        minimum_segment_size=minimum_segment_size)
    conversion_params = conversion_params._replace(
        use_calibration=precision_mode == 'INT8')
    return conversion_params


if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--input_saved_model_dir', type=str, default=None,
                        help='Directory containing the input saved model.')
    parser.add_argument('--output_saved_model_dir', type=str, default=None,
                        help='Directory in which the converted model is saved')
    parser.add_argument('--input_size', type=int, default=640,
                        help='Size of input images expected by the model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing validation set'
                             'TFRecord files.')
    parser.add_argument('--annotation_path', type=str,
                        help='Path that contains COCO annotations')
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
    parser.add_argument('--gpu_mem_cap', type=int, default=0,
                        help='Upper bound for GPU memory in MB.'
                             'Default is 0 which means allow_growth will be used')
    parser.add_argument('--max_workspace_size', type=int, default=(1 << 30),
                        help='workspace size in bytes')
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
            '({} <= {})'.format(args.num_iterations,
                                args.num_warmup_iterations))
    if args.num_calib_inputs < args.batch_size:
        raise ValueError(
            '--num_calib_inputs must not be smaller than --batch_size'
            '({} <= {})'.format(args.num_calib_inputs, args.batch_size))
    if args.data_dir is None and not args.use_synthetic:
        raise ValueError(
            "--data_dir required if you are not using synthetic data")
    if args.use_synthetic and args.num_iterations is None:
        raise ValueError("--num_iterations is required for --use_synthetic")
    if args.use_trt and not args.output_saved_model_dir:
        raise ValueError("--output_saved_model_dir must be set if use_trt=True")

    config_gpu_memory(args.gpu_mem_cap)

    params = get_trt_conversion_params(
        args.max_workspace_size,
        args.precision,
        args.minimum_segment_size
    )

    graph_func, times = get_graph_func(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        data_dir=args.data_dir,
        calib_data_dir=args.calib_data_dir,
        annotation_path=args.annotation_path,
        input_size=args.input_size,
        conversion_params=params,
        use_trt=args.use_trt,
        batch_size=args.batch_size,
        num_calib_inputs=args.num_calib_inputs,
        optimize_offline=args.optimize_offline)

    def print_dict(input_dict, prefix='  ', postfix=''):
        for k, v in sorted(input_dict.items()):
            print('{prefix}{arg_name}: {value}{postfix}'.format(
                prefix=prefix,
                arg_name=k,
                value='%.1f' % v if isinstance(v, float) else v,
                postfix=postfix
            ))

    print('Benchmark arguments:')
    print_dict(vars(args))
    print('TensorRT Conversion Params:')
    print_dict(dict(params._asdict()))
    print('Conversion times:')
    print_dict(times, postfix='s')

    results, predictions, image_ids = run_inference(
        graph_func,
        data_dir=args.data_dir,
        annotation_path=args.annotation_path,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        input_size=args.input_size,
        use_synthetic=args.use_synthetic,
        display_every=args.display_every,
        target_duration=args.target_duration
    )

    print('Results:')

    print('  mAP: %f' % eval_model(
        predictions,
        image_ids,
        args.annotation_path,
        args.output_saved_model_dir
    ))
    print('  images/sec: %d' % results['images_per_sec'])
    print('  99th_percentile(ms): %.2f' % results['99th_percentile'])
    print('  total_time(s): %.1f' % results['total_time'])
    print('  latency_mean(ms): %.2f' % results['latency_mean'])
    print('  latency_median(ms): %.2f' % results['latency_median'])
    print('  latency_min(ms): %.2f' % results['latency_min'])
    print('  latency_max(ms): %.2f' % results['latency_max'])
