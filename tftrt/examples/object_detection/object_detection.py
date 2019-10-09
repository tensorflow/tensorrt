# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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


from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from collections import namedtuple
from PIL import Image
import numpy as np
import time
import json
import subprocess
import os
import glob

from .graph_utils import force_nms_cpu as f_force_nms_cpu
from .graph_utils import replace_relu6 as f_replace_relu6
from .graph_utils import remove_assert as f_remove_assert

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2, image_resizer_pb2
from object_detection import exporter

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

Model = namedtuple('Model', ['name', 'url', 'extract_dir'])

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME = 'pipeline.config'
CHECKPOINT_PREFIX = 'model.ckpt'

MODELS = {
    'ssd_mobilenet_v1_coco':
    Model(
        'ssd_mobilenet_v1_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
        'ssd_mobilenet_v1_coco_2018_01_28',
    ),
    'ssd_mobilenet_v1_0p75_depth_quantized_coco':
    Model(
        'ssd_mobilenet_v1_0p75_depth_quantized_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz',
        'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18'
    ),
    'ssd_mobilenet_v1_ppn_coco':
    Model(
        'ssd_mobilenet_v1_ppn_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
        'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03'
    ),
    'ssd_mobilenet_v1_fpn_coco':
    Model(
        'ssd_mobilenet_v1_fpn_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
        'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    ),
    'ssd_mobilenet_v2_coco':
    Model(
        'ssd_mobilenet_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        'ssd_mobilenet_v2_coco_2018_03_29',
    ),
    'ssdlite_mobilenet_v2_coco':
    Model(
        'ssdlite_mobilenet_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz',
        'ssdlite_mobilenet_v2_coco_2018_05_09'),
    'ssd_inception_v2_coco':
    Model(
        'ssd_inception_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
        'ssd_inception_v2_coco_2018_01_28',
    ),
    'ssd_resnet_50_fpn_coco':
    Model(
        'ssd_resnet_50_fpn_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
        'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    ),
    'faster_rcnn_resnet50_coco':
    Model(
        'faster_rcnn_resnet50_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        'faster_rcnn_resnet50_coco_2018_01_28',
    ),
    'faster_rcnn_nas':
    Model(
        'faster_rcnn_nas',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
        'faster_rcnn_nas_coco_2018_01_28',
    ),
    'mask_rcnn_resnet50_atrous_coco':
    Model(
        'mask_rcnn_resnet50_atrous_coco',
        'http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz',
        'mask_rcnn_resnet50_atrous_coco_2018_01_28',
    ),
    'facessd_mobilenet_v2_quantized_open_image_v4':
    Model(
        'facessd_mobilenet_v2_quantized_open_image_v4',
        'http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz',
        'facessd_mobilenet_v2_quantized_320x320_open_image_v4')
}

Dataset = namedtuple(
    'Dataset',
    ['images_url', 'images_dir', 'annotation_url', 'annotation_path'])

DATASETS = {
    'val2014':
    Dataset(
        'http://images.cocodataset.org/zips/val2014.zip', 'val2014',
        'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'annotations/instances_val2014.json'),
    'train2014':
    Dataset(
        'http://images.cocodataset.org/zips/train2014.zip', 'train2014',
        'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'annotations/instances_train2014.json'),
    'val2017':
    Dataset(
        'http://images.cocodataset.org/zips/val2017.zip', 'val2017',
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'annotations/instances_val2017.json'),
    'train2017':
    Dataset(
        'http://images.cocodataset.org/zips/train2017.zip', 'train2017',
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'annotations/instances_train2017.json')
}


def download_model(model_name, input_dir=None, config_path=None, output_dir='.'):
    """Downloads a model from the TensorFlow Object Detection API

    Downloads a model from the TensorFlow Object Detection API to a specific
    output directory.  If an existing directory for the selected model already
    found under input_dir, the directory is copied to output_dir.

    Args
    ----
        model_name: A string representing the model to download.  This must be
            one of the keys in the module variable
            ``trt_samples.object_detection.MODELS``.
        input_dir: A string representing the directory to download the model from.
            If input_dir is None or if ``input_dir/<model_directory>`` already
            exists, the directory is copied to ``output_dir/<model_directory>``.
            Otherwise model is downloaded from Object Detection API repository.
        output_dir: A string representing the directory to download the model
            under.  A directory for the specified model will be created at
            ``output_dir/<model_directory>``.
    Returns
    -------
        config_path: A string representing the path to the object detection
            pipeline configuration file of the downloaded model.
        checkpoint_path: A string representing the path to the object detection
            model checkpoint.
    """
    global MODELS

    model_name

    model = MODELS[model_name]

    # make output directory if it doesn't exist
    subprocess.call(['mkdir', '-p', output_dir])

    tar_file = os.path.join(output_dir, os.path.basename(model.url))

    if not config_path:
        config_path = os.path.join(output_dir, model.extract_dir,
                                   PIPELINE_CONFIG_NAME)
    checkpoint_path = os.path.join(output_dir, model.extract_dir,
                                   CHECKPOINT_PREFIX)

    extract_dir = None
    if not input_dir:
        print('Downloading model from: %s' % model.url)
        subprocess.call(['wget', '-q', model.url, '-O', tar_file])
        subprocess.call(['tar', '-xzf', tar_file, '-C', output_dir])

        # hack fix to handle mobilenet_v2 config bug
        subprocess.call(['sed', '-i', '/batch_norm_trainable/d', config_path])
    else:
        extract_dir = os.path.join(input_dir, model.extract_dir)
        if os.path.exists(extract_dir):
            print('Found model at: %s' % extract_dir)
            print('Copying model to: %s' % output_dir)
            subprocess.call(['cp', '-r', extract_dir, output_dir])
        else:
            raise RuntimeError(
                'Cannot find model under {}'.format(extract_dir))
    return config_path, checkpoint_path


def build_model(model_name,
                input_dir=None,
                config_path=None,
                output_dir='.',
                override_nms_score_threshold=None,
                override_resizer_shape=None,
                batch_size=1,
                tmp_dir='.optimize_model_tmp_dir',
                remove_tmp_dir=True,
                output_path=None):
    """Downloads and builds a model from the TensorFlow Object Detection API

    It performs pre-tensorrt optimizations specific to the TensorFlow object
    detection API models.  Please see the list of arguments for other
    optimization parameters.

    Args
    ----
        model_name: check download_model
        input_dir: check download_model
        output_dir: check download_model
        override_nms_score_threshold: An optional float representing
            a NMS score threshold to override that specified in the object
            detection configuration file.
        override_resizer_shape: An optional list/tuple of integers
            representing a fixed shape to override the default image resizer
            specified in the object detection configuration file.
        batch_size: An integer representing the batch size
        tmp_dir: A string representing a directory for temporary files.  This
            directory will be created and removed by this function and should
            not already exist.  If the directory exists, an error will be
            thrown.
        remove_tmp_dir: A boolean indicating whether we should remove the
            tmp_dir or throw error.
        output_path: An optional string representing the path to save the
            optimized GraphDef to.

    Returns
    -------
        A GraphDef representing the optimized model.
    """

    # download model or use cached
    config_path, checkpoint_path = download_model(
        model_name=model_name,
        input_dir=input_dir,
        config_path=config_path,
        output_dir=output_dir)

    if os.path.exists(tmp_dir):
        if not remove_tmp_dir:
            raise RuntimeError(
                'Cannot create temporary directory, path exists: %s' % tmp_dir)
        subprocess.call(['rm', '-rf', tmp_dir])

    # load config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if override_nms_score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = override_nms_score_threshold
        if override_resizer_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = override_resizer_shape[
                0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = override_resizer_shape[
                1]
    elif config.model.HasField('faster_rcnn'):
        if override_nms_score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold = override_nms_score_threshold
        if override_resizer_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = override_resizer_shape[
                0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = override_resizer_shape[
                1]

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial), this will create tmp_dir
    with tf.Session(config=tf_config):
        with tf.Graph().as_default():
            exporter.export_inference_graph(
                INPUT_NAME,
                config,
                checkpoint_path,
                tmp_dir,
                input_shape=[batch_size, None, None, 3])

    # read frozen graph from file
    frozen_graph_path = os.path.join(tmp_dir, FROZEN_GRAPH_NAME)
    frozen_graph = tf.GraphDef()
    with open(frozen_graph_path, 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # write optimized model to disk
    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(frozen_graph.SerializeToString())

    # remove temporary directory
    subprocess.call(['rm', '-rf', tmp_dir])

    return frozen_graph

def optimize_model(frozen_graph,
                   use_trt=True,
                   force_nms_cpu=True,
                   replace_relu6=True,
                   remove_assert=True,
                   precision_mode='FP32',
                   minimum_segment_size=2,
                   max_workspace_size_bytes=1 << 32,
                   maximum_cached_engines=100,
                   calib_images_dir=None,
                   num_calib_images=None,
                   calib_batch_size=1,
                   calib_image_shape=None,
                   output_path=None):
    """Optimizes an object detection model using TensorRT

    Optimizes an object detection model using TensorRT.  This method also
    performs pre-tensorrt optimizations specific to the TensorFlow object
    detection API models.  Please see the list of arguments for other
    optimization parameters.

    Args
    ----
        frozen_graph: A GraphDef representing the optimized model.
        use_trt: A boolean representing whether to optimize with TensorRT. If
            False, regular TensorFlow will be used but other optimizations
            (like NMS device placement) will still be applied.
        force_nms_cpu: A boolean indicating whether to place NMS operations on
            the CPU.
        replace_relu6: A boolean indicating whether to replace relu6(x)
            operations with relu(x) - relu(x-6).
        remove_assert: A boolean indicating whether to remove Assert
            operations from the graph.
        precision_mode: A string representing the precision mode to use for
            TensorRT optimization.  Must be one of 'FP32', 'FP16', or 'INT8'.
        minimum_segment_size: An integer representing the minimum segment size
            to use for TensorRT graph segmentation.
        max_workspace_size_bytes: An integer representing the max workspace
            size for TensorRT optimization.
        maximum_cached_engines: An integer represenging the number of TRT engines
            that can be stored in the cache.
        calib_images_dir: A string representing a directory containing images to
            use for int8 calibration.
        num_calib_images: An integer representing the number of calibration
            images to use.  If None, will use all images in directory.
        calib_batch_size: An integer representing the batch size to use for calibration.
        calib_image_shape: A tuple of integers representing the height,
            width that images will be resized to for calibration.
        output_path: An optional string representing the path to save the
            optimized GraphDef to.

    Returns
    -------
        A GraphDef representing the optimized model.
    """
    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # optionally perform TensorRT optimization
    if use_trt:
        graph_size = len(frozen_graph.SerializeToString())
        num_nodes = len(frozen_graph.node)
        start_time = time.time()

        converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,
            nodes_blacklist=output_names,
            max_workspace_size_bytes=max_workspace_size_bytes,
            precision_mode=precision_mode,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=True,
            maximum_cached_engines=maximum_cached_engines)
        frozen_graph = converter.convert()

        end_time = time.time()
        print("graph_size(MB)(native_tf): %.1f" % (float(graph_size)/(1<<20)))
        print("graph_size(MB)(trt): %.1f" %
            (float(len(frozen_graph.SerializeToString()))/(1<<20)))
        print("num_nodes(native_tf): %d" % num_nodes)
        print("num_nodes(tftrt_total): %d" % len(frozen_graph.node))
        print("num_nodes(trt_only): %d" % len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp']))
        print("time(s) (trt_conversion): %.4f" % (end_time - start_time))

        # perform calibration for int8 precision
        if precision_mode == 'INT8':
            # get calibration images
            if calib_images_dir is None:
                raise ValueError('calib_images_dir must be provided for INT8 optimization.')
            image_paths = glob.glob(os.path.join(calib_images_dir, '*.jpg'))
            if len(image_paths) == 0:
                raise ValueError('No images were found in calib_images_dir for INT8 calibration.')
            image_paths = image_paths[:num_calib_images]
            num_batches = len(image_paths) // calib_batch_size

            def feed_dict_fn():
                # read batch of images
                batch_images = []
                for image_path in image_paths[feed_dict_fn.index:feed_dict_fn.index+calib_batch_size]:
                    image = _read_image(image_path, calib_image_shape)
                    batch_images.append(image)
                feed_dict_fn.index += calib_batch_size
                return {INPUT_NAME+':0': np.array(batch_images)}
            feed_dict_fn.index = 0

            print('Calibrating INT8...')
            start_time = time.time()
            frozen_graph = converter.calibrate(
                fetch_names=[x + ':0' for x in output_names],
                num_runs=num_batches,
                feed_dict_fn=feed_dict_fn)
            calibration_time = time.time() - start_time
            print("time(s) (trt_calibration): %.4f" % calibration_time)

    # write optimized model to disk
    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(frozen_graph.SerializeToString())

    return frozen_graph


def download_dataset(dataset_name, output_dir='.'):
    """Downloads a COCO dataset

    Downloads a COCO dataset to the specified output directory.  A new
    directory corresponding to the specified dataset will be created under
    output_dir.  This directory will contain the images of the dataset.

    Args
    ----
        dataset_name: A string representing the name of the dataset, it must
            be one of the keys in trt_samples.object_detection.DATASETS.

    Returns
    -------
        images_dir: A string representing the path of the directory containing
            images of the dataset.
        annotation_path: A string representing the path of the COCO annotation
            file for the dataset.
    """
    global DATASETS

    dataset = DATASETS[dataset_name]

    subprocess.call(['mkdir', '-p', output_dir])

    images_dir = os.path.join(output_dir, dataset.images_dir)
    images_zip_file = os.path.join(output_dir,
                                   os.path.basename(dataset.images_url))
    annotation_path = os.path.join(output_dir, dataset.annotation_path)
    annotation_zip_file = os.path.join(
        output_dir, os.path.basename(dataset.annotation_url))

    # download or use cached annotation
    if os.path.exists(annotation_path):
        print('Using cached annotation_path; %s' % (annotation_path))
    else:
        subprocess.call(
            ['wget', '-q', dataset.annotation_url, '-O', annotation_zip_file])
        subprocess.call(['unzip', annotation_zip_file, '-d', output_dir])

    # download or use cached images
    if os.path.exists(images_dir):
        print('Using cached images_dir; %s' % (images_dir))
    else:
        subprocess.call(['wget', '-q', dataset.images_url, '-O', images_zip_file])
        subprocess.call(['unzip', images_zip_file, '-d', output_dir])

    return images_dir, annotation_path


def benchmark_model(frozen_graph,
                    images_dir,
                    annotation_path,
                    batch_size=1,
                    image_shape=None,
                    num_images=4096,
                    tmp_dir='.benchmark_model_tmp_dir',
                    remove_tmp_dir=True,
                    output_path=None,
                    display_every=100,
                    use_synthetic=False,
                    num_warmup_iterations=50):
    """Computes accuracy and performance statistics

    Computes accuracy and performance statistics by executing over many images
    from the MSCOCO dataset defined by images_dir and annotation_path.

    Args
    ----
        frozen_graph: A GraphDef representing the object detection model to
            test.  Alternatively, a string representing the path to the saved
            frozen graph.
        images_dir: A string representing the path of the COCO images
            directory.
        annotation_path: A string representing the path of the COCO annotation
            file.
        batch_size: An integer representing the batch size to use when feeding
            images to the model.
        image_shape: An optional tuple of integers representing a fixed shape
            to resize all images before testing. For synthetic data the default
            image_shape is [600, 600, 3]
        num_images: An integer representing the number of images in the
            dataset to evaluate with.
        tmp_dir: A string representing the path where the function may create
            a temporary directory to store intermediate files.
        output_path: An optional string representing a path to store the
            statistics in JSON format.
        display_every: int, print log every display_every iteration
        num_warmup_iteration: An integer represtening number of initial iteration,
            that are not cover in performance statistics
    Returns
    -------
        statistics: A named dictionary of accuracy and performance statistics
        computed for the model.
    """
    if os.path.exists(tmp_dir):
        if not remove_tmp_dir:
            raise RuntimeError('Temporary directory exists; %s' % tmp_dir)
        subprocess.call(['rm', '-rf', tmp_dir])
    if batch_size > 1 and image_shape is None:
        raise RuntimeError(
            'Fixed image shape must be provided for batch size > 1')

    if not use_synthetic:
        coco = COCO(annotation_file=annotation_path)


        # get list of image ids to use for evaluation
        image_ids = coco.getImgIds()
        if num_images > len(image_ids):
            print(
                'Num images provided %d exceeds number in dataset %d, using %d images instead'
                % (num_images, len(image_ids), len(image_ids)))
            num_images = len(image_ids)
        image_ids = image_ids[0:num_images]

    # load frozen graph from file if string, otherwise must be GraphDef
    if isinstance(frozen_graph, str):
        frozen_graph_path = frozen_graph
        frozen_graph = tf.GraphDef()
        with open(frozen_graph_path, 'rb') as f:
            frozen_graph.ParseFromString(f.read())
    elif not isinstance(frozen_graph, tf.GraphDef):
        raise TypeError('Expected frozen_graph to be GraphDef or str')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    coco_detections = []  # list of all bounding box detections in coco format
    runtimes = []  # list of runtimes for each batch
    image_counts = []  # list of number of images in each batch

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            tf.import_graph_def(frozen_graph, name='')
            tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
            tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
            tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
            tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
            tf_num_detections = tf_graph.get_tensor_by_name(
                NUM_DETECTIONS_NAME + ':0')

            # load batches from coco dataset
            for image_idx in range(0, num_images, batch_size):
                if use_synthetic:
                    if image_shape is None:
                        batch_images = np.random.randint(0, 256, size=(batch_size, 600, 600, 3))
                    else:
                        batch_images = np.random.randint(0, 256, size=(batch_size, image_shape[0], image_shape[1], 3))
                else:
                    batch_image_ids = image_ids[image_idx:image_idx + batch_size]
                    batch_images = []
                    batch_coco_images = []
                    # read images from file
                    for image_id in batch_image_ids:
                        coco_img = coco.imgs[image_id]
                        batch_coco_images.append(coco_img)
                        image_path = os.path.join(images_dir,
                                                  coco_img['file_name'])
                        image = _read_image(image_path, image_shape)
                        batch_images.append(image)

                # run num_warmup_iterations outside of timing
                if image_idx < num_warmup_iterations:
                    boxes, classes, scores, num_detections = tf_sess.run(
                        [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                        feed_dict={tf_input: batch_images})
                else:
                    # execute model and compute time difference
                    t0 = time.time()
                    boxes, classes, scores, num_detections = tf_sess.run(
                        [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                        feed_dict={tf_input: batch_images})
                    t1 = time.time()

                    # log runtime and image count
                    runtimes.append(float(t1 - t0))
                    if len(runtimes) % display_every == 0:
                        print("    step %d/%d, iter_time(ms)=%.4f" % (
                            len(runtimes),
                            (num_images + batch_size - 1) / batch_size,
                            np.mean(runtimes) * 1000))
                    image_counts.append(len(batch_images))

                if not use_synthetic:
                    # add coco detections for this batch to running list
                    batch_coco_detections = []
                    for i, image_id in enumerate(batch_image_ids):
                        image_width = batch_coco_images[i]['width']
                        image_height = batch_coco_images[i]['height']

                        for j in range(int(num_detections[i])):
                            bbox = boxes[i][j]
                            bbox_coco_fmt = [
                                bbox[1] * image_width,  # x0
                                bbox[0] * image_height,  # x1
                                (bbox[3] - bbox[1]) * image_width,  # width
                                (bbox[2] - bbox[0]) * image_height,  # height
                            ]

                            coco_detection = {
                                'image_id': image_id,
                                'category_id': int(classes[i][j]),
                                'bbox': bbox_coco_fmt,
                                'score': float(scores[i][j])
                            }

                            coco_detections.append(coco_detection)

    if not use_synthetic:
        # write coco detections to file
        subprocess.call(['mkdir', '-p', tmp_dir])
        coco_detections_path = os.path.join(tmp_dir, 'coco_detections.json')
        with open(coco_detections_path, 'w') as f:
            json.dump(coco_detections, f)

        # compute coco metrics
        cocoDt = coco.loadRes(coco_detections_path)
        eval = COCOeval(coco, cocoDt, 'bbox')
        eval.params.imgIds = image_ids

        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        statistics = {
            'map': eval.stats[0],
            'avg_latency_ms': 1000.0 * np.mean(runtimes),
            'avg_throughput_fps': np.sum(image_counts) / np.sum(runtimes),
            'runtimes_ms': [1000.0 * r for r in runtimes]
        }
    else:
        statistics = {
            'avg_latency_ms': 1000.0 * np.mean(runtimes),
            'avg_throughput_fps': np.sum(image_counts) / np.sum(runtimes),
            'runtimes_ms': [1000.0 * r for r in runtimes]
        }

    if output_path is not None:
        subprocess.call(['mkdir', '-p', os.path.dirname(output_path)])
        with open(output_path, 'w') as f:
            json.dump(statistics, f)
    subprocess.call(['rm', '-rf', tmp_dir])

    return statistics


def _read_image(image_path, image_shape):
    image = Image.open(image_path).convert('RGB')
    if image_shape is not None:
        image = image.resize(image_shape[::-1])
    return np.array(image)
