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

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import numpy as np
import sys
import glob
import shutil
import subprocess
import nets.nets_factory
import tensorflow.contrib.slim as slim
import official.resnet.imagenet_main
from preprocessing import inception_preprocessing, vgg_preprocessing

class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""
    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def before_run(self, run_context):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.iter_times.append(duration)
        current_step = len(self.iter_times)
        if current_step % self.display_every == 0:
            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%d" % (
                current_step, self.num_steps, duration * 1000,
                self.batch_size / self.iter_times[-1]))

class BenchmarkHook(tf.train.SessionRunHook):
    """Limits run duration and number of iterations"""
    def __init__(self, target_duration=None, iteration_limit=None):
        self.target_duration = target_duration
        self.start_time = None
        self.current_iteration = 0
        self.iteration_limit = iteration_limit

    def before_run(self, run_context):
        if not self.start_time:
            self.start_time = time.time()
            if self.target_duration:
                print("    running for target duration {} seconds from {}".format(
                    self.target_duration, time.asctime(time.localtime(self.start_time))))

    def after_run(self, run_context, run_values):
        if self.target_duration:
            current_time = time.time()
            if (current_time - self.start_time) > self.target_duration:
                print("    target duration {} reached at {}, requesting stop".format(
                    self.target_duration, time.asctime(time.localtime(current_time))))
                run_context.request_stop()

        if self.iteration_limit:
            self.current_iteration += 1
            if self.current_iteration >= self.iteration_limit:
                run_context.request_stop()

def run(frozen_graph, model, data_files, batch_size,
    num_iterations, num_warmup_iterations, use_synthetic, display_every=100,
    mode='validation', target_duration=None):
    """Evaluates a frozen graph

    This function evaluates a graph on the ImageNet validation set.
    tf.estimator.Estimator is used to evaluate the accuracy of the model
    and a few other metrics. The results are returned as a dict.

    frozen_graph: GraphDef, a graph containing input node 'input' and outputs 'logits' and 'classes'
    model: string, the model name (see NETS table in graph.py)
    data_files: List of TFRecord files used for inference
    batch_size: int, batch size for TensorRT optimizations
    num_iterations: int, number of iterations(batches) to run for
    num_warmup_iterations: int, number of iteration(batches) to exclude from benchmark measurments
    use_synthetic: bool, if true run using real data, otherwise synthetic
    display_every: int, print log every @display_every iteration
    mode: validation - using estimator.evaluate with accuracy measurments,
          benchmark - using estimator.predict
    """
    # Define model function for tf.estimator.Estimator
    def model_fn(features, labels, mode):
        logits_out, classes_out = tf.import_graph_def(frozen_graph,
            input_map={'input': features},
            return_elements=['logits:0', 'classes:0'],
            name='')
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                      predictions={'classes': classes_out})
        if mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=classes_out, name='acc_op')
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={'accuracy': accuracy})

    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(model, mode)

    def get_tfrecords_count(files):
        num_records = 0
        for fn in files:
            for record in tf.python_io.tf_record_iterator(fn):
                num_records += 1
        return num_records

    # Define the dataset input function for tf.estimator.Estimator
    def input_fn():
        if use_synthetic:
            input_width, input_height = get_netdef(model).get_input_dims()
            features = np.random.normal(
                loc=112, scale=70,
                size=(batch_size, input_height, input_width, 3)).astype(np.float32)
            features = np.clip(features, 0.0, 255.0)
            labels = np.random.randint(
                low=0,
                high=get_netdef(model).get_num_classes(),
                size=(batch_size),
                dtype=np.int32)
            with tf.device('/device:GPU:0'):
                features = tf.convert_to_tensor(tf.get_variable("features", dtype=tf.float32, initializer=tf.constant(features)))
                labels = tf.identity(tf.constant(labels))
        else:
            if mode == 'validation':
                dataset = tf.data.TFRecordDataset(data_files)
                dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=8))
                dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
                dataset = dataset.repeat(count=1)
                iterator = dataset.make_one_shot_iterator()
                features, labels = iterator.get_next()
            elif mode == 'benchmark':
                dataset = tf.data.Dataset.from_tensor_slices(data_files)
                dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=8))
                dataset = dataset.repeat(count=1)
                iterator = dataset.make_one_shot_iterator()
                features = iterator.get_next()
                labels = np.random.randint(
                    low=0,
                    high=get_netdef(model).get_num_classes(),
                    size=(batch_size),
                    dtype=np.int32)
                labels = tf.identity(tf.constant(labels))
            else:
                raise ValueError("Mode must be either 'validation' or 'benchmark'")
        return features, labels

    # Evaluate model
    if use_synthetic:
        num_records = num_iterations * batch_size
    elif mode == 'validation':
        num_records = get_tfrecords_count(data_files)
    elif mode == 'benchmark':
        num_records = len(data_files)
    else:
        raise ValueError("Mode must be either 'validation' or 'benchmark'")
    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=num_records)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf_config),
        model_dir='model_dir')
    results = {}
    if mode == 'validation':
        results = estimator.evaluate(input_fn, steps=num_iterations, hooks=[logger])
    elif mode == 'benchmark':
        benchmark_hook = BenchmarkHook(target_duration=target_duration, iteration_limit=num_iterations)
        prediction_results = [p for p in estimator.predict(input_fn, predict_keys=["classes"],  hooks=[logger, benchmark_hook])]
    else:
        raise ValueError("Mode must be either 'validation' or 'benchmark'")
    # Gather additional results
    iter_times = np.array(logger.iter_times[num_warmup_iterations:])
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    results['latency_median'] = np.median(iter_times) * 1000
    results['latency_min'] = np.min(iter_times) * 1000
    return results

class NetDef(object):
    """Contains definition of a model

    name: Name of model
    url: (optional) Where to download archive containing checkpoint
    model_dir_in_archive: (optional) Subdirectory in archive containing
        checkpoint files.
    preprocess: Which preprocessing method to use for inputs.
    input_size: Input dimensions.
    slim: If True, use tensorflow/research/slim/nets to build graph. Else, use
        model_fn to build graph.
    postprocess: Postprocessing function on predictions.
    model_fn: Function to build graph if slim=False
    num_classes: Number of output classes in model. Background class will be
        automatically adjusted for if num_classes is 1001.
    """
    def __init__(self, name, url=None, model_dir_in_archive=None,
                checkpoint_name=None, preprocess='inception',
            input_size=224, slim=True, postprocess=tf.nn.softmax, model_fn=None, num_classes=1001):
        self.name = name
        self.url = url
        self.model_dir_in_archive = model_dir_in_archive
        self.checkpoint_name = checkpoint_name
        if preprocess == 'inception':
            self.preprocess = inception_preprocessing.preprocess_image
        elif preprocess == 'vgg':
            self.preprocess = vgg_preprocessing.preprocess_image
        self.input_width = input_size
        self.input_height = input_size
        self.slim = slim
        self.postprocess = postprocess
        self.model_fn = model_fn
        self.num_classes = num_classes

    def get_input_dims(self):
        return self.input_width, self.input_height

    def get_num_classes(self):
        return self.num_classes

    def get_url(self):
        return self.url


def get_netdef(model):
    """Creates the dictionary NETS with model names as keys and NetDef as values.
    Returns the NetDef corresponding to the model specified in the parameter.
    model: string, the model name (see NETS table)
    """
    NETS = {
        'mobilenet_v1': NetDef(
            name='mobilenet_v1',
            url='http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz'),

        'mobilenet_v2': NetDef(
            name='mobilenet_v2_140',
            url='https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz'),

        'nasnet_mobile': NetDef(
            name='nasnet_mobile',
            url='https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz'),

        'nasnet_large': NetDef(
            name='nasnet_large',
            url='https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz',
            input_size=331),

        'resnet_v1_50': NetDef(
            name='resnet_v1_50',
            url='http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v1_fp32_20181001.tar.gz',
            model_dir_in_archive='resnet_imagenet_v1_fp32_20181001',
            slim=False,
            preprocess='vgg',
            model_fn=official.resnet.imagenet_main.ImagenetModel(resnet_size=50, resnet_version=1)),

        'resnet_v2_50': NetDef(
            name='resnet_v2_50',
            url='http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v2_fp32_20181001.tar.gz',
            model_dir_in_archive='resnet_imagenet_v2_fp32_20181001',
            slim=False,
            preprocess='vgg',
            model_fn=official.resnet.imagenet_main.ImagenetModel(resnet_size=50, resnet_version=2)),

        'resnet_v2_152': NetDef(
            name='resnet_v2_152',
            slim=False,
            preprocess='vgg',
            model_fn=official.resnet.imagenet_main.ImagenetModel(resnet_size=152, resnet_version=2)),

        'vgg_16': NetDef(
            name='vgg_16',
            url='http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
            preprocess='vgg',
            num_classes=1000),

        'vgg_19': NetDef(
            name='vgg_19',
            url='http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
            preprocess='vgg',
            num_classes=1000),

        'inception_v3': NetDef(
            name='inception_v3',
            url='http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
            input_size=299),

        'inception_v4': NetDef(
            name='inception_v4',
            url='http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
            input_size=299),
    }
    return NETS[model]

def deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

def get_preprocess_fn(model, mode='validation'):
    """Creates a function to parse and process a TFRecord using the model's parameters

    model: string, the model name (see NETS table)
    mode: string, which mode to use (validation or benchmark)
    returns: function, the preprocessing function for a record
    """
    def validation_process(record):
        # Parse TFRecord
        imgdata, label, bbox, text = deserialize_image_record(record)
        label -= 1 # Change to 0-based (don't use background class)
        try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except: image = tf.image.decode_png(imgdata, channels=3)
        # Use model's preprocessing function
        netdef = get_netdef(model)
        image = netdef.preprocess(image, netdef.input_height, netdef.input_width, is_training=False)
        return image, label

    def benchmark_process(path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        net_def = get_netdef(model)
        input_width, input_height = net_def.get_input_dims()
        image = net_def.preprocess(image, input_width, input_height, is_training=False)
        return image

    if mode == 'validation':
        return validation_process
    elif mode == 'benchmark':
        return benchmark_process
    else:
        raise ValueError("Mode must be either 'validation' or 'benchmark'")



def build_classification_graph(model, model_dir=None, default_models_dir='./data'):
    """Builds an image classification model by name

    This function builds an image classification model given a model
    name, parameter checkpoint file path, and number of classes.  This
    function performs some graph processing to produce a graph that is
    well optimized by the TensorRT package in TensorFlow 1.7+.

    model: string, the model name (see NETS table)
    model_dir: string, optional user provided checkpoint location
    default_models_dir: string, directory to store downloaded model checkpoints
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    netdef = get_netdef(model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            tf_input = tf.placeholder(tf.float32, [None, netdef.input_height, netdef.input_width, 3], name='input')
            if netdef.slim:
                # TF Slim Model: get model function from nets_factory
                network_fn = nets.nets_factory.get_network_fn(netdef.name, netdef.num_classes,
                        is_training=False)
                tf_net, tf_end_points = network_fn(tf_input)
            else:
                # TF Official Model: get model function from NETS
                tf_net = netdef.model_fn(tf_input, training=False)

            tf_output = tf.identity(tf_net, name='logits')
            num_classes = tf_output.get_shape().as_list()[1]
            if num_classes == 1001:
                # Shift class down by 1 if background class was included
                tf_output_classes = tf.add(tf.argmax(tf_output, axis=1), -1, name='classes')
            else:
                tf_output_classes = tf.argmax(tf_output, axis=1, name='classes')

            # Get checkpoint.
            checkpoint_path = get_checkpoint(model, model_dir, default_models_dir)
            print('Using checkpoint:', checkpoint_path)
            # load checkpoint
            tf_saver = tf.train.Saver()
            tf_saver.restore(save_path=checkpoint_path, sess=tf_sess)

            # freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=['logits', 'classes']
            )

    return frozen_graph

def get_checkpoint(model, model_dir=None, default_models_dir='.'):
    """Get the checkpoint. User may provide their own checkpoint via model_dir.
    If model_dir is None, attempts to download the checkpoint using url property
    from model definition (see get_netdef()). default_models_dir/model is first
    checked to see if the checkpoint was already downloaded. If not, the
    checkpoint will be downloaded from the url.

    model: string, the model name (see NETS table)
    model_dir: string, optional user provided checkpoint location
    default_models_dir: string, the directory where files are downloaded to
    returns: string, path to the checkpoint file containing trained model params
    """
    # User has provided a checkpoint
    if model_dir:
        checkpoint_path = find_checkpoint_in_dir(model_dir)
        if not checkpoint_path:
            print('No checkpoint was found in', model_dir)
            exit(1)
        return checkpoint_path

    # User has not provided a checkpoint. We need to download one. First check
    # if checkpoint was already downloaded and stored in default_models_dir.
    model_dir = os.path.join(default_models_dir, model)
    checkpoint_path = find_checkpoint_in_dir(model_dir)
    if checkpoint_path:
        return checkpoint_path

    # Checkpoint has not yet been downloaded. Download checkpoint if model has
    # defined a URL.
    if get_netdef(model).url:
        download_checkpoint(model, model_dir)
        return find_checkpoint_in_dir(model_dir)

    print('No model_dir was provided and the model does not define a download' \
          ' URL.')
    exit(1)

def find_checkpoint_in_dir(model_dir):
    # tf.train.latest_checkpoint will find checkpoints if a 'checkpoint' file is
    # present in the directory.
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    if checkpoint_path:
        return checkpoint_path

    # tf.train.latest_checkpoint did not find anything. Find .ckpt file
    # manually.
    files = glob.glob(os.path.join(model_dir, '*.ckpt*'))
    if len(files) == 0:
        return None
    # Use last file for consistency if more than one (may not actually be
    # "latest").
    checkpoint_path = sorted(files)[-1]
    # Trim after .ckpt-* segment. For example:
    # model.ckpt-257706.data-00000-of-00002 -> model.ckpt-257706
    parts = checkpoint_path.split('.')
    ckpt_index = [i for i in range(len(parts)) if 'ckpt' in parts[i]][0]
    checkpoint_path = '.'.join(parts[:ckpt_index+1])
    return checkpoint_path


def download_checkpoint(model, destination_path):
    #copy files from source to destination (without any directories)
    def copy_files(source, destination):
        try:
            shutil.copy2(source, destination)
        except (OSError, IOError) as e:
            pass
        except shutil.Error as e:
            pass

    # Make directories if they don't exist.
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    # Download archive.
    archive_path = os.path.join(destination_path,
                                os.path.basename(get_netdef(model).url))
    if not os.path.isfile(archive_path):
        subprocess.check_call(['wget', '--no-check-certificate', '-q',
                               get_netdef(model).url, '-O', archive_path])
    # Extract.
    subprocess.check_call(['tar', '-xzf', archive_path, '-C', destination_path])
    # Move checkpoints out of archive sub directories into destination_path
    if get_netdef(model).model_dir_in_archive:
        source_files = os.path.join(destination_path,
                                    get_netdef(model).model_dir_in_archive,
                                    '*')
        for f in glob.glob(source_files):
            copy_files(f, destination_path)

def get_frozen_graph(
    model,
    model_dir=None,
    use_trt=False,
    engine_dir=None,
    use_dynamic_op=False,
    precision='FP32',
    batch_size=8,
    minimum_segment_size=2,
    calib_files=None,
    num_calib_inputs=None,
    use_synthetic=False,
    cache=False,
    default_models_dir='./data',
    max_workspace_size=(1<<32)):
    """Retreives a frozen GraphDef from model definitions in classification.py and applies TF-TRT

    model: str, the model name (see NETS table in classification.py)
    use_trt: bool, if true, use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    num_nodes = {}
    times = {}
    graph_sizes = {}

    # Load from pb file if frozen graph was already created and cached
    if cache:
        # Graph must match the model, TRT mode, precision, and batch size
        prebuilt_graph_path = "graphs/frozen_graph_%s_%d_%s_%d.pb" % (model, int(use_trt), precision, batch_size)
        if os.path.isfile(prebuilt_graph_path):
            print('Loading cached frozen graph from \'%s\'' % prebuilt_graph_path)
            start_time = time.time()
            with tf.gfile.GFile(prebuilt_graph_path, "rb") as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())
            times['loading_frozen_graph'] = time.time() - start_time
            num_nodes['loaded_frozen_graph'] = len(frozen_graph.node)
            num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
            graph_sizes['loaded_frozen_graph'] = len(frozen_graph.SerializeToString())
            return frozen_graph, num_nodes, times, graph_sizes

    # Build graph and load weights
    frozen_graph = build_classification_graph(model, model_dir, default_models_dir)
    num_nodes['native_tf'] = len(frozen_graph.node)
    graph_sizes['native_tf'] = len(frozen_graph.SerializeToString())

    # Convert to TensorRT graph
    if use_trt:
        start_time = time.time()
        frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=['logits', 'classes'],
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=precision.upper(),
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=use_dynamic_op
        )
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
        graph_sizes['trt'] = len(frozen_graph.SerializeToString())

        if engine_dir:
            segment_number = 0
            for node in frozen_graph.node:
                if node.op == "TRTEngineOp":
                    engine = node.attr["serialized_segment"].s
                    engine_path = engine_dir+'/{}_{}_{}_segment{}.trtengine'.format(model, precision, batch_size, segment_number)
                    segment_number += 1
                    with open(engine_path, "wb") as f:
                        f.write(engine)

        if precision == 'INT8':
            calib_graph = frozen_graph
            graph_sizes['calib'] = len(calib_graph.SerializeToString())
            # INT8 calibration step
            print('Calibrating INT8...')
            start_time = time.time()
            run(calib_graph, model, calib_files, batch_size,
                num_calib_inputs // batch_size, 0, use_synthetic=use_synthetic)
            times['trt_calibration'] = time.time() - start_time

            start_time = time.time()
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)
            times['trt_int8_conversion'] = time.time() - start_time
            # This is already set but overwriting it here to ensure the right size
            graph_sizes['trt'] = len(frozen_graph.SerializeToString())

            del calib_graph
            print('INT8 graph created.')

    # Cache graph to avoid long conversions each time
    if cache:
        if not os.path.exists(os.path.dirname(prebuilt_graph_path)):
            try:
                os.makedirs(os.path.dirname(prebuilt_graph_path))
            except Exception as e:
                raise e
        start_time = time.time()
        with tf.gfile.GFile(prebuilt_graph_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        times['saving_frozen_graph'] = time.time() - start_time

    return frozen_graph, num_nodes, times, graph_sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='inception_v4',
        choices=['mobilenet_v1', 'mobilenet_v2', 'nasnet_mobile', 'nasnet_large',
                 'resnet_v1_50', 'resnet_v2_50', 'resnet_v2_152', 'vgg_16', 'vgg_19',
                 'inception_v3', 'inception_v4'],
        help='Which model to use.')
    parser.add_argument('--data_dir', type=str, default=None,
        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
        help='Directory containing TFRecord files for calibrating INT8.')
    parser.add_argument('--model_dir', type=str, default=None,
        help='Directory containing model checkpoint. If not provided, a ' \
             'checkpoint may be downloaded automatically and stored in ' \
             '"{--default_models_dir}/{--model}" for future use.')
    parser.add_argument('--default_models_dir', type=str, default='./data',
        help='Directory where downloaded model checkpoints will be stored and ' \
             'loaded from if --model_dir is not provided.')
    parser.add_argument('--use_trt', action='store_true',
        help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--engine_dir', type=str, default=None,
        help='Directory where to write trt engines. Engines are written only if the directory ' \
             'is provided. This option is ignored when not using tf_trt.')
    parser.add_argument('--use_trt_dynamic_op', action='store_true',
        help='If set, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8'], default='FP32',
        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=8,
        help='Number of images per batch.')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--num_iterations', type=int, default=None,
        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--use_synthetic', action='store_true',
        help='If set, one batch of random data is generated and used at every iteration.')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
        help='Number of initial iterations skipped from timing')
    parser.add_argument('--num_calib_inputs', type=int, default=500,
        help='Number of inputs (e.g. images) used for calibration '
        '(last batch is skipped in case it is not full)')
    parser.add_argument('--max_workspace_size', type=int, default=(1<<32),
        help='workspace size in bytes')
    parser.add_argument('--cache', action='store_true',
        help='If set, graphs will be saved to disk after conversion. If a converted graph is present on disk, it will be loaded instead of building the graph again.')
    parser.add_argument('--mode', choices=['validation', 'benchmark'], default='validation',
        help='Which mode to use (validation or benchmark)')
    parser.add_argument('--target_duration', type=int, default=None,
        help='If set, script will run for specified number of seconds.')
    args = parser.parse_args()

    if args.precision != 'FP32' and not args.use_trt:
        raise ValueError('TensorRT must be enabled for FP16 or INT8 modes (--use_trt).')
    if args.precision == 'INT8' and not args.calib_data_dir and not args.use_synthetic:
        raise ValueError('--calib_data_dir is required for INT8 mode')
    if args.num_iterations is not None and args.num_iterations <= args.num_warmup_iterations:
        raise ValueError('--num_iterations must be larger than --num_warmup_iterations '
            '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))
    if args.num_calib_inputs < args.batch_size:
        raise ValueError('--num_calib_inputs must not be smaller than --batch_size'
            '({} <= {})'.format(args.num_calib_inputs, args.batch_size))
    if args.mode == 'validation' and args.use_synthetic:
        raise ValueError('Cannot use both validation mode and synthetic dataset')
    if args.data_dir is None and not args.use_synthetic:
        raise ValueError("--data_dir required if you are not using synthetic data")
    if args.use_synthetic and args.num_iterations is None:
        raise ValueError("--num_iterations is required for --use_synthetic")

    def get_files(data_dir, filename_pattern):
        if data_dir == None:
            return []
        files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
        if files == []:
            raise ValueError('Can not find any files in {} with '
                             'pattern "{}"'.format(data_dir, filename_pattern))
        return files

    calib_files = []
    data_files = []
    if not args.use_synthetic:
        if args.mode == "validation":
            data_files = get_files(args.data_dir, 'validation*')
        elif args.mode == "benchmark":
            data_files = [os.path.join(path, name) for path, _, files in os.walk(args.data_dir) for name in files]
        else:
            raise ValueError("Mode must be either 'validation' or 'benchamark'")
        calib_files = get_files(args.calib_data_dir, 'train*')

    frozen_graph, num_nodes, times, graph_sizes = get_frozen_graph(
        model=args.model,
        model_dir=args.model_dir,
        use_trt=args.use_trt,
        engine_dir=args.engine_dir,
        use_dynamic_op=args.use_trt_dynamic_op,
        precision=args.precision,
        batch_size=args.batch_size,
        minimum_segment_size=args.minimum_segment_size,
        calib_files=calib_files,
        num_calib_inputs=args.num_calib_inputs,
        use_synthetic=args.use_synthetic,
        cache=args.cache,
        default_models_dir=args.default_models_dir,
        max_workspace_size=args.max_workspace_size)

    def print_dict(input_dict, str='', scale=None):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            v = v * scale if scale else v
            print('{}{}'.format(headline, '%.1f'%v if type(v)==float else v))

    print_dict(vars(args))
    print("url: " + get_netdef(args.model).get_url())
    print_dict(num_nodes, str='num_nodes')
    print_dict(graph_sizes, str='graph_size(MB)', scale=1./(1<<20))
    print_dict(times, str='time(s)')

    # Evaluate model
    print('running inference...')
    results = run(
        frozen_graph,
        model=args.model,
        data_files=data_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        use_synthetic=args.use_synthetic,
        display_every=args.display_every,
        mode=args.mode,
        target_duration=args.target_duration)

    # Display results
    print('results of {}:'.format(args.model))
    if args.mode == 'validation':
        print('    accuracy: %.2f' % (results['accuracy'] * 100))
    print('    images/sec: %d' % results['images_per_sec'])
    print('    99th_percentile(ms): %.2f' % results['99th_percentile'])
    print('    total_time(s): %.1f' % results['total_time'])
    print('    latency_mean(ms): %.2f' % results['latency_mean'])
    print('    latency_median(ms): %.2f' % results['latency_median'])
    print('    latency_min(ms): %.2f' % results['latency_min'])
