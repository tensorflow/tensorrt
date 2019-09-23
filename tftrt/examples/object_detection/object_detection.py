import pprint
import os
import time
import json
import numpy as np
import tensorflow as tf
import argparse
import subprocess
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'
DEFAULT_NAMES = {
    'input': INPUT_NAME,
    'boxes': BOXES_NAME,
    'classes': CLASSES_NAME,
    'scores': SCORES_NAME,
    'num_detections': NUM_DETECTIONS_NAME,
    'masks': MASKS_NAME,
}
NAME_MAP = {
    'output_0':'boxes',
    'output_1':'classes',
    'output_2':'num_detections',
    'output_3':'scores',
}

def get_model_func(saved_model_dir,
                   use_trt,
                   tmp_dir,
                   cache,
                   annotation_path=None,
                   num_calib_images=None,
                   calib_batch_size=None,
                   images_dir=None,
                   image_shape=None,
                   conversion_params=None):
    if use_trt and conversion_params is None:
        raise ValueError("Provide conversion params if you're using TRT")
    loaded = tf.saved_model.load(saved_model_dir)
    infer = loaded.signatures['serving_default']
    times = {}
    num_nodes = {}
    graph_sizes = {}
    cache_dir = os.path.join(tmp_dir, saved_model_dir)
    if cache and os.path.exists(cache_dir):
        loaded = tf.saved_model.load(cache_dir)
        return loaded.signatures['serving_default']

    if use_trt:
        print("Converting graph with TRT")
        start_time = time.time()
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir,
            conversion_params=conversion_params,
        )
        if conversion_params.precision_mode != 'INT8':
            converter.convert()
        else:
            def input_fn():
                coco = COCO(annotation_file=annotation_path)
                image_ids = coco.getImgIds()
                num_images = num_calib_images
                if num_images > len(image_ids):
                    print(
                        'Num images provided %d exceeds number in dataset %d, using %d images instead'
                        % (num_images, len(image_ids), len(image_ids)))
                    num_images = len(image_ids)
                image_ids = image_ids[0:num_images]
                dataset = get_dataset(images_dir, annotation_path, calib_batch_size, image_ids,
                                      use_synthetic=False, coco=coco, image_shape=image_shape)
                for i, batch_feats in enumerate(dataset):
                    yield (batch_feats,)
            converter.convert(calibration_input_fn=input_fn)
        converter.save(cache_dir)
        loaded = tf.saved_model.load(cache_dir)
        graph_func = loaded.signatures['serving_default']

        converted_graph_def = converter._converted_graph_def
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(converted_graph_def.node)
        num_nodes['trt_only'] = len([1 for n in converted_graph_def.node if str(n.op)=='TRTEngineOp'])
        graph_sizes['trt'] = len(converted_graph_def.SerializeToString())
    else:
        loaded = tf.saved_model.load(saved_model_dir)
        graph_func = loaded.signatures['serving_default']
    return graph_func

def get_dataset(images_dir,
                annotation_path,
                batch_size,
                image_ids,
                coco,
                use_synthetic,
                image_shape=(640,640),):
    if use_synthetic:
        features = np.random.normal(
            loc=112, scale=70,
            size=(batch_size, *image_shape, 3)).astype(np.float32)
        features = np.clip(features, 0.0, 255.0)
        features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
            "features", dtype=tf.float32, initializer=tf.constant(features)))
        dataset = tf.data.Dataset.from_tensor_slices([features])
        dataset = dataset.repeat()
    else:
        image_paths = []
        for image_id in image_ids:
            coco_img = coco.imgs[image_id]
            image_paths.append(os.path.join(images_dir, coco_img['file_name']))
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        def preprocess_fn(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            if image_shape is not None:
                image = tf.image.resize(image, size=image_shape)
                image = tf.cast(image, tf.uint8)
            return image
        dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(count=1)
    return dataset

def benchmark_model(graph_func,
                    images_dir,
                    annotation_path,
                    batch_size=1,
                    image_shape=(640, 640),
                    num_images=2048,
                    output_path=None,
                    display_every=100,
                    use_synthetic=False,
                    num_warmup_iterations=50,
                    model=None):
    if not use_synthetic:
        coco = COCO(annotation_file=annotation_path)
        image_ids = coco.getImgIds()
        if num_images > len(image_ids):
            print(
                'Num images provided %d exceeds number in dataset %d, using %d images instead'
                % (num_images, len(image_ids), len(image_ids)))
            num_images = len(image_ids)
        image_ids = image_ids[0:num_images]

    coco_detections = []  # list of all bounding box detections in coco format
    runtimes = []  # list of runtimes for each batch
    image_counts = []  # list of number of images in each batch

    dataset = get_dataset(images_dir, annotation_path, batch_size, image_ids,
                          use_synthetic=use_synthetic, coco=coco, image_shape=image_shape)
    iter_times = []
    statistics = {}
    predictions = {}
    for i, batch_feats in enumerate(dataset):
        start_time = time.time()
        batch_preds = graph_func(batch_feats)
        iter_time = time.time() - start_time
        iter_times.append(iter_time)
        for key in batch_preds.keys():
            if key not in predictions:
                predictions[key] = [batch_preds[key]]
            else:
                predictions[key].append(batch_preds[key])
        if i % display_every == 1:
            print("Step {}/{}, iter_time={}".format(i - 1, num_images//batch_size, iter_time))
    iter_times = iter_times[num_warmup_iterations:]
    statistics['throughput'] = batch_size / np.mean(np.array(iter_times))
    statistics['throughput median'] = batch_size / np.median(np.array(iter_times))
    return statistics, predictions, image_ids

def eval_model(predictions, image_ids, annotation_path, tmp_dir):

    for old_key in list(predictions.keys()):
        if old_key in NAME_MAP:
            new_key = NAME_MAP[old_key]
            predictions[new_key] = predictions[old_key]
            del predictions[old_key]
    for key in predictions:
        predictions[key] = [t.numpy() for t in predictions[key]]
        predictions[key] = np.vstack(predictions[key])
        if key == 'num_detections':
            predictions[key] = predictions[key].ravel()
        
    subprocess.call(['mkdir', '-p', tmp_dir])
    coco = COCO(annotation_file=annotation_path)
    coco_detections = []
    for i, image_id in enumerate(image_ids):
        coco_img = coco.imgs[image_id]
        image_width = coco_img['width']
        image_height = coco_img['height']

        for j in range(int(predictions['num_detections'][i])):
            bbox = preds['boxes'][i][j]
            y1, x1, y2, x2 = list(bbox)
            bbox_coco_fmt = [
                x1 * image_width,  # x0
                y1 * image_height,  # x1
                (x2 - x1) * image_width,  # width
                (y2 - y1) * image_height,  # height
            ]
            coco_detection = {
                'image_id': image_id,
                'category_id': int(preds['classes'][i][j]),
                'bbox': [int(coord) for coord in bbox_coco_fmt],
                'score': float(preds['scores'][i][j])
            }

            coco_detections.append(coco_detection)
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

    return {'map': eval.stats[0]}

def setup(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if not gpu_mem_cap:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_cap)])

            print(len(gpus), "Physical GPU's")
        except RuntimeError as e:
            print(e)
    
def get_trt_conversion_params(use_trt,
                              max_workspace_size_bytes,
                              precision_mode,
                              minimum_segment_size,
                              is_dynamic_op,
                              max_batch_size):
    if use_trt:
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(
            max_workspace_size_bytes=max_workspace_size_bytes)
        conversion_params = conversion_params._replace(precision_mode=precision_mode)
        conversion_params = conversion_params._replace(minimum_segment_size=minimum_segment_size)
        conversion_params = conversion_params._replace(is_dynamic_op=is_dynamic_op)
        conversion_params = conversion_params._replace(use_calibration=precision_mode=='INT8')
        conversion_params = conversion_params._replace(max_batch_size=max_batch_size)
        return conversion_params
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--use_trt', action='store_true',
        help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8'], default='FP32',
        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--data_dir', type=str, default='/data/coco/coco-2017/coco2017/val2017', 
        help='Directory of input images')
    parser.add_argument('--annotation_path', type=str, default='/data/coco/coco-2017/coco2017/annotations/instances_val2017.json', 
        help='Path that contains COCO annotations')
    parser.add_argument('--saved_model_dir', type=str, default='./mask_rcnn_tpu',
        help='Path to SavedModel directory.')
    parser.add_argument('--gpu_mem_cap', type=int, default=0,
        help='If > 0, gpu memory maximum will be set instead of allowing growth. In MB.')
    parser.add_argument('--calib_images_dir', type=str,
        help='Directory containing calibration images')
    parser.add_argument('--num_calib_images', type=int, default=16,
        help='Number of images to get calibration statistics from')
    parser.add_argument('--num_images', type=int, default=4096,
        help='Number of images to benchmark on.')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
        help='Number of batches to exclude from timing.')
    parser.add_argument('--display_every', type=int, default=100,
        help='How frequently to display inference progress')
    parser.add_argument('--input_size', type=int, default=None,
        help='Size of a single side of the input image size.')
    parser.add_argument('--cache', action='store_true',
        help='If set, a previously generated saved model will be used if available')
    parser.add_argument('--use_synthetic', action='store_true',
        help='Whether to use synthetic data as opposed to real data.')
    parser.add_argument('--batch_size', type=int, default=1,
        help='Number of images per batch.')
    args = parser.parse_args()

    conversion_params = get_trt_conversion_params(args.use_trt,
                                                  max_workspace_size_bytes=1<<30,
                                                  precision_mode=args.precision,
                                                  minimum_segment_size=2,
                                                  is_dynamic_op=True,
                                                  max_batch_size=args.batch_size)
    setup(args.gpu_mem_cap)
    tmp_dir = '.tmp_{}_{}_{}'.format(args.saved_model_dir,
                               'tf' if args.use_trt else 'trt',
                               args.precision)

    graph_func = get_model_func(saved_model_dir=args.saved_model_dir,
                                use_trt=args.use_trt,
                                tmp_dir=tmp_dir,
                                cache=args.cache,
                                annotation_path=args.annotation_path,
                                num_calib_images=args.num_calib_images,
                                calib_batch_size=1,
                                images_dir=args.data_dir,
                                image_shape=(args.input_size,args.input_size),
                                conversion_params=conversion_params)
    if not args.cache:
        if os.path.exists(tmp_dir):
            subprocess.call(['rm', '-rf', tmp_dir])

    stats, preds, image_ids = benchmark_model(graph_func,
                                              args.data_dir,
                                              batch_size=args.batch_size,
                                              image_shape=(args.input_size,args.input_size),
                                              annotation_path=args.annotation_path,
                                              num_images=args.num_images,
                                              output_path=None,
                                              display_every=args.display_every,
                                              use_synthetic=False,
                                              num_warmup_iterations=10,)
    print('Precision for next stats: ', args.precision)
    print('Use trt?: ', args.use_trt)
    print('Model: ', args.saved_model_dir)
    print('Batch size: ', args.batch_size)
    pprint.pprint(stats)
    if not args.use_synthetic:
        map_ = eval_model(preds, image_ids, args.annotation_path, tmp_dir=tmp_dir)
        print(map_)

