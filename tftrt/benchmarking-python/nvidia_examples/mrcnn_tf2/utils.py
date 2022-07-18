import argparse
import copy
import glob
import math
import multiprocessing
import os
import sys
import time

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorflow as tf


def _get_source_id_from_encoded_image(parsed_tensors):
    return tf.strings.as_string(
        tf.strings.to_hash_bucket_fast(
            parsed_tensors['image/encoded'], 2**63 - 1
        )
    )


class TfExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/source_id': tf.io.FixedLenFeature((), tf.string),
            'image/height': tf.io.FixedLenFeature((), tf.int64),
            'image/width': tf.io.FixedLenFeature((), tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/area': tf.io.VarLenFeature(tf.float32),
            'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
            'image/object/mask': tf.io.VarLenFeature(tf.string),
        }

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_masks(self, parsed_tensors):
        """Decode a set of PNG masks to the tf.float32 tensors."""

        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8),
                axis=-1
            )
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors['image/height']
        width = parsed_tensors['image/width']
        masks = parsed_tensors['image/object/mask']
        return tf.cond(
            pred=tf.greater(tf.size(input=masks), 0),
            true_fn=lambda: tf.
            map_fn(_decode_png_mask, masks, dtype=tf.float32),
            false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32)
        )

    def decode(self, serialized_example):
        """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - image: a uint8 tensor of shape [None, None, 3].
        - source_id: a string scalar tensor.
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features
        )
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=''
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0
                    )

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        is_crowd = tf.cast(
            parsed_tensors['image/object/is_crowd'], dtype=tf.bool
        )

        masks = self._decode_masks(parsed_tensors)

        source_id = tf.cond(
            pred=tf.greater(
                tf.strings.length(input=parsed_tensors['image/source_id']), 0
            ),
            true_fn=lambda: parsed_tensors['image/source_id'],
            false_fn=lambda: _get_source_id_from_encoded_image(parsed_tensors)
        )

        decoded_tensors = {
            'image':
                image,
            'source_id':
                source_id,
            'height':
                parsed_tensors['image/height'],
            'width':
                parsed_tensors['image/width'],
            'groundtruth_classes':
                parsed_tensors['image/object/class/label'],
            'groundtruth_is_crowd':
                is_crowd,
            'groundtruth_area':
                parsed_tensors['image/object/area'],
            'groundtruth_boxes':
                boxes,
            'groundtruth_instance_masks':
                masks,
            'groundtruth_instance_masks_png':
                parsed_tensors['image/object/mask'],
        }

        return decoded_tensors


def scale_boxes(boxes, y_scale, x_scale):
    """scale box coordinates in x and y dimensions.

    Args:
      boxlist: BoxList holding N boxes
      y_scale: (float) scalar tensor
      x_scale: (float) scalar tensor
      scope: name scope.

    Returns:
      boxlist: BoxList holding N boxes
    """

    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=1
    )
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxes = tf.concat([y_min, x_min, y_max, x_max], 1)
    return scaled_boxes


def process_boxes_classes_indices_for_training(data, use_category):
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

    if not use_category:
        classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

    return boxes, classes


def create_dummy_labels(args):
    labels = {
        'cropped_gt_masks':
            tf.zeros(
                (args.max_num_instances, args.gt_mask_size, args.gt_mask_size),
                dtype=tf.float32
            ),
        'gt_boxes':
            tf.zeros((args.max_num_instances, 4), dtype=tf.float32),
        'gt_classes':
            tf.zeros((args.max_num_instances, 1), dtype=tf.float32),
    }
    x, y = args.image_size

    x = x // 2**args.min_level
    y = y // 2**args.min_level

    for level in range(args.min_level, args.max_level + 1):
        labels['score_targets_%d' % level] = tf.zeros((x, y, args.num_anchors),
                                                      dtype=tf.int32)
        labels['box_targets_%d' % level
              ] = tf.zeros((x, y, args.num_anchors * 4), dtype=tf.float32)
        x = x // 2
        y = y // 2

    return labels


def prepare_labels_for_eval(data, max_num_instances):
    """Create labels dict for infeed from data of tf.Example."""
    image = data['image']

    height, width = tf.shape(image)[0], tf.shape(image)[1]

    boxes = data['groundtruth_boxes']

    classes = tf.cast(data['groundtruth_classes'], dtype=tf.float32)

    num_labels = tf.shape(classes)[0]

    boxes = pad_to_fixed_size(boxes, -1, [max_num_instances, 4])
    classes = pad_to_fixed_size(classes, -1, [max_num_instances, 1])

    is_crowd = tf.cast(data['groundtruth_is_crowd'], dtype=tf.float32)
    is_crowd = pad_to_fixed_size(is_crowd, 0, [max_num_instances, 1])

    labels = dict()

    labels['width'] = width
    labels['height'] = height
    labels['groundtruth_boxes'] = boxes
    labels['groundtruth_classes'] = classes
    labels['num_groundtruth_labels'] = num_labels
    labels['groundtruth_is_crowd'] = is_crowd

    return labels


def dataset_parser(value, args):
    """Parse data to a fixed dimension input image and learning targets.

    Args:
    value: A dictionary contains an image and groundtruth annotations.

    Returns:
    features: a dictionary that contains the image and auxiliary
      information. The following describes {key: value} pairs in the
      dictionary.
      image: Image tensor that is preproessed to have normalized value and
        fixed dimension [image_size, image_size, 3]
      image_info: image information that includes the original height and
        width, the scale of the proccessed image to the original image, and
        the scaled height and width.
      source_ids: Source image id. Default value -1 if the source id is
        empty in the groundtruth annotation.
    labels: a dictionary that contains auxiliary information plus (optional)
      labels. The following describes {key: value} pairs in the dictionary.
      `labels` is only for training.
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of objectiveness score at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      gt_boxes: Groundtruth bounding box annotations. The box is represented
         in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
         fixed dimension [args.max_num_instances, 4].
      gt_classes: Groundtruth classes annotations. The tennsor is padded
        with -1 to the fixed dimension [args.max_num_instances].
      cropped_gt_masks: groundtrugh masks cropped by the bounding box and
        resized to a fixed size determined by params.gt_mask_size
      regenerate_source_id: `bool`, if True TFExampleParser will use hashed
        value of `image/encoded` for `image/source_id`.
    """
    example_decoder = TfExampleDecoder()

    with tf.xla.experimental.jit_scope(compile_ops=True):

        with tf.name_scope('parser'):

            data = example_decoder.decode(value)
            data['groundtruth_is_crowd'] = process_groundtruth_is_crowd(data)
            image = tf.image.convert_image_dtype(
                data['image'], dtype=tf.float32
            )
            source_id = process_source_id(data['source_id'])
            boxes, classes = process_boxes_classes_indices_for_training(
                data, use_category=args.use_category
            )

            image, image_info, boxes = preprocess_image(
                image,
                boxes=boxes,
                image_size=args.image_size,
                max_level=args.max_level,
            )
            boxes = pad_to_fixed_size(boxes, -1, [args.max_num_instances, 4])
            classes = pad_to_fixed_size(
                classes, -1, [args.max_num_instances, 1]
            )

            features = {
                'source_ids': source_id,
                'images': image,
                'image_info': image_info,
            }

            # Additional labels needed for training, that are expected by SavedModel
            additional_labels = create_dummy_labels(args)
            features.update(additional_labels)

            labels = prepare_labels_for_eval(data, args.max_num_instances)
            labels['source_ids'] = source_id

            return features, labels


def normalize_image(image):
    """Normalize the image.

    Args:
    image: a tensor of shape [height, width, 3] in dtype=tf.float32.

    Returns:
    normalized_image: a tensor which has the same shape and dtype as image,
      with pixel values normalized.
    """
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.reshape(offset, shape=(1, 1, 3))

    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.reshape(scale, shape=(1, 1, 3))

    normalized_image = (image-offset) / scale

    return normalized_image


def resize_and_pad(image, target_size, stride, boxes):
    """Resize and pad images, boxes and masks.

    Resize and pad images, (optionally boxes and masks) given the desired output
    size of the image and stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `target_size`.
    2. Pad the rescaled image such that the height and width of the image become
     the smallest multiple of the stride that is larger or equal to the desired
     output diemension.

    Args:
    image: an image tensor of shape [original_height, original_width, 3].
    target_size: a tuple of two integers indicating the desired output
      image size. Note that the actual output size could be different from this.
    stride: the stride of the backbone network. Each of the output image sides
      must be the multiple of this.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    masks: (Optional) a tensor of shape [num_masks, height, width]
      representing the object masks. Note that the size of the mask is the
      same as the image.

    Returns:
    image: the processed image tensor after being resized and padded.
    image_info: a tensor of shape [5] which encodes the height, width before
      and after resizing and the scaling factor.
    boxes: None or the processed box tensor after being resized and padded.
      After the processing, boxes will be in the absolute coordinates w.r.t.
      the scaled image.
    masks: None or the processed mask tensor after being resized and padded.
    """

    input_height, input_width, _ = tf.unstack(
        tf.cast(tf.shape(input=image), dtype=tf.float32), axis=0
    )

    target_height, target_width = target_size

    scale_if_resize_height = target_height / input_height
    scale_if_resize_width = target_width / input_width

    scale = tf.minimum(scale_if_resize_height, scale_if_resize_width)

    scaled_height = tf.cast(scale * input_height, dtype=tf.int32)
    scaled_width = tf.cast(scale * input_width, dtype=tf.int32)

    image = tf.image.resize(
        image, [scaled_height, scaled_width],
        method=tf.image.ResizeMethod.BILINEAR
    )

    padded_height = int(math.ceil(target_height * 1.0 / stride) * stride)
    padded_width = int(math.ceil(target_width * 1.0 / stride) * stride)

    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_height, padded_width
    )
    image.set_shape([padded_height, padded_width, 3])

    image_info = tf.stack([
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32), 1.0 / scale, input_height,
        input_width
    ])

    scaled_boxes = scale_boxes(boxes, scaled_height, scaled_width)

    return image, image_info, scaled_boxes


def pad_to_fixed_size(data, pad_value, output_shape):
    """Pad data to a fixed length at the first dimension.

    Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

    Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
    """
    max_num_instances = output_shape[0]
    dimension = output_shape[1]

    data = tf.reshape(data, [-1, dimension])
    num_instances = tf.shape(input=data)[0]

    pad_length = max_num_instances - num_instances

    paddings = pad_value * tf.ones([pad_length, dimension])

    padded_data = tf.reshape(tf.concat([data, paddings], axis=0), output_shape)
    return padded_data


def preprocess_image(image, boxes, image_size, max_level):
    image = normalize_image(image)

    # Scaling and padding.
    image, image_info, boxes = resize_and_pad(
        image=image,
        target_size=image_size,
        stride=2**max_level,
        boxes=boxes,
    )
    return image, image_info, boxes


def process_groundtruth_is_crowd(data):
    return tf.cond(
        pred=tf.greater(tf.size(input=data['groundtruth_is_crowd']), 0),
        true_fn=lambda: data['groundtruth_is_crowd'],
        false_fn=lambda: tf.
        zeros_like(data['groundtruth_classes'], dtype=tf.bool)
    )


def process_source_id(source_id):
    """Processes source_id to the right format."""
    if source_id.dtype == tf.string:
        source_id = tf.cast(tf.strings.to_number(source_id), tf.int64)

    with tf.control_dependencies([source_id]):
        source_id = tf.cond(
            pred=tf.equal(tf.size(input=source_id), 0),
            true_fn=lambda: tf.cast(tf.constant(-1), tf.int64),
            false_fn=lambda: tf.identity(source_id)
        )
    source_id = tf.expand_dims(source_id, -1)

    return source_id


# %%%%%%%%%%%%%%%%%%%%%%%%%%% EVALUATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def process_predictions(predictions):
    """ Process the model predictions for COCO eval.
    Converts boxes from [y1, x1, y2, x2] to [x1, y1, w, h] and scales them by image scale.
    Flattens source_ids

    Args:
        predictions (dict): Predictions returned by model

    Returns:
        Converted prediction.
    """
    image_info = predictions['image_info']
    detection_boxes = predictions['detection_boxes']

    for pred_id, box_id in np.ndindex(*detection_boxes.shape[:2]):
        # convert from [y1, x1, y2, x2] to [x1, y1, w, h] * scale
        scale = image_info[pred_id, 2]
        y1, x1, y2, x2 = detection_boxes[pred_id, box_id, :]

        new_box = np.array([x1, y1, x2 - x1, y2 - y1]) * scale

        detection_boxes[pred_id, box_id, :] = new_box

    # flatten source ids
    predictions['source_ids'] = predictions['source_ids'].flatten()

    return predictions


class MaskCOCO(COCO):
    """COCO object for mask evaluation.
    """

    def reset(self, dataset):
        """Reset the dataset and groundtruth data index in this object.

        Args:
          dataset: dict of groundtruth data. It should has similar structure as the
            COCO groundtruth JSON file. Must contains three keys: {'images',
              'annotations', 'categories'}.
            'images': list of image information dictionary. Required keys: 'id',
              'width' and 'height'.
            'annotations': list of dict. Bounding boxes and segmentations related
              information. Required keys: {'id', 'image_id', 'category_id', 'bbox',
                'iscrowd', 'area', 'segmentation'}.
            'categories': list of dict of the category information.
              Required key: 'id'.
            Refer to http://cocodataset.org/#format-data for more details.

        Raises:
          AttributeError: If the dataset is empty or not a dict.
        """
        assert dataset, 'Groundtruth should not be empty.'
        assert isinstance(
            dataset, dict
        ), 'annotation file format {} not supported'.format(type(dataset))
        self.anns, self.cats, self.imgs = dict(), dict(), dict()
        self.dataset = copy.deepcopy(dataset)
        self.createIndex()

    def loadRes(self, detection_results):
        """Load result file and return a result api object.

        Args:
          detection_results: a dictionary containing predictions results.
        Returns:
          res: result MaskCOCO api object
        """
        res = MaskCOCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        predictions = self.load_predictions(detection_results)
        assert isinstance(
            predictions, list
        ), 'results in not an array of objects'
        if predictions:
            image_ids = [pred['image_id'] for pred in predictions]
            assert set(image_ids) == (set(image_ids) & set(self.getImgIds())), \
                'Results do not correspond to current coco set'

            if (predictions and 'bbox' in predictions[0] and
                    predictions[0]['bbox']):
                res.dataset['categories'] = copy.deepcopy(
                    self.dataset['categories']
                )
                for idx, pred in enumerate(predictions):
                    bb = pred['bbox']
                    x1, x2, y1, y2 = [
                        bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]
                    ]
                    if 'segmentation' not in pred:
                        pred['segmentation'] = [[
                            x1, y1, x1, y2, x2, y2, x2, y1
                        ]]
                    pred['area'] = bb[2] * bb[3]
                    pred['id'] = idx + 1
                    pred['iscrowd'] = 0

            res.dataset['annotations'] = predictions

        res.createIndex()
        return res

    def load_predictions(self, detection_results):
        """Create prediction dictionary list from detection and mask results.

        Args:
          detection_results: a dictionary containing numpy arrays which corresponds
            to prediction results.
          include_mask: a boolean, whether to include mask in detection results.
          is_image_mask: a boolean, where the predict mask is a whole image mask.

        Returns:
          a list of dictionary including different prediction results from the model
            in numpy form.
        """
        predictions = []
        for i, image_id in enumerate(detection_results['source_ids']):

            for box_index in range(int(detection_results['num_detections'][i])):
                prediction = {
                    'image_id':
                        int(image_id),
                    'bbox':
                        detection_results['detection_boxes'][i]
                        [box_index].tolist(),
                    'score':
                        detection_results['detection_scores'][i][box_index],
                    'category_id':
                        int(
                            detection_results['detection_classes'][i][box_index]
                        ),
                }

                predictions.append(prediction)

        return predictions


class EvaluationMetric:
    """COCO evaluation metric class."""

    def __init__(self):
        """Constructs COCO evaluation class.

        The class provides the interface to metrics_fn in TPUEstimator. The
        _evaluate() loads a JSON file in COCO annotation format as the
        groundtruths and runs COCO evaluation.

        Args:
          filename: Ground truth JSON file name. If filename is None, use
            groundtruth data passed from the dataloader for evaluation.
          include_mask: boolean to indicate whether or not to include mask eval.
        """
        self.metric_id = 0
        self.coco_gt = MaskCOCO()

    def predict_metric_fn(self, predictions, groundtruth_data):
        """Generates COCO metrics."""
        image_ids = list(set(predictions['source_ids']))
        self.coco_gt.reset(groundtruth_data)
        coco_dt = self.coco_gt.loadRes(predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = coco_eval.stats
        metrics = metrics.astype(np.float32)
        return metrics[self.metric_id]


def _denormalize_to_coco_bbox(bbox, height, width):
    """Denormalize bounding box.

    Args:
      bbox: numpy.array[float]. Normalized bounding box. Format: ['ymin', 'xmin',
        'ymax', 'xmax'].
      height: int. image height.
      width: int. image width.

    Returns:
      [x, y, width, height]
    """
    y1, x1, y2, x2 = bbox
    y1 *= height
    x1 *= width
    y2 *= height
    x2 *= width
    box_height = y2 - y1
    box_width = x2 - x1
    return [float(x1), float(y1), float(box_width), float(box_height)]


def _extract_image_info(prediction, b):
    return {
        'id': int(prediction['source_ids'][b]),
        'width': int(prediction['width'][b]),
        'height': int(prediction['height'][b]),
    }


def _extract_bbox_annotation(target, b, obj_i):
    """Constructs COCO format bounding box annotation."""
    height = target['height'][b]
    width = target['width'][b]

    bbox = _denormalize_to_coco_bbox(
        target['groundtruth_boxes'][b][obj_i, :], height, width
    )

    if 'groundtruth_area' in target:
        area = float(target['groundtruth_area'][b][obj_i])

    else:
        # Using the box area to replace the polygon area. This value will not affect
        # real evaluation but may fail the unit test.
        area = bbox[2] * bbox[3]

    annotation = {
        'id': b*1000 + obj_i,  # place holder of annotation id.
        'image_id': int(target['source_ids'][b]),  # source_id,
        'category_id': int(target['groundtruth_classes'][b][obj_i]),
        'bbox': bbox,
        'iscrowd': int(target['groundtruth_is_crowd'][b][obj_i]),
        'area': area,
        'segmentation': [],
    }
    return annotation


def _extract_categories(annotations):
    """Extract categories from annotations."""
    categories = {}
    for anno in annotations:
        category_id = int(anno['category_id'])
        categories[category_id] = {'id': category_id}
    return list(categories.values())


def extract_coco_groundtruth(target):
    """Extract COCO format groundtruth.

    Args:
      target: dictionary of batch of expected result. the first dimension
        each element is the batch.

    Returns:
      Tuple of (images, annotations).
      images: list[dict].Required keys: 'id', 'width' and 'height'. The values are
        image id, width and height.
      annotations: list[dict]. Required keys: {'id', 'source_ids', 'category_id',
        'bbox', 'iscrowd'}. The 'id' value is the annotation id
        and can be any **positive** number (>=1).
        Refer to http://cocodataset.org/#format-data for more details.
    Raises:
      ValueError: If any groundtruth fields is missing.
    """
    required_fields = [
        'source_ids', 'width', 'height', 'num_groundtruth_labels',
        'groundtruth_boxes', 'groundtruth_classes'
    ]
    for key in required_fields:
        if key not in target.keys():
            raise ValueError(
                'Missing groundtruth field: "{}" keys: {}'.format(
                    key, target.keys()
                )
            )

    images = []
    annotations = []
    for b in range(target['source_ids'].shape[0]):
        # Constructs image info.
        image = _extract_image_info(target, b)
        images.append(image)

        # Constructs annotations.
        num_labels = int(target['num_groundtruth_labels'][b])
        for obj_i in range(num_labels):
            annotation = _extract_bbox_annotation(target, b, obj_i)

            annotations.append(annotation)
    return images, annotations


def create_coco_format_dataset(
    images, annotations, regenerate_annotation_id=True
):
    """Creates COCO format dataset with COCO format images and annotations."""
    if regenerate_annotation_id:
        for i in range(len(annotations)):
            # WARNING: The annotation id must be positive.
            annotations[i]['id'] = i + 1

    categories = _extract_categories(annotations)
    dataset = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    return dataset
