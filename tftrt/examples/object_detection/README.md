# TensorRT / TensorFlow Object Detection

This package demonstrated object detection using TensorRT integration in TensorFlow. 
It includes utilities for accuracy and performance benchmarking, along with 
utilities for model construction and optimization.

* [Setup](#setup)
* [Download and build](#od_download)
* [Optimize](#od_optimize)
* [Benchmark](#od_benchmark)
* [Test](#od_test)
* [Verified Models](#od_verified_models)

<a name="setup"></a>
## Setup

1. Install object detection dependencies (from tftrt/examples/object_detection)

```bash
git submodule update --init
./install_dependencies.sh
```

2. Ensure you've installed the tftrt package (from root folder of repository)

```bash
python setup.py install --user
```

<a name="od"></a>
## Object Detection

<a name="od_download"></a>
### Download and build
```python
from tftrt.examples.object_detection import build_model

frozen_graph = build_model(
    model_name="ssd_resnet_50_fpn_coco",
    input_dir="/models/object_detection/combined_nms_enabled",
    batch_size=8,
    override_nms_score_threshold=0.3,
)
# help(build_model) for more
```

<a name="od_optimize"></a>
### Optimize

```python
from tftrt.examples.object_detection import optimize_model

frozen_graph = optimize_model(
    frozen_graph,
    use_trt=True,
    precision_mode="INT8",
    calib_images_dir="/data/coco-2017/train2017",
    num_calib_images=8,
    calib_batch_size=8,
    calib_image_shape=[640, 640],
    max_workspace_size_bytes=17179869184,
)
# help(optimize_model) for other parameters
```

<a name="od_benchmark"></a>
### Benchmark

First, we download the validation dataset

```python
from tftrt.examples.object_detection import download_dataset

images_dir, annotation_path = download_dataset('val2014', output_dir='dataset')
# help(download_dataset) for more
```

Next, we run inference over the dataset to benchmark the optimized model

```python
from tftrt.examples.object_detection import benchmark_model

statistics = benchmark_model(
    frozen_graph=frozen_graph, 
    images_dir="/data/coco2017/val2017",
    annotation_path="/data/coco2017/annotations/instances_val2017.json",
    batch_size=8,
    image_shape=[640, 640],
    num_images=4096,
    output_path="stats/ssd_resnet_50_fpn_coco_trt_int8.json"
)
# help(benchmark_model) for more parameters
```

<a name="od_test"></a>
### Test
To simplify evaluation of different models with different optimization parameters
we include a ``test`` function that ingests a JSON file containing test arguments
and combines the model download, optimization, and benchmark steps.  Below is an
example JSON file, call it ``my_test.json``

```json
{
  "model_config": {
    "model_name": "ssd_resnet_50_fpn_coco",
    "input_dir": "/models/object_detection/combined_nms_enabled",
    "batch_size": 8,
    "override_nms_score_threshold": 0.3
  },
  "optimization_config": {
    "use_trt": true,
    "precision_mode": "INT8",
    "calib_images_dir": "/data/coco2017/train2017",
    "num_calib_images": 8,
    "calib_batch_size": 8,
    "calib_image_shape": [640, 640],
    "max_workspace_size_bytes": 17179869184
  },
  "benchmark_config": {
    "images_dir": "/data/coco2017/val2017",
    "annotation_path": "/data/coco2017/annotations/instances_val2017.json",
    "batch_size": 8,
    "image_shape": [640, 640],
    "num_images": 4096,
    "output_path": "stats/ssd_resnet_50_fpn_coco_trt_int8.json"
  },
  "assertions": [
    "statistics['map'] > (0.277 - 0.01)"
  ]
}
```

We execute the test using the ``test`` python function

```python
from tftrt.examples.object_detection import test

test('my_test.json')
# help(test) for more details
```

Alternatively, we can directly call the object_detection.test module, which
is configured to execute this function by default.

```shell
python -m tftrt.examples.object_detection.test my_test.json
```

For the example configuration shown above, the following steps will be performed

1. Downloads ssd_inception_v2_coco
2. Optimizes with TensorRT and FP16 precision
3. Benchmarks against the MSCOCO 2017 validation dataset
4. Asserts that the MAP is greater than some reference value

<a name="od_verified_models"></a>
### Verified Models
We have verified the accuracy and performance of the following models that are supported by the package:

    'ssd_mobilenet_v1_coco'
    'ssd_mobilenet_v1_fpn_coco'
    'ssd_mobilenet_v2_coco'
    'ssdlite_mobilenet_v2_coco'
    'ssd_inception_v2_coco'
    'ssd_resnet_50_fpn_coco'
    'faster_rcnn_resnet50_coco'
    'faster_rcnn_nas'
    'mask_rcnn_resnet50_atrous_coco'
