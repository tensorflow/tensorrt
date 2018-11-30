# TensorRT / TensorFlow Object Detection

This package demonstrated object detection using TensorRT integration in TensorFlow. 
It includes utilities for accuracy and performance benchmarking, along with 
utilities for model construction and optimization.

* [Setup](#setup)
* [Download](#od_download)
* [Optimize](#od_optimize)
* [Benchmark](#od_benchmark)
* [Test](#od_test)

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
### Download
```python
from tftrt.examples.object_detection import download_model

config_path, checkpoint_path = download_model('ssd_mobilenet_v1_coco', output_dir='models')
# help(download_model) for more
```

<a name="od_optimize"></a>
### Optimize

```python
from tftrt.examples.object_detection import optimize_model

frozen_graph = optimize_model(
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    use_trt=True,
    precision_mode='FP16'
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
    images_dir=images_dir, 
    annotation_path=annotation_path
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
  "source_model": {
    "model_name": "ssd_inception_v2_coco",
    "output_dir": "models"
  },
  "optimization_config": {
    "use_trt": true,
    "precision_mode": "FP16",
    "force_nms_cpu": true,
    "replace_relu6": true,
    "remove_assert": true,
    "override_nms_score_threshold": 0.3,
    "max_batch_size": 1
  },
  "benchmark_config": {
    "images_dir": "coco/val2017",
    "annotation_path": "coco/annotations/instances_val2017.json",
    "batch_size": 1,
    "image_shape": [600, 600],
    "num_images": 4096,
    "output_path": "stats/ssd_inception_v2_coco_trt_fp16.json"
  },
  "assertions": [
    "statistics['map'] > (0.268 - 0.005)"
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
