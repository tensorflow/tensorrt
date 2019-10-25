# TensorRT / TensorFlow Object Detection

This package provides scripts to benchmark performance and accuracy of
object detection models using TensorRT integration in TensorFlow 2.0.

The input to the script is a SavedModel direcotry that includes
a pre-trained model. Passing a data directory (e.g. COCO) is also necessary in case of
validating accuracy (e.g. mAP).

<a name="setup"></a>
## Setup

Install object detection dependencies (from tftrt/examples/object_detection)

```bash
git submodule update --init
./install_dependencies.sh
```

<a name="od"></a>
## Usage

Run `python object_detection.py --help` to see what arguments are available.

Example:

```
python object_detection.py \
    --saved_model_dir input_saved_model \
    --data_dir /data/coco/val2017 \
    --annotation_path /data/coco/annotations/instances_val2017.json \
    --input_size 640 \
    --batch_size 8 \
    --use_trt \
    --precision FP16
```
