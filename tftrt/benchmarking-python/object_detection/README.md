# TensorRT / TensorFlow Object Detection

This package provides scripts to benchmark performance and accuracy of
object detection models using TensorRT integration in TensorFlow 2.0.

The input to the script is a SavedModel direcotry that includes
a pre-trained model. Passing a data directory (e.g. COCO) is also necessary in case of
validating accuracy (e.g. mAP).

<a name="setup"></a>
## Setup

Install object detection dependencies (from tftrt/benchmarking-python/object_detection)

```bash
git submodule update --init
../helper_scripts/install_pycocotools.sh
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
    --use_tftrt \
    --precision FP16
```

## Ready to Use Scripts:

#### 1. faster_rcnn_resnet50_coco
```bash
# Tensorflow - FP32
./models/faster_rcnn_resnet50_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/faster_rcnn_resnet50_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/faster_rcnn_resnet50_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/faster_rcnn_resnet50_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/faster_rcnn_resnet50_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/faster_rcnn_resnet50_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```

#### 2. ssd_inception_v2_coco
```bash
# Tensorflow - FP32
./models/ssd_inception_v2_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_inception_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/ssd_inception_v2_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_inception_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/ssd_inception_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/ssd_inception_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```

#### 3. ssd_mobilenet_v1_coco
```bash
# Tensorflow - FP32
./models/ssd_mobilenet_v1_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_mobilenet_v1_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/ssd_mobilenet_v1_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_mobilenet_v1_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/ssd_mobilenet_v1_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/ssd_mobilenet_v1_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```

#### 4. ssd_mobilenet_v1_fpn_coco
```bash
# Tensorflow - FP32
./models/ssd_mobilenet_v1_fpn_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_mobilenet_v1_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/ssd_mobilenet_v1_fpn_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_mobilenet_v1_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/ssd_mobilenet_v1_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/ssd_mobilenet_v1_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```

#### 5. ssd_mobilenet_v2_coco
```bash
# Tensorflow - FP32
./models/ssd_mobilenet_v2_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/ssd_mobilenet_v2_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/ssd_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/ssd_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```

#### 6. ssd_resnet_50_fpn_coco
```bash
# Tensorflow - FP32
./models/ssd_resnet_50_fpn_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_resnet_50_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/ssd_resnet_50_fpn_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssd_resnet_50_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/ssd_resnet_50_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/ssd_resnet_50_fpn_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```

#### 7. ssdlite_mobilenet_v2_coco
```bash
# Tensorflow - FP32
./models/ssdlite_mobilenet_v2_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssdlite_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models

# TF-TRT - FP32
./models/ssdlite_mobilenet_v2_coco/run_inference.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"    

# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./models/ssdlite_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP32"

# TF-TRT - FP16
./models/ssdlite_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="FP16"

# TF-TRT - INT8
./models/ssdlite_mobilenet_v2_coco/run_inference.sh \
    --use_xla \
    --data_dir=/data/coco2017 --input_saved_model_dir=/models \
    --use_tftrt --precision="INT8"
```
