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

## Ready to Use Scripts:

#### 1. faster_rcnn_resnet50_coco
```bash
# Tensorflow - FP32
./scripts/faster_rcnn_resnet50_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/faster_rcnn_resnet50_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/faster_rcnn_resnet50_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/faster_rcnn_resnet50_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/faster_rcnn_resnet50_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/faster_rcnn_resnet50_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```

#### 2. ssd_inception_v2_coco
```bash
# Tensorflow - FP32
./scripts/ssd_inception_v2_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_inception_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/ssd_inception_v2_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_inception_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/ssd_inception_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/ssd_inception_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```

#### 3. ssd_mobilenet_v1_coco
```bash
# Tensorflow - FP32
./scripts/ssd_mobilenet_v1_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_mobilenet_v1_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/ssd_mobilenet_v1_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_mobilenet_v1_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/ssd_mobilenet_v1_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/ssd_mobilenet_v1_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```

#### 4. ssd_mobilenet_v1_fpn_coco
```bash
# Tensorflow - FP32
./scripts/ssd_mobilenet_v1_fpn_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_mobilenet_v1_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/ssd_mobilenet_v1_fpn_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_mobilenet_v1_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/ssd_mobilenet_v1_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/ssd_mobilenet_v1_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```

#### 5. ssd_mobilenet_v2_coco
```bash
# Tensorflow - FP32
./scripts/ssd_mobilenet_v2_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/ssd_mobilenet_v2_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/ssd_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/ssd_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```

#### 6. ssd_resnet_50_fpn_coco
```bash
# Tensorflow - FP32
./scripts/ssd_resnet_50_fpn_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_resnet_50_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/ssd_resnet_50_fpn_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssd_resnet_50_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/ssd_resnet_50_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/ssd_resnet_50_fpn_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```

#### 7. ssdlite_mobilenet_v2_coco
```bash
# Tensorflow - FP32
./scripts/ssdlite_mobilenet_v2_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models

# Tensorflow - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssdlite_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models
    
# TF-TRT - FP32
./scripts/ssdlite_mobilenet_v2_coco.sh \
    --use_xla --no_tf32 \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"    
    
# TF-TRT - TF32 (identical to FP32 on an NVIDIA Turing GPU or older)
./scripts/ssdlite_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP32"
    
# TF-TRT - FP16
./scripts/ssdlite_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="FP16"
    
# TF-TRT - INT8
./scripts/ssdlite_mobilenet_v2_coco.sh \
    --use_xla \
    --data_dir=/data/coco2017 --model_dir=/models \
    --use_tftrt --tftrt_precision="INT8"
```
