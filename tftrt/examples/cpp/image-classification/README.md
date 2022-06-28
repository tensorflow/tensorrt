# TF-TRT example for conversion in C++

## Introduction

This directory contains example code to demonstrate TF-TRT conversion using the C++ API.

### Acknowledgment

The MNIST inference example is based on https://github.com/bmzhao/saved-model-example

## How to run

### Build TF
```
git clone https://github.com/tensorflow/tensorrt.git
git clone https://github.com/tensorflow/tensorflow.git tensorflow-source
mkdir bazel-cache
docker run --gpus=all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace -w /workspace -v $PWD/bazel-cache:/root/.cache/bazel nvcr.io/nvidia/tensorflow:21.06-tf2-py3

# Inside the container
cp /opt/tensorflow/nvbuild* /opt/tensorflow/bazel_build.sh .
./nvbuild.sh --noclean --v2
```

### Build the TF-TRT example
```
cd tensorrt/tftrt/examples/cpp/image-classification
mkdir build && cd build
cmake ..
make
```


### Run TF-TRT conversion and infer the converted model

Run inference
```
cd /workspace/tensorrt/tftrt/examples/cpp/image-classification/build
TF_CPP_VMODULE=trt_convert=2,trt_optimization_pass=2,trt_engine_utils=2,trt_engine_op=2,segment=2,trt_shape_optimization_profiles=2,trt_lru_cache=2,convert_graph=2,trt_engine_resource_ops=2 ./tf_trt_example --saved_model_dir=saved_model_dir
```
