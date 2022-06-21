# Benchmark Runner

This straightforward example uses TF's C++ API to serve a frozen CNN and measure throughput. Built off of the [example here](https://github.com/tensorflow/tensorrt/tree/fb0a2cf638c8707041e42451c601247f04c7e6d8/tftrt/examples/cpp/image-classification).

## Docker Environment

Pull the image:

```
docker pull nvcr.io/nvidia/tensorflow:22.05-tf2-py3
```

Start the container:

```
docker run --gpus all --rm -it -p 8888:8888 --name TFTRT_CPP nvcr.io/nvidia/tensorflow:22.05-tf2-py3
```

Clone the repo:

```
git clone https://github.com/tensorflow/tensorrt
```

## Model Conversion

To define and convert the basic CNN to TF-TRT:

```
python3 tf2_save_model.py
```

## Building

```
cd tensorrt/tftrt/examples/cpp/benchmark_runner
mkdir build && cd build
cmake ..
make
```

## Running

```
./tf_trt_benchmark_runner --model_path="/path/to/model/dir"
```
