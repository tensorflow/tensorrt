# Benchmark Runner

This straightforward program uses TF's C++ API to serve a frozen CNN and measure throughput.

## Docker Environment

Pull the image:

```
docker pull nvcr.io/nvidia/tensorflow:22.03-tf2-py3
```

Start the container:

```
docker run --gpus all --rm -it -p 8888:8888 --name TFTRT_CPP nvcr.io/nvidia/tensorflow:22.03-tf2-py3
```

Clone the repo:

```
git clone https://github.com/nvkevihu/tensorrt
cd tensorrt
```

Link to the TF example source directory:

```
ln -s /workspace/tensorrt/tftrt/examples-cpp/benchmark_runner /opt/tensorflow/tensorflow-source/tensorflow/examples/benchmark_runner
```

## Model Conversion

To define and convert the basic CNN to TF-TRT:

```
python3 tf2_save_model.py
```

## Building

```
cd /opt/tensorflow/tensorflow-source/tensorflow/examples/benchmark_runner
cp tftrt-build.sh /opt/tensorflow
cd /opt/tensorflow 
bash ./tftrt-build.sh
```

## Running

Measuring only throughput:

```
/opt/tensorflow/tensorflow-source/bazel-bin/tensorflow/examples/benchmark_runner/tftrt_benchmark_runner
```

Profiling 20 iterations:

```
/opt/tensorflow/tensorflow-source/bazel-bin/tensorflow/examples/benchmark_runner/tftrt_benchmark_runner --eval_iters=20 --trace=true
```
