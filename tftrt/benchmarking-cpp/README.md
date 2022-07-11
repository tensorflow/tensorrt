# Benchmark Runner

This straightforward example uses TF's C++ API to serve a saved model and measure throughput. Built off of the [example here](https://github.com/tensorflow/tensorrt/tree/master/tftrt/benchmarking-python/cpp/image-classification).

## Docker Environment

Start the container:

```
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --name TFTRT_CPP nvcr.io/nvidia/tensorflow:22.06-tf2-py3
```

Clone the repo:

```
git clone https://github.com/tensorflow/tensorrt
```

## Model Conversion

To convert a saved model to TF-TRT:

```
python3 convert_model.py --model-dir /path/to/model/dir --output-dir /path/to/dest/dir
```

## Building

The binary relies on a modified Tensorflow, which will need to be rebuilt. Internal users can use a container with Tensorflow already modified and built, instead of building with Bazel, which will take much longer.

### Bazel

The `setup.sh` script applies the Tensorflow patch and prepares the container for the Bazel build.

```
/workspace/tensorrt/tftrt/benchmarking-cpp/build-scripts/setup.sh
cd /opt/tensorflow
./tftrt-build.sh
```

The binary will be located at `/opt/tensorflow/tensorflow-source/bazel-bin/tensorflow/examples/benchmarking-cpp/tftrt_benchmark_runner`.

### Prebuilt

For internal NVIDIA users, a container with a prebuilt modified Tensorflow is available. In the container, use CMake to build the binary without needing to rebuild Tensorflow:

```
cd /workspace/tensorrt/tftrt/benchmarking-cpp
mkdir build && cd build
cmake ..
make
```

The binary will be located at `/workspace/tensorrt/tftrt/benchmarking-cpp/tftrt_benchmark_runner`.

## Running

```
./tftrt_benchmark_runner --model_path="/path/to/dest/dir"
```

### Profiling

To profile, set the `--out_dir` flag. Run `tensorboard --logdir [out_dir]` to view results.
