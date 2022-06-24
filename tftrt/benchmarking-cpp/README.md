# Benchmark Runner

This straightforward example uses TF's C++ API to serve a saved model and measure throughput. Built off of the [example here](https://github.com/tensorflow/tensorrt/tree/fb0a2cf638c8707041e42451c601247f04c7e6d8/tftrt/examples/cpp/image-classification).

## Docker Environment

Pull the image:

```
docker pull nvcr.io/nvidia/tensorflow:22.06-tf2-py3
```

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

```
/workspace/tensorrt/tftrt/benchmarking-cpp/build-scripts/setup.sh
cd /opt/tensorflow
./tftrt-build.sh
```

## Running

```
/opt/tensorflow/tensorflow-source/bazel-bin/tensorflow/examples/benchmarking-cpp/tftrt_benchmark_runner --model_path="/path/to/dest/dir"
```

## Profiling

To profile, set the `--out_dir` flag. Run `tensorboard --logdir [out_dir]` to view results.
