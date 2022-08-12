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

To profile, set the `--out_dir` flag. This creates a log directory and serializes the `XSpace` to a location that TensorBoard expects (i.e. `[out_dir]/plugins/profile/[run_id]/[host_id].xplane.pb`).

Run `tensorboard --logdir [out_dir]` to view results. TensorBoard will generate the available dashboards using the `XSpace` directly.
