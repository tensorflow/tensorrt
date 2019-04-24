## NCF examples

The example script `inference.py` runs inference with NVIDIA NCF model implementation.
This script is included in the NVIDIA Tensorflow Docker
containers under `/workspace/nvidia-examples'.


## Model

Model that we use is available here:
`https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF`

### Setup for running within an NVIDIA Tensorflow Docker container


If you are running these examples within the [NVIDIA TensorFlow docker
container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow):

```
cd ../third_party/models
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Prepare dataset

We are using standard movielense dataset, which is available here:
`https://grouplens.org/datasets/movielens/`

To use it for our script you need to prepare it first (we require csv file).
You can do that using script, which is here:
`tensorrt/tftrt/examples/third_party/DeepLearningExamples/TensorFlow/Recommendation/NCF/prepare_dataset.sh`
You need to provide path where you download ml-20m dataset.

### Setup for running standalone

If you are running these examples within your own TensorFlow environment,
perform the following steps:

```
# Clone this repository (tensorflow/tensorrt) if you haven't already.
git clone https://github.com/tensorflow/tensorrt.git --recurse-submodules

# Add official models to python path
cd tensorrt/tftrt/examples/third_party/models/
export PYTHONPATH="$PYTHONPATH:$PWD"
```
## Usage

The main Python script is `inference.py`. Here is some example of usage:

```
python inference.py
    --data_dir /data/cache/ml-20m/
    --use_trt
    --precision FP16
```

Where:

`--data_dir`: Path to the ml-20m test dataset

`--use_trt`: Convert the graph to a TensorRT graph.

`--precision`: Precision mode to use, in this case FP16.


Run with `--help` to see all available options.


