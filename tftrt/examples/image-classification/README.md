# Image classification examples

This example includes scripts to run inference using a number of popular image classification models.

You can turn on TensorFlow-TensorRT integration with the flag `--use_trt`. This
will apply TensorRT inference optimization to speed up execution for portions of
the model's graph where supported, and will fall back to native TensorFlow for
layers and operations which are not supported.
See https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html for more information.

When using TF-TRT, you can also control the precision with `--precision`.
float32 is the default (`--precision fp32`) with float16 (`--precision fp16`) or
int8 (`--precision int8`) allowing further performance improvements.
int8 mode requires a calibration step which is done
automatically.

## Models

We have verified the following models.

* MobileNet v1
* MobileNet v2
* NASNet - Large
* NASNet - Mobile
* ResNet50 v1
* ResNet50 v2
* VGG16
* VGG19
* Inception v3
* Inception v4

For the accuracy numbers of these models on the
ImageNet validation dataset, see
[Verified Models](https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html#verified-models)

## Setup
```
# Clone [tensorflow/models](https://github.com/tensorflow/models)
git clone https://github.com/tensorflow/models.git

# Add the models directory to PYTHONPATH to install tensorflow/models.
cd models
export PYTHONPATH="$PYTHONPATH:$PWD"

# Run the TF Slim setup.
cd research/slim
python setup.py install

# You may also need to install the requests package
pip install requests
```
Note: the PYTHONPATH environment variable will be not be saved between different
shells. You can either repeat that step each time you work in a new shell, or
add `export PYTHONPATH="$PYTHONPATH:/path/to/tensorflow_models"` to your .bashrc
file (replacing /path/to/tensorflow_models with the path to your
tensorflow/models repository).

See [Setting Up The Environment
](https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html#image-class-envirn)
for more information.

### Data

The example supports using a dataset in TFRecords or synthetic data.
In case of using TFRecord files, the scripts assume that TFRecords
are named according to the pattern: `validation-*-of-00128`.

The reported accuracy numbers are the results of running the scripts on
the ImageNet validation dataset.
You can download and process Imagenet using [this script provided by TF
Slim](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_imagenet.sh).
Please note that this script downloads both the training and validation sets,
and this example only requires the validation set.

See [Obtaining The ImageNet Data
](https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html#image-class-data)
for more information.

## Usage

`python inference.py --data_dir /imagenet_validation_data --model vgg_16 [--use_trt]`

Run with `--help` to see all available options.

See [General Script Usage
](https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html#image-class-usage)
for more information.

### Accuracy tests

There is the script `check_accuracy.py` provided in the example that parses the output log of `inference.py`
to find the reported accuracy, and reports whether that accuracy matches with the
baseline numbers.

See [Checking Accuracy
](https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html#image-class-accuracy)
for more information.
