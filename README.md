# Documentation for TensorRT in TensorFlow (TF-TRT)

The documentaion on how to accelerate inference in TensorFlow with TensorRT (TF-TRT) is here: https://docs.nvidia.com/deeplearning/dgx/tf-trt-user-guide/index.html

# Examples for TensorRT in TensorFlow (TF-TRT)

This repository contains a number of different examples
that show how to use
[TF-TRT](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensorrt).
TF-TRT is a part of TensorFlow
that optimizes TensorFlow graphs using
[TensorRT](https://developer.nvidia.com/tensorrt).
We have used these examples to verify the accuracy and
performance of TF-TRT. For more information see
[Verified Models](https://docs.nvidia.com/deeplearning/dgx/tf-trt-user-guide/index.html#verified-models).

## Examples

* [Image Classification](tftrt/examples/image-classification)
* [Object Detection](tftrt/examples/object_detection)


# Using TensorRT in TensorFlow (TF-TRT)

This module provides necessary bindings and introduces
`TRTEngineOp` operator that wraps a subgraph in TensorRT.
This module is under active development.


## Installing TF-TRT

Currently Tensorflow nightly builds include TF-TRT by default,
which means you don't need to install TF-TRT separately.
You can pull the latest TF containers from docker hub or
install the latest TF pip package to get access to the latest TF-TRT.

If you want to use TF-TRT on NVIDIA Jetson platform, you can find
the download links for the relevant Tensorflow pip packages here:
https://docs.nvidia.com/deeplearning/dgx/index.html#installing-frameworks-for-jetson


## Installing TensorRT

In order to make use of TF-TRT, you will need a local installation
of TensorRT from the
[NVIDIA Developer website](https://developer.nvidia.com/tensorrt).
Installation instructions for compatibility with TensorFlow are provided on the
[TensorFlow GPU support](https://www.tensorflow.org/install/gpu) guide.


## Documentation

[TF-TRT documentaion](https://docs.nvidia.com/deeplearning/dgx/tf-trt-user-guide/index.html)
gives an overview of the supported functionalities, provides tutorials
and verified models, explains best practices with troubleshooting guides.


## Tests

TF-TRT includes both Python tests and C++ unit tests.
Most of Python tests are located in the test directory
and they can be executed uring `bazel test` or directly
with the Python command. Most of the C++ unit tests are
used to test the conversion functions that convert each TF op to
a number of TensorRT layers.


## Compilation

In order to compile the module, you need to have a local TensorRT installation
(libnvinfer.so and respective include files). During the configuration step,
TensorRT should be enabled and installation path should be set. If installed
through package managers (deb,rpm), configure script should find the necessary
components from the system automatically. If installed from tar packages, user
has to set path to location where the library is installed during configuration.

```shell
bazel build --config=cuda --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/
```


## License

[Apache License 2.0](LICENSE)
