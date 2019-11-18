```
# Copyright 2019 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```

<!-- #region -->


# TensorFlow-TensorRT (TF-TRT) C++ Image Recognition Demo

This example shows how you can load a native TensorFlow model (saved as a frozen graph), convert it to a TF-TRT optimized model (via the TF-TRT Python API), then load and serve the model in C++.

This example is built based upon Google TensorFlow C++ image classificaition example at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image.

## Docker environment
Docker images provide a convinient and repeatable environment for experimentation. This workflow was tested in the NVIDIA NGC TensorFlow 19.09 docker container that comes with a TensorFlow 1.14 build. Tools required for building this example, such as Bazel, NVIDIA CUDA, CUDNN, NCCL libraries are all readily setup.

To replecate the below steps, start by pulling the NGC TF container:

```
docker pull nvcr.io/nvidia/tensorflow:19.09-py3
```

Then start the container with nvidia-docker:

```
nvidia-docker run --rm -it -p 8888:8888 --name TFTRT_CPP nvcr.io/nvidia/tensorflow:19.09-py3
```

You will land at `/workspace` within the docker container. Clone the TF-TRT example repository with:

```
git clone https://github.com/vinhngx/tensorrt
cd tensorrt 
git checkout vinhn-tf-cpp-1.14

```

Then copy the content of this C++ example directory to the TensorFlow example source directory:

```
cp -r ./tftrt/examples/cpp/image-classification/ /opt/tensorflow/tensorflow-source/tensorflow/examples/
cd /opt/tensorflow/tensorflow-source/tensorflow/examples/image-classification
```


## Native Tensoflow Model

The TensorFlow `GraphDef` that contains the model definition and weights is not
packaged in the repo because of its size. Instead, you must first download the
file to the `data` directory in the source tree:
<!-- #endregion -->

```bash
mkdir data
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C ./data -xz
```

<!-- #region -->
## Convert to TF-TRT Model

A TF-TRT conversion script is provided in `tf-trt-conversion.py`. Execute this script with:

``` 
cd /opt/tensorflow/tensorflow-source/tensorflow/examples/image-classification
cp ../label_image/data/grace_hopper.jpg ./data
python tf-trt-conversion.py
```

This script will load the native TensorFlow model downloaded above and convert it, by default, to an FP32 TF-TRT model. As part of the conversion, the script will also carry out benchmarking and print out the throughput statistics. 

A Jupyter notebook version is provided in `tf-trt-conversion.ipynb` for your own experimentation. 

By default, this script will produce a TF-TRT model at `/opt/tensorflow/tensorflow-source/tensorflow/examples/image-classification/data/inception_v3_2016_08_28_frozen_tftrt_fp32.pb`.
<!-- #endregion -->

<!-- #region -->
## Build the C++ example
The NVIDIA NGC container should have everything you need to run this example installed already.

To build it, first, you need to copy the two build scripts `image_classification_bazel_build.sh` and `image_classification_nvbuild.sh` to `/opt/tensorflow`:

```
cp image_classification_bazel_build.sh /opt/tensorflow
cp image_classification_nvbuild.sh /opt/tensorflow
```

Then from `/opt/tensorflow`, run the build command with `--noclean` option on the first build:
<!-- #endregion -->

```bash
cd /opt/tensorflow 
bash ./image_classification_nvbuild.sh  --python3.6 --noclean
```

For subsequent build, add the `--noconfig` option to speed up the build process:

```bash
bash ./image_classification_nvbuild.sh  --python3.6 --noclean --noconfig
```

That should build a binary executable `tftrt_label_image` that you can then run like this:

```bash
tensorflow-source/bazel-bin/tensorflow/examples/image-classification/tftrt_label_image
```

This uses the default image example image that ships with the framework at `/opt/tensorflow/tensorflow-source/tensorflow/examples/label_image/data/grace_hopper.jpg` using the converted TF-TRT FP32 model at `/opt/tensorflow/tensorflow-source/tensorflow/examples/image-classification/data/inception_v3_2016_08_28_frozen_tftrt_fp32.pb`, and should
output something similar to this:

```
I tensorflow/examples/label_image/main.cc:206] military uniform (653): 0.834306
I tensorflow/examples/label_image/main.cc:206] mortarboard (668): 0.0218692
I tensorflow/examples/label_image/main.cc:206] academic gown (401): 0.0103579
I tensorflow/examples/label_image/main.cc:206] pickelhaube (716): 0.00800814
I tensorflow/examples/label_image/main.cc:206] bulletproof vest (466): 0.00535088
```

In this case, we're using the default image of Admiral Grace Hopper, and you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.8.

The program will also benchmark and output the throughput. Observe the improved throughput offered by moving from Python to C++ serving.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
tensorflow-source/bazel-bin/tensorflow/examples/label_image/tftrt_label_image --image=my_image.png
```

## What's next

Try to build TF-TRT FP16 and INT8 models and test on your own data, and serve them with C++.

```bash

```
