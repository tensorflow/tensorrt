#!/bin/bash
# Build the C++ TFTRT Example
set -e
if [[ ! -f /opt/tensorflow/nvbuild.sh || ! -f /opt/tensorflow/nvbuildopts ]]; then
  echo This TF-TRT example is intended to be executed in the NGC TensorFlow container environment. Get one with, e.g. `docker pull nvcr.io/nvidia/tensorflow:19.10-py3`.
  exit 1
fi

# TODO: to programatically determine the python and tf API versions
PYVER=3.6 #TODO get this by parsing `python --version`
TFAPI=1 #TODO get this by parsing tf.__version__

/opt/tensorflow/nvbuild.sh --configonly --python$PYVER --v$TFAPI

BUILD_OPTS="$(cat /opt/tensorflow/nvbuildopts)"
if [[ "$TFAPI" == "2" ]]; then
  BUILD_OPTS="--config=v2 $BUILD_OPTS"
fi

cd /opt/tensorflow/tensorflow-source
bazel build $BUILD_OPTS tensorflow/examples/image-classification/...
