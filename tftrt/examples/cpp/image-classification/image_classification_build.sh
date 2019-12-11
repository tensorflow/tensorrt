#!/bin/bash
# Build the C++ TFTRT Example

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
