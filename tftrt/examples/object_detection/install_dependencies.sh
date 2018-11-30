#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

TF_MODELS_DIR=third_party/models
COCO_API_DIR=third_party/cocoapi

python -V 2>&1 | grep "Python 3" || \
  ( export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends python-tk )

RESEARCH_DIR=$TF_MODELS_DIR/research
SLIM_DIR=$RESEARCH_DIR/slim
PYCOCO_DIR=$COCO_API_DIR/PythonAPI

pushd $RESEARCH_DIR

# GET PROTOC 3.5

BASE_URL="https://github.com/google/protobuf/releases/download/v3.5.1/"
PROTOC_DIR=protoc
PROTOC_EXE=$PROTOC_DIR/bin/protoc

mkdir -p $PROTOC_DIR
pushd $PROTOC_DIR
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ] ; then
  filename="protoc-3.5.1-linux-aarch_64.zip"
elif [ "$ARCH" == "x86_64" ] ; then
  filename="protoc-3.5.1-linux-x86_64.zip"
else
  echo ERROR: $ARCH not supported.
  exit 1;
fi
wget --no-check-certificate ${BASE_URL}${filename}
unzip ${filename}
popd

# BUILD PROTOBUF FILES
$PROTOC_EXE object_detection/protos/*.proto --python_out=.

# INSTALL OBJECT DETECTION

pip install -e .

popd

pushd $SLIM_DIR
pip install -e .
popd

# INSTALL PYCOCOTOOLS

pushd $PYCOCO_DIR
pip install -e .
popd
