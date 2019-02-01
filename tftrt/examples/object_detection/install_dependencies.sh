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

echo Setup local variables...
TF_MODELS_DIR=../third_party/models
RESEARCH_DIR=$TF_MODELS_DIR/research
SLIM_DIR=$RESEARCH_DIR/slim
COCO_API_DIR=../third_party/cocoapi
PYCOCO_DIR=$COCO_API_DIR/PythonAPI
PROTO_BASE_URL="https://github.com/google/protobuf/releases/download/v3.5.1/"
PROTOC_DIR=$PWD/protoc

#echo Install python-tk ...
#python -V 2>&1 | grep "Python 3" || \
#  ( export DEBIAN_FRONTEND=noninteractive && \
#    apt-get update && \
#    apt-get install -y --no-install-recommends python-tk )

set -v

echo Download protobuf...
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
wget --no-check-certificate ${PROTO_BASE_URL}${filename}
unzip -o ${filename}
popd

echo Compile object detection protobuf files...
pushd $RESEARCH_DIR
$PROTOC_DIR/bin/protoc object_detection/protos/*.proto --python_out=.
popd

echo Install tensorflow/models/research...
pushd $RESEARCH_DIR
pip install .
popd

echo Install tensorflow/models/research/slim...
pushd $SLIM_DIR
pip install .
popd

echo Install cocodataset/cocoapi/PythonAPI...
pushd $PYCOCO_DIR
python setup.py build_ext --inplace
make
# pip install .
python setup.py install
popd
