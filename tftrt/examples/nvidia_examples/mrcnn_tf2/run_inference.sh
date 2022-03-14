#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pip --no-cache-dir --no-cache install Cython pytest ujson

python -c "from pycocotools.coco import COCO" > /dev/null 2>&1
DEPENDENCIES_STATUS=$?

if [[ ${DEPENDENCIES_STATUS} != 0 ]]; then
    git clone -b v2.5.0 https://github.com/pybind/pybind11 /opt/pybind11
    cd /opt/pybind11 && cmake . && make install && pip install .
    pip install "git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI"
fi

python ${BASE_DIR}/infer.py \
    --data_dir=/data/coco2017/tfrecord \
    --input_saved_model_dir=/models/nvidia_examples/mrcnn_tf2 \
    --batch_size=8 \
    --output_tensors_name="detection_boxes,detection_classes,detection_scores,image_info,num_detections,source_ids" \
    --total_max_samples=5200 \
    ${@}

