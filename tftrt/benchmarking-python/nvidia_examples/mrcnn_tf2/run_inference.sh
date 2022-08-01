#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python -c "from pycocotools.coco import COCO" > /dev/null 2>&1
DEPENDENCIES_STATUS=$?

if [[ ${DEPENDENCIES_STATUS} != 0 ]]; then
    bash "${BASE_DIR}/../../helper_scripts/install_pycocotools.sh"
fi

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/coco2017/tfrecord \
    --calib_data_dir=/data/coco2017/tfrecord \
    --input_saved_model_dir=/models/nvidia_examples/mrcnn_tf2 \
    --model_name "mrcnn_tf2" \
    --model_source "nvidia_examples" \
    --batch_size=8 \
    --output_tensors_name="detection_boxes,detection_classes,detection_scores,image_info,num_detections,source_ids" \
    --total_max_samples=5200 \
    ${@}
