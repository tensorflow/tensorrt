#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="${BASE_DIR}/../efficientnet_base"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/imagenet \
    --calib_data_dir=/data/imagenet \
    --input_saved_model_dir=/models/nvidia_examples/efficientnet_v1_B4_tf2 \
    --model_name "efficientnet_v1_B4_tf2" \
    --model_source "nvidia_examples" \
    --batch_size=128 \
    --output_tensors_name="output_1" \
    --total_max_samples=55000 \
    --input_size=380 \
    ${@}
