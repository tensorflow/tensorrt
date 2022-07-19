#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/../efficientnet_base/infer.py \
    --data_dir=/data/imagenet \
    --calib_data_dir=/data/imagenet \
    --input_saved_model_dir=/models/nvidia_examples/efficientnet_v2_tf2 \
    --batch_size=128 \
    --output_tensors_name="output_1" \
    --total_max_samples=55000 \
    --input_size=384 \
    ${@}
