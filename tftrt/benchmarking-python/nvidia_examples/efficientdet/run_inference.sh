#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/imagenet \
    --calib_data_dir=/data/imagenet \
    --input_saved_model_dir=/models/nvidia_examples/efficientdet \
    --batch_size=128 \
    --model_name "efficientdet" \
    --model_source "nvidia_examples" \
    --output_tensors_name="output_1_1,output_1_2,output_1_3,output_1_4,output_1_5,output_2_1,output_2_2,output_2_3,output_2_4,output_2_5" \
    --total_max_samples=55000 \
    --input_size=512 \
    ${@}
