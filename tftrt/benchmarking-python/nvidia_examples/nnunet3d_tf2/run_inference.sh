#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/msd_task01/numpy_3d_bigger_tf2 \
    --calib_data_dir=/data/msd_task01/numpy_3d_bigger_tf2 \
    --input_saved_model_dir=/models/nvidia_examples/nnunet3d_tf2 \
    --model_name "nnunet3d_tf2" \
    --model_source "nvidia_examples" \
    --batch_size=1 \
    --output_tensors_name="output_1" \
    --total_max_samples=500 \
    ${@}
