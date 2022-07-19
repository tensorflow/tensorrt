#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/data/msd_task01/numpy_2d_bigger_tf2 \
    --calib_data_dir=/data/msd_task01/numpy_2d_bigger_tf2 \
    --input_saved_model_dir=/models/nvidia_examples/nnunet2d_tf2 \
    --batch_size=1 \
    --num_image_slices=32 \
    --output_tensors_name="output_1" \
    --total_max_samples=16000 \
    ${@}
