#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/criteo \
    --calib_data_dir=/data/criteo \
    --input_saved_model_dir=/models/nvidia_examples/dlrm_tf2 \
    --model_name "dlrm_tf2" \
    --model_source "nvidia_examples" \
    --batch_size=65536 \
    --output_tensors_name="output_1" \
    --total_max_samples=92000000 \
    ${@}
