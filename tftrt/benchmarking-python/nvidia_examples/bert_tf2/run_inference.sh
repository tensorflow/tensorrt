#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/squad/bert_tf2 \
    --calib_data_dir=/data/squad/bert_tf2 \
    --input_saved_model_dir=/models/nvidia_examples/bert_tf2 \
    --model_name "bert_tf2" \
    --model_source "nvidia_examples" \
    --batch_size=64 \
    --output_tensors_name="end_positions,start_positions" \
    --total_max_samples=11000 \
    ${@}
