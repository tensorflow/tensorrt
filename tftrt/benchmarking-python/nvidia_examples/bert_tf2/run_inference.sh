#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/data/squad/bert_tf2 \
    --input_saved_model_dir=/models/nvidia_examples/bert_tf2 \
    --batch_size=64 \
    --output_tensors_name="end_positions,start_positions" \
    --total_max_samples=11000 \
    ${@}
