#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/data/seq_20 \
    --input_saved_model_dir=/models/nvidia_examples/sim_tf2/21.10.0_fp32 \
    --batch_size=256 \
    --output_tensors_name="sim_model,sim_model_1" \
    --total_max_samples=220000 \
    ${@}
