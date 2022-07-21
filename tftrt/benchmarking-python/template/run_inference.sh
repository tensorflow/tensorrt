#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/path/to/script \
    --calib_data_dir=/path/to/script \
    --input_saved_model_dir=/path/to/saved_model \
    --batch_size=<BATCH_SIWZE> \
    --output_tensors_name="logits,probs" \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1000 \
    --use_synthetic_data  \
    --num_iterations=1000  \
    ${@}
