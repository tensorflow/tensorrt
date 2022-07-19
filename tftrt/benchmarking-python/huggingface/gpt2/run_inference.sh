#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/tmp \
    --input_saved_model_dir=/models/huggingface/gpt2/gpt2/model \
    --tokenizer_model_dir=/models/huggingface/gpt2/gpt2/tokenizer \
    --batch_size=32 \
    --sequence_length=1024 \
    --output_tensors_name="last_hidden_state" \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1000 \
    --use_synthetic_data  \
    --num_iterations=1000  \
    ${@}
