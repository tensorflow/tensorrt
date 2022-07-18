#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/data/c4/realnewslike \
    --input_saved_model_dir=/models/huggingface/t5/t5-small/saved_models/model \
    --vocab_dir=/models/huggingface/t5/t5-small/saved_models/tokenizer \
    --tokenizer_model_dir=/models/huggingface/t5/t5-small/saved_models/tokenizer \
    --batch_size=32 \
    --output_tensors_name="encoder_last_hidden_state,logits,past_key_values" \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1000 \
    --use_synthetic_data  \
    --num_iterations=1000  \
    ${@}
