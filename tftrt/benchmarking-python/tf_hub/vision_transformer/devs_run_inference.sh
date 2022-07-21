#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/data/imagenet \
    --calib_data_dir /data/imagenet \
    --input_saved_model_dir=/models/tf_hub/vision_transformers/vit_b8_classification \
    --batch_size=32 \
    --output_tensors_name="output_0" \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1 \
    --use_synthetic_data  \
    --num_iterations=1000  \
    ${@}