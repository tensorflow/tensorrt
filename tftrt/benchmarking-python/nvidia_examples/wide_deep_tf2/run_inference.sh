#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pip install --no-deps \
    tensorflow-transform~=0.24 \
    tensorflow-metadata~=0.24 \
    tfx-bsl~=0.24

python ${BASE_DIR}/infer.py \
    --data_dir=/data/outbrain \
    --input_saved_model_dir=/models/nvidia_examples/wide_deep_tf2 \
    --batch_size=32768 \
    --output_tensors_name "output_1" \
    --total_max_samples=28000000 \
    ${@}
