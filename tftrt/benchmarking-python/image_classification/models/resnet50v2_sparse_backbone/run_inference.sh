#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

bash ${BASE_DIR}/base_run_inference.sh \
    --model_name="resnet50v2_sparse_backbone" \
    --input_signature_key="resnet50v2" \
    --use_synthetic_data \
    ${@}
