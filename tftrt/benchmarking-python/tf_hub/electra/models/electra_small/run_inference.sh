#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

OUTPUT_TENSOR_NAME="bert_encoder"
INDEX_UPPERBOUND=14

for output_tensor_index in $(seq ${INDEX_UPPERBOUND}); do
    OUTPUT_TENSOR_NAME="${OUTPUT_TENSOR_NAME},bert_encoder_${output_tensor_index}"
done

set -x

bash ${BASE_DIR}/base_run_inference.sh --model_name="electra_small" --output_tensors_name=${OUTPUT_TENSOR_NAME} ${@}