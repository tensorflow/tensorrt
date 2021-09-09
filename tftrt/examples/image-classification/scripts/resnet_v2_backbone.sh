#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

bash ${SCRIPT_DIR}/base_script.sh \
    --model_name="resnet50v2_backbone" \
    --input_signature_key="resnet50v2" \
    --skip_accuracy_testing \
    ${@}
