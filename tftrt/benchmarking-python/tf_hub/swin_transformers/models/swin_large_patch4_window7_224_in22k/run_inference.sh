#!/bin/bash

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

bash ${BASE_DIR}/base_run_inference.sh --model_name="swin_large_patch4_window7_224_in22k" --input_size=224 --num_classes=21841 ${@}