#!/bin/bash

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

bash ${BASE_DIR}/base_run_inference.sh --model_name="swin_base_patch4_window12_384" --input_size=384 ${@}