#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

bash ${SCRIPT_DIR}/base_script.sh --model_name="resnet_v1.5_50_tfv2" ${@}