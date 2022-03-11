#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

bash ${BASE_DIR}/base_run_inference.sh --model_name="ssd_resnet_50_fpn_coco" ${@}