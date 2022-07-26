#!/bin/bash

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

bash ${BASE_DIR}/base_run_inference.sh --model_name="a0" --num_frames=5 --input_size=172 ${@}
