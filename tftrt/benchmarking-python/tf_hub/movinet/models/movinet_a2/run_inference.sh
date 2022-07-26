#!/bin/bash

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../.."

NUM_FRAMES=5
INPUT_SIZE=224

bash ${BASE_DIR}/base_run_inference.sh --model_name="a2" --num_frames=5 --input_size=224 ${@}
