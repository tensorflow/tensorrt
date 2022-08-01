#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=/data/squad/electra \
    --calib_data_dir=/data/squad/electra \
    --input_saved_model_dir=/models/nvidia_examples/electra_tf2 \
    --model_name "electra_tf2" \
    --model_source "nvidia_examples" \
    --batch_size=64 \
    --do_lower_case \
    --output_tensors_name="tf_electra_for_question_answering,tf_electra_for_question_answering_1,tf_electra_for_question_answering_2,tf_electra_for_question_answering_3,tf_electra_for_question_answering_4" \
    --total_max_samples=11000 \
    ${@}
