#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python ${BASE_DIR}/infer.py \
    --data_dir=/data/squad/electra \
    --input_saved_model_dir=/models/nvidia_examples/electra_tf2 \
    --batch_size=64 \
    --do_lower_case \
    --output_tensors_name="tf_electra_for_question_answering,tf_electra_for_question_answering_1,tf_electra_for_question_answering_2,tf_electra_for_question_answering_3,tf_electra_for_question_answering_4" \
    --total_max_samples=11000 \
    ${@}
