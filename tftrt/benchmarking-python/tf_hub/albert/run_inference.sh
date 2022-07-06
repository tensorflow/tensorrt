#!/bin/bash

nvidia-smi

set -x

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pip install tensorflow_text tensorflow_hub scipy==1.4.1

python ${BASE_DIR}/infer.py \
    --data_dir=/workspace/tftrt/benchmarking-python/tf_hub/albert/data \
    --input_saved_model_dir=/models/tf_hub/albert \
    --batch_size=1 \
    --vocab_size=32000 \
    --sequence_length=128 \
    --output_tensors_name="albert_encoder,albert_encoder_1,albert_encoder_2,albert_encoder_3,albert_encoder_4,albert_encoder_5,albert_encoder_6,albert_encoder_7,albert_encoder_8,albert_encoder_9,albert_encoder_10,albert_encoder_11,albert_encoder_12,albert_encoder_13" \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1000 \
    --use_synthetic_data  \
    --num_iterations=1000  \
    ${@}

#     # Execute the example
# COMMAND="python transformers.py \
#         --data_dir ${DATA_DIR} \
#         --calib_data_dir ${DATA_DIR} \
#         --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
#         --output_saved_model_dir /tmp/$RANDOM \
#         --batch_size ${BATCH_SIZE} \
#         --vocab_size ${VOCAB_SIZE} \
#         --sequence_length=${SEQ_LEN} \
#         --max_workspace_size ${MAX_WORKSPACE_SIZE} \
#         --total_max_samples=${MAX_SAMPLES} \
#         --output_tensors_name=${OUTPUT_TENSORS_NAME} \
#         ${BYPASS_ARGUMENTS}"
