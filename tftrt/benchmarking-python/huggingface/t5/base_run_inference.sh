#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Runtime Parameters
MODEL_NAME=""
DATASET_NAME="realnewslike"

# Default Argument Values
BATCH_SIZE=32
SEQ_LEN=128

NUM_ITERATIONS=1000
OUTPUT_TENSOR_NAMES="encoder_last_hidden_state,logits,past_key_values"

BYPASS_ARGUMENTS=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --dataset_name=*)
        DATASET_NAME="${arg#*=}"
        shift # Remove --dataset_name= from processing
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove --batch_size= from processing
        ;;
        --sequence_length=*)
        SEQ_LEN="${arg#*=}"
        shift # Remove --sequence_length= from processing
        ;;
        --num_iterations=*)
        NUM_ITERATIONS="${arg#*=}"
        shift # Remove --num_iterations= from processing
        ;;
        --output_tensors_name=*)
        OUTPUT_TENSOR_NAMES="${arg#*=}"
        shift # Remove --output_tensors_name= from processing
        ;;
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        shift # Remove --data_dir= from processing
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        --tokenizer_model_dir=*)
        TOKENIZER_DIR="${arg#*=}"
        shift # Remove --tokenizer_model_dir= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

# Trimming front and back whitespaces
BYPASS_ARGUMENTS=$(echo ${BYPASS_ARGUMENTS} | tr -s " ")

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo "[*] DATASET_NAME: ${DATASET_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] TOKENIZER_DIR: ${TOKENIZER_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
# Custom T5 Task Flags
echo "[*] SEQ_LEN: ${SEQ_LEN}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"

echo -e "********************************************************************\n"

DATA_DIR="${DATA_DIR}/${DATASET_NAME}"
MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}/saved_models/model"
TOKENIZER_DIR="${TOKENIZER_DIR}/${MODEL_NAME}/saved_models/tokenizer"

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

if [[ ! -d ${TOKENIZER_DIR} ]]; then
    echo "ERROR: \`--tokenizer_model_dir=/path/to/directory\` does not exist. [Received: \`${TOKENIZER_DIR}\`]"
    exit 1
fi

# Install Dependencies

TF_VERSION=$(python -c "import tensorflow as tf; print('.'.join(tf.__version__.split('.')[:2]))")
pip install --upgrade \
    prefetch_generator \
    orjson \
    t5==0.4.0 \
    tensorflow-text==${TF_VERSION}

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --input_saved_model_dir=${MODEL_DIR} \
    --model_name "${MODEL_NAME}" \
    --model_source "huggingface" \
    --tokenizer_model_dir=${TOKENIZER_DIR}\
    --vocab_model_dir=${TOKENIZER_DIR}\
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1 \
    --use_synthetic_data  \
    --num_iterations=${NUM_ITERATIONS} \
    ${BYPASS_ARGUMENTS}
