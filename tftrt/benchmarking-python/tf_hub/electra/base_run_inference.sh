#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#install packages for the rest of the script
pip install tensorflow_text

# Runtime Parameters
MODEL_NAME=""
OUTPUT_TENSOR_NAMES=""

# Default Argument Values
BATCH_SIZE=32
SEQ_LEN=128
VOCAB_SIZE=30522
TOTAL_MAX_SAMPLES=50000

BYPASS_ARGUMENTS=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --output_tensors_name=*)
        OUTPUT_TENSOR_NAMES="${arg#*=}"
        shift # Remove --output_tensors_name= from processing
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove --batch_size= from processing
        ;;
        --sequence_length=*)
        SEQ_LEN="${arg#*=}"
        shift # Remove --sequence_length= from processing
        ;;
        --vocab_size=*)
        VOCAB_SIZE="${arg#*=}"
        shift # Remove --vocab_size= from processing
        ;;
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        shift # Remove --data_dir= from processing
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        --tokenizer_dir=*)
        TOKENIZER_DIR="${arg#*=}"
        shift # Remove --tokenizer_model_dir= from processing
        ;;
        --total_max_samples=*)
        TOTAL_MAX_SAMPLES="${arg#*=}"
        shift # Remove --total_max_samples= from processing
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
echo ""
echo "[*] SEQ_LEN: ${SEQ_LEN}"
echo "[*] VOCAB_SIZE: ${VOCAB_SIZE}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] TOKENIZER_DIR: ${TOKENIZER_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo "[*] TOTAL_MAX_SAMPLES: ${TOTAL_MAX_SAMPLES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"

echo -e "********************************************************************\n"

MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}"

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

if [[ ! -d ${TOKENIZER_DIR} ]]; then
    echo "ERROR: \`--tokenizer_dir=/path/to/directory\` does not exist. [Received: \`${TOKENIZER_DIR}\`]"
    exit 1
fi

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --tokenizer_dir=${TOKENIZER_DIR}\
    --input_saved_model_dir=${MODEL_DIR} \
    --model_name "${MODEL_NAME}" \
    --model_source "tf_hub" \
    --sequence_length=${SEQ_LEN} \
    --vocab_size=${VOCAB_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    --total_max_samples=${TOTAL_MAX_SAMPLES} \
    ${BYPASS_ARGUMENTS}
