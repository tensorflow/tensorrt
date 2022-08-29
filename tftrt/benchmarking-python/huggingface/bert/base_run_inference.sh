#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Runtime Parameters
MODEL_NAME=""
MODEL_DIR=""

# Default Argument Values
BYPASS_ARGUMENTS=""
BATCH_SIZE=32
SEQ_LEN=128
# TODO: remove when real dataloader is implemented
DATA_DIR="/tmp"

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove --batch_size= from processing
        ;;
        --data_dir=*)
        shift # Remove --data_dir= from processing
        ;;
        --total_max_samples=*)
        shift # Remove --total_max_samples= from processing
        ;;
        --output_tensors_name=*)
        shift # Remove --output_tensors_name= from processing
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        --vocab_size=*)
        shift # Remove --vocab_size= from processing
        ;;
        --sequence_length=*)
        SEQ_LEN="${arg#*=}"
        shift # Remove --sequence_length= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

# Trimming front and back whitespaces
BYPASS_ARGUMENTS=$(echo ${BYPASS_ARGUMENTS} | tr -s " ")

# ============== Set model specific parameters ============= #

MIN_SEGMENT_SIZE=5
VOCAB_SIZE=-1
MAX_SAMPLES=1
OUTPUT_TENSORS_NAME="last_hidden_state,pooler_output"

case ${MODEL_NAME} in
  "bert_base_uncased" | "bert_large_uncased")
    VOCAB_SIZE=30522
    ;;

  "bert_base_cased" | "bert_large_cased")
    VOCAB_SIZE=28996
    ;;

  "bart_base" | "bart_large")
    VOCAB_SIZE=50265
    MIN_SEGMENT_SIZE=90
    OUTPUT_TENSORS_NAME="encoder_last_hidden_state,last_hidden_state"
    ;;
esac

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
# Custom Transormer Task Flags
echo "[*] VOCAB_SIZE: ${VOCAB_SIZE}"
echo "[*] SEQ_LEN: ${SEQ_LEN}"
echo "[*] MAX_SAMPLES: ${MAX_SAMPLES}"
echo "[*] OUTPUT_TENSORS_NAME: ${OUTPUT_TENSORS_NAME}"
echo ""
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"

echo -e "********************************************************************\n"

# ======================= ARGUMENT VALIDATION ======================= #

# ----------------------  Dataset Directory --------------

if [[ -z ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

# ----------------------  Model Directory --------------

if [[ -z ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

INPUT_SAVED_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}/pb_model

if [[ ! -d ${INPUT_SAVED_MODEL_DIR} ]]; then
    echo "ERROR: the directory \`${INPUT_SAVED_MODEL_DIR}\` does not exist."
    exit 1
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

set -x

python ${BASE_DIR}/infer.py \
    --data_dir ${DATA_DIR} \
    --calib_data_dir ${DATA_DIR} \
    --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
    --output_saved_model_dir /tmp/$RANDOM \
    --model_name "${MODEL_NAME}" \
    --model_source "huggingface" \
    --batch_size ${BATCH_SIZE} \
    --vocab_size ${VOCAB_SIZE} \
    --sequence_length=${SEQ_LEN} \
    --total_max_samples=${MAX_SAMPLES} \
    --output_tensors_name=${OUTPUT_TENSORS_NAME} \
    ${BYPASS_ARGUMENTS}
