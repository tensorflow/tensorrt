#!/usr/bin/env bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#install packages for the rest of the script
pip install tensorflow_text tensorflow_hub scipy==1.4.1


# Runtime Parameters
MODEL_NAME=""

# Default Argument Values
BATCH_SIZE=32
SEQ_LEN=128
VOCAB_SIZE=33000

NUM_ITERATIONS=1000
OUTPUT_TENSOR_NAMES="albert_encoder,albert_encoder_1,albert_encoder_2,albert_encoder_3,albert_encoder_4,albert_encoder_5,albert_encoder_6,albert_encoder_7,albert_encoder_8,albert_encoder_9,albert_encoder_10,albert_encoder_11,albert_encoder_12,albert_encoder_13"

BYPASS_ARGUMENTS=""

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
        --sequence_length=*)
        SEQ_LEN="${arg#*=}"
        shift # Remove --sequence_length= from processing
        ;;
        --num_iterations=*)
        NUM_ITERATIONS="${arg#*=}"
        shift # Remove --num_iterations= from processing
        ;;
        --vocab_size=*)
        VOCAB_SIZE="${arg#*=}"
        shift # Remove --vocab_size= from processing
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
        --tokenizer_dir=*)
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
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] TOKENIZER_DIR: ${TOKENIZER_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
# Custom ALBERT Flags
echo "[*] SEQ_LEN: ${SEQ_LEN}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"

echo -e "********************************************************************\n"

MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}/"

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
    --input_saved_model_dir=${MODEL_DIR} \
    --tokenizer_dir=${TOKENIZER_DIR}\
    --model_name "${MODEL_NAME}" \
    --model_source "tf_hub" \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1 \
    --use_synthetic_data  \
    --num_iterations=${NUM_ITERATIONS} \
    ${BYPASS_ARGUMENTS}
