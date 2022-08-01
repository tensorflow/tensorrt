#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Default Argument Values
BATCH_SIZE=128
OUTPUT_TENSOR_NAMES="classes"
TOTAL_MAX_SAMPLES=50000
DATA_DIR="/data/imagenet"
MODEL_DIR="/models/nvidia_examples/resnet50-v1.5_tf1"

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
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"

echo -e "********************************************************************\n"

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --input_saved_model_dir=${MODEL_DIR} \
    --model_name "resnet50-v1.5_tf1" \
    --model_source "nvidia_examples" \
    --batch_size=${BATCH_SIZE} \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    --input_size 224 \
    --num_classes 1000 \
    --preprocess_method resnet50_v1_5_tf1_ngc \
    --total_max_samples=${TOTAL_MAX_SAMPLES} \
    ${BYPASS_ARGUMENTS}
