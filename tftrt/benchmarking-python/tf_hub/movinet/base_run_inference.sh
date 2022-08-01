#!/bin/#!/usr/bin/env bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Runtime Parameters
MODEL_NAME=""

# Default Argument Values
BATCH_SIZE=32
NUM_ITERATIONS=1000
OUTPUT_TENSOR_NAMES="classifier_head"

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
        --num_iterations=*)
        NUM_ITERATIONS="${arg#*=}"
        shift # Remove --num_iterations= from processing
        ;;
        --input_size=*)
        INPUT_SIZE="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --num_frames=*)
        NUM_FRAMES="${arg#*=}"
        shift # Remove --num_frames= from processing
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
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
# Custom MoViNets flags
echo "[*] INPUT_SIZE: (${INPUT_SIZE}, ${INPUT_SIZE})"
echo "[*] NUM_FRAMES: ${NUM_FRAMES}"
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

set -x

python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --input_saved_model_dir=${MODEL_DIR} \
    --model_name "movinet_${MODEL_NAME}" \
    --model_source "tf_hub" \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1 \
    --use_synthetic_data  \
    --num_iterations=${NUM_ITERATIONS} \
    ${BYPASS_ARGUMENTS}
