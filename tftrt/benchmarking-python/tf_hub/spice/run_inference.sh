#!/bin/#!/usr/bin/env bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BATCH_SIZE="1"
OUTPUT_TENSOR_NAMES="pitch,uncertainty"

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --batch_size=*)
        shift # Remove --batch_size= from processing
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
	      --total_max_samples=*)
        shift # Remove --total_max_samples= from processing
        ;;
        --samples_per_input=*)
        SAMPLES_PER_INPUT="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

echo -e "\n********************************************************************"
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
echo "[*] SAMPLES_PER_INPUT: ${SAMPLES_PER_INPUT}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: $(echo \"${BYPASS_ARGUMENTS}\" | tr -s ' ')"
echo -e "********************************************************************\n"


if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

# Dataset Directory

python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --input_saved_model_dir=${MODEL_DIR} \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    --batch_size ${BATCH_SIZE} \
    --samples_per_input=${SAMPLES_PER_INPUT} \
    `# The following is set because we will be running synthetic benchmarks` \
    --use_synthetic_data  \
    --num_iterations=${NUM_ITERATIONS} \
    --total_max_samples=1 \
    ${@}
