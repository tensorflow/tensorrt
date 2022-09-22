#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BATCH_SIZE="1"
INPUT_SIZE=384
OUTPUT_TENSOR_NAMES="detection_boxes,detection_classes,detection_scores,image_info,num_detections"
NUM_ITERATIONS=1000

DATA_DIR="/tmp"

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;   
        --output_tensors_name=*)
        OUTPUT_TENSOR_NAMES="${arg#*=}"
        shift # Remove --output_tensors_name= from processing
        ;;  
        --batch_size=*)
        shift # Remove --batch_size= from processing
        ;;   
        --input_size=*)
        INPUT_SIZE="${arg#*=}"
        shift # Remove --input_size= from processing
        ;;
        --data_dir=*)
        shift # Remove --data_dir= from processing
        ;;
        --num_iterations=*)
        NUM_ITERATIONS="${arg#*=}"
        shift # Remove --num_iterations= from processing
        ;;
        --total_max_samples=*)
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
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] NUM_ITERATIONS: ${NUM_ITERATIONS}"
echo ""
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
    --model_name "spinetnet_49_mobile" \
    --model_source "tf_models" \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    --batch_size=${BATCH_SIZE} \
    --input_size=${INPUT_SIZE} \
    `# The following is set because we will be running synthetic benchmarks` \
    --use_synthetic_data  \
    --num_iterations=${NUM_ITERATIONS} \
    --total_max_samples=1 \
    ${BYPASS_ARGUMENTS}
