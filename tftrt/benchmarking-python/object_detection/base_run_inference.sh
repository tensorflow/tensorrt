#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

nvidia-smi

# Runtime Parameters
MODEL_NAME=""
DATA_DIR=""
MODEL_DIR=""

# Default Argument Values
BYPASS_ARGUMENTS=""
BATCH_SIZE=8

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
        DATA_DIR="${arg#*=}"
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
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

# Trimming front and back whitespaces
BYPASS_ARGUMENTS=$(echo ${BYPASS_ARGUMENTS} | tr -s " ")

# ============== Set model specific parameters ============= #

INPUT_SIZE=640
MAX_SAMPLES=5000
OUTPUT_TENSORS_NAME="boxes,classes,num_detections,scores"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
# Custom Object Detection Task Flags
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] INPUT_SIZE: ${INPUT_SIZE}"
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

VAL_DATA_DIR=${DATA_DIR}/val2017
ANNOTATIONS_DATA_FILE=${DATA_DIR}/annotations/instances_val2017.json

if [[ ! -d ${VAL_DATA_DIR} ]]; then
    echo "ERROR: the directory \`${VAL_DATA_DIR}\` does not exist."
    exit 1
fi

if [[ ! -f ${ANNOTATIONS_DATA_FILE} ]]; then
    echo "ERROR: the file \`${ANNOTATIONS_DATA_FILE}\` does not exist."
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

INPUT_SAVED_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}_640_bs${BATCH_SIZE}

if [[ ! -d ${INPUT_SAVED_MODEL_DIR} ]]; then
    echo "ERROR: the directory \`${INPUT_SAVED_MODEL_DIR}\` does not exist."
    exit 1
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BENCH_DIR}

# Step 1: Installing dependencies if needed:
python -c "from pycocotools.coco import COCO" > /dev/null 2>&1
DEPENDENCIES_STATUS=$?

if [[ ${DEPENDENCIES_STATUS} != 0 ]]; then
    bash "${BASE_DIR}/../helper_scripts/install_pycocotools.sh"
fi

set -x

# Step 2: Execute the example
python ${BASE_DIR}/infer.py \
    --data_dir ${VAL_DATA_DIR} \
    --calib_data_dir ${VAL_DATA_DIR} \
    --annotation_path ${ANNOTATIONS_DATA_FILE} \
    --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
    --output_saved_model_dir /tmp/$RANDOM \
    --model_name "${MODEL_NAME}" \
    --model_source "tf_models_object" \
    --batch_size ${BATCH_SIZE} \
    --input_size ${INPUT_SIZE} \
    --total_max_samples=${MAX_SAMPLES} \
    --output_tensors_name=${OUTPUT_TENSORS_NAME} \
    ${BYPASS_ARGUMENTS}
