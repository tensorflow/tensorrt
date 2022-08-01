#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Runtime Parameters
MODEL_NAME=""
DATA_DIR=""
MODEL_DIR=""

# Default Argument Values
BYPASS_ARGUMENTS=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
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
        --output_tensors_name=*)
        shift # Remove --output_tensors_name= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

# Trimming front and back whitespaces
BYPASS_ARGUMENTS=$(echo ${BYPASS_ARGUMENTS} | tr -s " ")

# ============== Set model specific parameters ============= #

INPUT_SIZE=224
PREPROCESS_METHOD="vgg"
NUM_CLASSES=1001
MAX_SAMPLES=50000
OUTPUT_TENSORS_NAME="logits"

case ${MODEL_NAME} in
  "inception_v3" | "inception_v4")
    INPUT_SIZE=299
    PREPROCESS_METHOD="inception"
    ;;

  "mobilenet_v1" | "mobilenet_v2")
    PREPROCESS_METHOD="inception"
    ;;

  "nasnet_large")
    INPUT_SIZE=331
    PREPROCESS_METHOD="inception"
    ;;

  "nasnet_mobile")
    PREPROCESS_METHOD="inception"
    ;;

  "resnet_v1.5_50_tfv2" )
    NUM_CLASSES=1000
    OUTPUT_TENSORS_NAME="activation_49"
    ;;

  "vgg_16" | "vgg_19" )
    NUM_CLASSES=1000
    ;;

  "resnet50v2_backbone" | "resnet50v2_sparse_backbone" )
    INPUT_SIZE=256
    OUTPUT_TENSORS_NAME="outputs"
    ;;
esac

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
# Custom Image Classification Task Flags
echo "[*] INPUT_SIZE: ${INPUT_SIZE}"
echo "[*] PREPROCESS_METHOD: ${PREPROCESS_METHOD}"
echo "[*] NUM_CLASSES: ${NUM_CLASSES}"
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

INPUT_SAVED_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}

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
    --model_source "tf_models_image" \
    --input_size ${INPUT_SIZE} \
    --preprocess_method ${PREPROCESS_METHOD} \
    --num_classes ${NUM_CLASSES} \
    --total_max_samples=${MAX_SAMPLES} \
    --output_tensors_name=${OUTPUT_TENSORS_NAME} \
    ${BYPASS_ARGUMENTS}
