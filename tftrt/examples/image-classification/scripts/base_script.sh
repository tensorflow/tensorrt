#!/bin/bash

MODEL_NAME=""
BATCH_SIZE=32

# Default values of arguments
USE_XLA=0
USE_TF32=1
USE_TFTRT=0
TFTRT_PRECISION="FP32"

USE_SYNTHETIC_DATA=""

DATA_DIR=""
MODEL_DIR=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --use_xla)
        USE_XLA=1
        shift # Remove --use_xla from processing
        ;;
        --no_tf32)
        USE_TF32=0
        shift # Remove --no_tf32 from processing
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove --batch_size= from processing
        ;;
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        shift # Remove --data_dir= from processing
        ;;
        --model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --model_dir= from processing
        ;;
        --use_tftrt)
        USE_TFTRT=1
        shift # Remove --use_tftrt from processing
        ;;
        --tftrt_precision=*)
        TFTRT_PRECISION="${arg#*=}"
        shift # Remove --tftrt_precision= from processing
        ;;
        --use_synthetic)
        USE_SYNTHETIC_DATA=1
        shift # Remove --use_synthetic from processing
        ;;
    esac
done

if [[ ${USE_XLA} == "1" ]]; then
    TF_XLA_FLAGS="TF_XLA_FLAGS=--tf_xla_auto_jit=2"
else
    unset TF_XLA_FLAGS
    TF_XLA_FLAGS=""
fi

if [[ ${USE_TF32} == "0" ]]; then
    NVIDIA_TF32_OVERRIDE="NVIDIA_TF32_OVERRIDE=0"
else
    unset NVIDIA_TF32_OVERRIDE
    NVIDIA_TF32_OVERRIDE=""
fi

if [[ ${USE_SYNTHETIC_DATA} == "1" ]]; then
    USE_SYNTHETIC_DATA_FLAG="--use_synthetic"
else
    unset USE_SYNTHETIC_DATA_FLAG
    USE_SYNTHETIC_DATA_FLAG=""
fi

# ============== Set model specific parameters ============= #

INPUT_SIZE=224
PREPROCESS_METHOD="vgg"
NUM_CLASSES=1001

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

  "resnet_v1.5_50_tfv2" | "vgg_16" | "vgg_19" )
    NUM_CLASSES=1000
    ;;
esac

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
echo "[*] USE_XLA: ${USE_XLA}"
echo "[*] TF_XLA_FLAGS: ${TF_XLA_FLAGS}"
echo ""
echo "[*] USE_SYNTHETIC_DATA: ${USE_SYNTHETIC_DATA}"
echo "[*] USE_SYNTHETIC_DATA_FLAG: ${USE_SYNTHETIC_DATA_FLAG}"
echo ""
echo "[*] USE_TF32: ${USE_TF32}"
echo "[*] NVIDIA_TF32_OVERRIDE: ${NVIDIA_TF32_OVERRIDE}"
echo ""
echo "[*] USE_TFTRT: ${USE_TFTRT}"
echo "[*] TFTRT_PRECISION: ${TFTRT_PRECISION}"
echo ""
echo "[*] INPUT_SIZE: ${INPUT_SIZE}"
echo "[*] PREPROCESS_METHOD: ${PREPROCESS_METHOD}"
echo "[*] NUM_CLASSES: ${NUM_CLASSES}"
echo -e "********************************************************************\n"

# ======================= ARGUMENT VALIDATION ======================= #

# Dataset Directory

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
    echo "ERROR: \`--model_dir=/path/to/directory\` is missing."
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

INPUT_SAVED_MODEL_DIR=${MODEL_DIR}/${MODEL_NAME}

if [[ ! -d ${INPUT_SAVED_MODEL_DIR} ]]; then
    echo "ERROR: the directory \`${INPUT_SAVED_MODEL_DIR}\` does not exist."
    exit 1
fi

# TFTRT Arguments

ALLOWED_TFTRT_PRECISION="FP32 FP16 INT8"

if ! $(echo ${ALLOWED_TFTRT_PRECISION} | grep -w ${TFTRT_PRECISION} > /dev/null); then
    echo "ERROR: Unknown TFTRT_PRECISION received: \`${TFTRT_PRECISION}\`. [Allowed: ${ALLOWED_TFTRT_PRECISION}]"
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd ${BENCH_DIR}

# Execute the example

PREPEND_COMMAND="TF_CPP_MIN_LOG_LEVEL=2 ${TF_XLA_FLAGS} ${NVIDIA_TF32_OVERRIDE}"

if [[ ${USE_TFTRT} == "0" ]]; then
    COMMAND="${PREPEND_COMMAND} python image_classification.py \
        --data_dir ${DATA_DIR} \
        --calib_data_dir ${DATA_DIR} \
        --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
        --num_warmup_iterations 100 \
        --display_every 50 \
        ${USE_SYNTHETIC_DATA_FLAG} \
        --batch_size ${BATCH_SIZE} \
        --input_size ${INPUT_SIZE} \
        --preprocess_method ${PREPROCESS_METHOD} \
        --num_classes ${NUM_CLASSES}"
else
    COMMAND="${PREPEND_COMMAND} python image_classification.py \
        --data_dir ${DATA_DIR} \
        --calib_data_dir ${DATA_DIR} \
        --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
        --output_saved_model_dir /tmp/${RANDOM}/ \
        --num_warmup_iterations 100 \
        --display_every 50 \
        ${USE_SYNTHETIC_DATA_FLAG} \
        --batch_size ${BATCH_SIZE} \
        --use_trt \
        --optimize_offline \
        --precision ${TFTRT_PRECISION} \
        --max_workspace_size $((2**32)) \
        --input_size ${INPUT_SIZE} \
        --preprocess_method ${PREPROCESS_METHOD} \
        --num_classes ${NUM_CLASSES}"
fi

echo -e "**Executing:**\n\n${COMMAND}\n"
sleep 5

eval ${COMMAND}