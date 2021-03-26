#!/bin/bash

MODEL_NAME="nasnet_mobile"
BATCH_SIZE=32

# Default values of arguments
USE_XLA=0
USE_TF32=1
USE_TFTRT=0
TFTRT_PRECISION="FP32"

DATA_DIR=""
MODEL_DIR=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --use_xla)
        USE_XLA=1
        shift # Remove --xla from processing
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

echo -e "\n********************************************************************"
echo "MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo ""
echo "[*] USE_XLA: ${USE_XLA}"
echo "[*] TF_XLA_FLAGS: ${TF_XLA_FLAGS}"
echo ""
echo "[*] USE_TF32: ${USE_TF32}"
echo "[*] NVIDIA_TF32_OVERRIDE: ${NVIDIA_TF32_OVERRIDE}"
echo ""
echo "[*] USE_TFTRT: ${USE_TFTRT}"
echo "[*] TFTRT_PRECISION: ${TFTRT_PRECISION}"
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
        --mode validation \
        --num_warmup_iterations 100 \
        --display_every 50 \
        --batch_size ${BATCH_SIZE} \
        --preprocess_method inception"
else
    COMMAND="${PREPEND_COMMAND} python image_classification.py \
        --data_dir ${DATA_DIR} \
        --calib_data_dir ${DATA_DIR} \
        --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
        --output_saved_model_dir /tmp/${RANDOM}/ \
        --mode validation \
        --num_warmup_iterations 100 \
        --display_every 50 \
        --batch_size ${BATCH_SIZE} \
        --preprocess_method inception \
        --use_trt \
        --optimize_offline \
        --precision ${TFTRT_PRECISION} \
        --max_workspace_size $((2**32))"
fi

echo -e "**Executing:**\n\n${COMMAND}\n"
sleep 5

eval ${COMMAND}