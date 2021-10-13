#!/bin/bash

# Runtime Parameters
MODEL_NAME=""
DATA_DIR=""
MODEL_DIR=""

BATCH_SIZE=8
NUM_ITERATIONS_FLAG=""

# TF-TRT Parameters
USE_TFTRT=0
TFTRT_PRECISION="FP32"

# Default Argument Values
TF_XLA_FLAGS=""
NVIDIA_TF32_OVERRIDE=""
USE_SYNTHETIC_DATA_FLAG=""
USE_DYNAMIC_SHAPE_FLAG=""
SKIP_ACCURACY_TESTING_FLAG=""
INPUT_SIGNATURE_KEY_FLAG=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --use_xla)
        TF_XLA_FLAGS="TF_XLA_FLAGS=--tf_xla_auto_jit=2"
        shift # Remove --use_xla from processing
        ;;
        --no_tf32)
        NVIDIA_TF32_OVERRIDE="NVIDIA_TF32_OVERRIDE=0"
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
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        --use_tftrt)
        USE_TFTRT=1
        shift # Remove --use_tftrt from processing
        ;;
        --precision=*)
        TFTRT_PRECISION="${arg#*=}"
        shift # Remove --precision= from processing
        ;;
        --use_synthetic_data)
        USE_SYNTHETIC_DATA_FLAG="--use_synthetic_data"
        shift # Remove --use_synthetic_data from processing
        ;;
        --use_dynamic_shape)
        USE_DYNAMIC_SHAPE_FLAG="--use_dynamic_shape"
        shift # Remove --use_dynamic_shape from processing
        ;;
        --input_signature_key=*)
        INPUT_SIGNATURE_KEY_FLAG="--input_signature_key=${arg#*=}"
        shift # Remove --input_signature_key from processing
        ;;
        --num_iterations=*)
        NUM_ITERATIONS_FLAG="--num_iterations=${arg#*=}"
        shift # Remove --num_iterations from processing
        ;;
    esac
done

# ============== Set model specific parameters ============= #

MIN_SEGMENT_SIZE=2
MAX_WORKSPACE_SIZE=$((2**32))

case ${MODEL_NAME} in
  "faster_rcnn_resnet50_coco" | "ssd_mobilenet_v1_fpn_coco")
    MIN_SEGMENT_SIZE=4
    MAX_WORKSPACE_SIZE=$((2**24))
    ;;
esac

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] NUM_ITERATIONS_FLAG: ${NUM_ITERATIONS_FLAG}"
echo ""
echo "[*] NVIDIA_TF32_OVERRIDE: ${NVIDIA_TF32_OVERRIDE}"
echo "[*] TF_XLA_FLAGS: ${TF_XLA_FLAGS}"
echo ""
echo "[*] INPUT_SIGNATURE_KEY_FLAG: ${INPUT_SIGNATURE_KEY_FLAG}"
echo "[*] SKIP_ACCURACY_TESTING_FLAG: ${SKIP_ACCURACY_TESTING_FLAG}"
echo "[*] USE_SYNTHETIC_DATA_FLAG: ${USE_SYNTHETIC_DATA_FLAG}"
echo ""
echo "[*] USE_TFTRT: ${USE_TFTRT}"
echo "[*] TFTRT_PRECISION: ${TFTRT_PRECISION}"
echo "[*] MAX_WORKSPACE_SIZE: ${MAX_WORKSPACE_SIZE}"
echo "[*] MIN_SEGMENT_SIZE: ${MIN_SEGMENT_SIZE}"
echo "[*] USE_DYNAMIC_SHAPE_FLAG: ${USE_DYNAMIC_SHAPE_FLAG}"
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

# TFTRT Arguments

ALLOWED_TFTRT_PRECISION="FP32 FP16 INT8"

if ! $(echo ${ALLOWED_TFTRT_PRECISION} | grep -w ${TFTRT_PRECISION} > /dev/null); then
    echo "ERROR: Unknown TFTRT_PRECISION received: \`${TFTRT_PRECISION}\`. [Allowed: ${ALLOWED_TFTRT_PRECISION}]"
fi

# %%%%%%%%%%%%%%%%%%%%%%% ARGUMENT VALIDATION %%%%%%%%%%%%%%%%%%%%%%% #

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd ${BENCH_DIR}

# Step 1: Installing dependencies if needed:
python -c "from pycocotools.coco import COCO" > /dev/null 2>&1
DEPENDENCIES_STATUS=$?

if [[ ${DEPENDENCIES_STATUS} != 0 ]]; then
    bash install_dependencies.sh
fi

# Step 2: Execute the example

PREPEND_COMMAND="TF_CPP_MIN_LOG_LEVEL=2 ${TF_XLA_FLAGS} ${NVIDIA_TF32_OVERRIDE}"

COMMAND="${PREPEND_COMMAND} python object_detection.py \
    --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
    --output_saved_model_dir /tmp/$RANDOM \
    --data_dir ${VAL_DATA_DIR} \
    --calib_data_dir ${VAL_DATA_DIR} \
    --annotation_path ${ANNOTATIONS_DATA_FILE} \
    --num_warmup_iterations 100 \
    --display_every 50 \
    ${SKIP_ACCURACY_TESTING_FLAG} \
    ${USE_SYNTHETIC_DATA_FLAG} \
    ${INPUT_SIGNATURE_KEY_FLAG} \
    --input_size 640 \
    --batch_size ${BATCH_SIZE} \
    ${NUM_ITERATIONS_FLAG}"

if [[ ${USE_TFTRT} != "0" ]]; then
    COMMAND="${COMMAND} \
        --use_tftrt \
        --optimize_offline \
        --precision ${TFTRT_PRECISION} \
        --minimum_segment_size ${MIN_SEGMENT_SIZE} \
        ${USE_DYNAMIC_SHAPE_FLAG} \
        --max_workspace_size ${MAX_WORKSPACE_SIZE}"
fi

echo -e "**Executing:**\n\n${COMMAND}\n"
sleep 5

eval ${COMMAND}
