#!/bin/bash

nvidia-smi

# Runtime Parameters
MODEL_NAME=""
DATA_DIR=""
MODEL_DIR=""

# Default Argument Values
NVIDIA_TF32_OVERRIDE=""

BYPASS_ARGUMENTS=""
TF_AUTO_JIT_XLA_FLAG=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --no_tf32)
        NVIDIA_TF32_OVERRIDE="NVIDIA_TF32_OVERRIDE=0"
        shift # Remove --no_tf32 from processing
        ;;
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        shift # Remove --data_dir= from processing
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        --output_tensors_name=*)
        shift # Remove --output_tensors_name= from processing
        ;;
        --output_tensor_indices=*)
        shift # Remove --output_tensor_indices= from processing
        ;;
        --use_xla_auto_jit)
        TF_AUTO_JIT_XLA_FLAG="TF_XLA_FLAGS=--tf_xla_auto_jit=2"
        shift # Remove --use_xla_auto_jit from processing
        ;;
        *)
        BYPASS_ARGUMENTS=" ${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

# ============== Set model specific parameters ============= #

INPUT_SIZE=224
PREPROCESS_METHOD="vgg"
NUM_CLASSES=1001
OUTPUT_TENSOR_NAME_FLAG=""
OUTPUT_TENSOR_IDX_FLAG=""

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

  "resnet50-v1.5_tf1_ngc" )
    NUM_CLASSES=1000
    OUTPUT_TENSOR_IDX_FLAG="--output_tensor_indices=0"
    OUTPUT_TENSOR_NAME_FLAG="--output_tensors_name=classes"
    PREPROCESS_METHOD="resnet50_v1_5_tf1_ngc_preprocess"
    ;;

  "resnet50v2_backbone" | "resnet50v2_sparse_backbone" )
    INPUT_SIZE=256
    ;;
esac

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo ""
echo "[*] NVIDIA_TF32_OVERRIDE: ${NVIDIA_TF32_OVERRIDE}"
echo ""
# Custom Image Classification Task Flags
echo "[*] INPUT_SIZE: ${INPUT_SIZE}"
echo "[*] PREPROCESS_METHOD: ${PREPROCESS_METHOD}"
echo "[*] NUM_CLASSES: ${NUM_CLASSES}"
echo "[*] OUTPUT_TENSOR_IDX_FLAG: ${OUTPUT_TENSOR_IDX_FLAG}"
echo "[*] OUTPUT_TENSOR_NAME_FLAG: ${OUTPUT_TENSOR_NAME_FLAG}"
echo ""
echo "[*] TF_AUTO_JIT_XLA_FLAG: ${TF_AUTO_JIT_XLA_FLAG}"
echo "[*] BYPASS_ARGUMENTS: $(echo \"${BYPASS_ARGUMENTS}\" | tr -s ' ')"
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

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd ${BENCH_DIR}

# Execute the example

PREPEND_COMMAND="${TF_AUTO_JIT_XLA_FLAG} ${NVIDIA_TF32_OVERRIDE}"

COMMAND="${PREPEND_COMMAND} python image_classification.py \
    --data_dir ${DATA_DIR} \
    --calib_data_dir ${DATA_DIR} \
    --input_saved_model_dir ${INPUT_SAVED_MODEL_DIR} \
    --output_saved_model_dir /tmp/$RANDOM \
    --input_size ${INPUT_SIZE} \
    --preprocess_method ${PREPROCESS_METHOD} \
    --num_classes ${NUM_CLASSES} \
    ${OUTPUT_TENSOR_IDX_FLAG} \
    ${OUTPUT_TENSOR_NAME_FLAG} \
    ${BYPASS_ARGUMENTS}"

COMMAND=$(echo "${COMMAND}" | tr -s " ")

echo -e "**Executing:**\n\n${COMMAND}\n"
sleep 5

eval ${COMMAND}
