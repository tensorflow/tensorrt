#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Runtime Parameters
MODEL_NAME=""

# Default Argument Values
BATCH_SIZE=32
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
OUTPUT_TENSOR_NAMES="output_0"
TOTAL_MAX_SAMPLES=50000
=======

OUTPUT_TENSOR_NAMES="output_0"
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
OUTPUT_TENSOR_NAMES="output_0"
TOTAL_MAX_SAMPLES=50000
>>>>>>> Fixing Shell Scripts

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
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
        TOTAL_MAX_SAMPLES="${arg#*=}"
        shift # Remove --total_max_samples= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
=======
        shift # Remove --total_max_samples= from processing
        ;;
        *)
        BYPASS_ARGUMENTS=" ${BYPASS_ARGUMENTS} ${arg}"
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
        TOTAL_MAX_SAMPLES="${arg#*=}"
        shift # Remove --total_max_samples= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
>>>>>>> Fixing Shell Scripts
        ;;
    esac
done

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
# Trimming front and back whitespaces
BYPASS_ARGUMENTS=$(echo ${BYPASS_ARGUMENTS} | tr -s " ")

=======
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
# Trimming front and back whitespaces
BYPASS_ARGUMENTS=$(echo ${BYPASS_ARGUMENTS} | tr -s " ")

>>>>>>> Fixing Shell Scripts
echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"
=======
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
=======
>>>>>>> Fix preprocessing for vit
echo ""
echo "[*] BATCH_SIZE: ${BATCH_SIZE}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
<<<<<<< refs/remotes/origin/master
echo "[*] BYPASS_ARGUMENTS: $(echo \"${BYPASS_ARGUMENTS}\" | tr -s ' ')"
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
echo "[*] BYPASS_ARGUMENTS: ${BYPASS_ARGUMENTS}"
>>>>>>> Fixing Shell Scripts

echo -e "********************************************************************\n"

MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}"

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
set -x
=======

<<<<<<< refs/remotes/origin/master
# Dataset Directory
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
set -x
>>>>>>> Fixing Shell Scripts

=======
>>>>>>> Fix preprocessing for vit
python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --input_saved_model_dir=${MODEL_DIR} \
    --batch_size=${BATCH_SIZE} \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
    --total_max_samples=${TOTAL_MAX_SAMPLES} \
    ${BYPASS_ARGUMENTS}
=======
    `# The following is set because we will be running synthetic benchmarks` \
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
    --total_max_samples=1 \
    ${@}
>>>>>>> Init vit scripts. Needs to tune args in scripts
=======
    --total_max_samples=50000 \
    ${@}
>>>>>>> Fix preprocessing for vit
=======
    --total_max_samples=1 \
    ${@}
>>>>>>> switch back to synthetic data and clean up
=======
    --total_max_samples=${TOTAL_MAX_SAMPLES} \
    ${BYPASS_ARGUMENTS}
>>>>>>> Fixing Shell Scripts
