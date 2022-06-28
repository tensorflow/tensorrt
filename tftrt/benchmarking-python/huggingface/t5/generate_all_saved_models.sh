#!/usr/bin/bash
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd /tmp

git lfs version || exit_code=$?
if [[ ${exit_code} -ne 0 ]]; then
    echo "Installing Git LFS"
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
    tar -xzf git-lfs-linux-amd64-v3.2.0.tar.gz
    bash $(pwd)/git-lfs-3.2.0/install.sh
else
    echo "Git LFS already installed"
fi


pip install -U transformers[sentencepiece]

cd ${SCRIPT_DIR}

T5_MODELS=(
    "t5-small"
    "t5-base"
    "t5-large"
    "google/t5-v1_1-base"
)

for MODEL_NAME in "${T5_MODELS[@]}"; do
    echo "---------------------------------------------------------------------"
    echo "Processing Model: ${MODEL_NAME} ..."

    MODEL_DIR=${SCRIPT_DIR}/${MODEL_NAME#*/}   # Get the substring after "/"

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "${MODEL_DIR} does NOT exist on your filesystem."
        git clone https://huggingface.co/${MODEL_NAME}
    else
        echo "${MODEL_DIR} exist on your filesystem."
    fi

    cd ${MODEL_DIR}
    git lfs pull
    cd ${SCRIPT_DIR}

    OUTPUT_DIR="${MODEL_DIR}/saved_models"
    rm -rf ${OUTPUT_DIR}

    sleep 2

    python generate_saved_models.py \
        --model_name=${MODEL_NAME} \
        --output_directory=${OUTPUT_DIR}

    SAVED_MODEL_DIR="${OUTPUT_DIR}/model"
    script -q -c "saved_model_cli show --dir ${SAVED_MODEL_DIR} --all" /dev/null | tee "${SAVED_MODEL_DIR}/analysis.txt"

done
