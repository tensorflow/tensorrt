#!/usr/bin/bash

BASE_DIR="/models/huggingface/transformers"

GPT2_MODELS=(
    "bert-base-uncased"
    "bert-base-cased"
    "bert-large-uncased"
    "bert-large-cased"
    "facebook/bart-base"
    "facebook/bart-large"
)


for model_name in "${GPT2_MODELS[@]}"; do
    echo "Processing: ${model_name} ..."
    sleep 1

    MODEL_DIR=${model_name#*/}  # Remove `facebook/`
    MODEL_DIR=${MODEL_DIR//-/_}  # Replace "-" by "_"

    MODEL_DIR="${BASE_DIR}/${MODEL_DIR}"
    rm -rf ${MODEL_DIR} && mkdir -p ${MODEL_DIR}
    echo "Model Dir: ${MODEL_DIR}"

    set -x
    python generate_saved_models.py --output_directory ${MODEL_DIR} --model_name ${model_name}
    saved_model_cli show --dir "${MODEL_DIR}/pb_model" --all 2>&1 | tee ${MODEL_DIR}/pb_model/analysis.txt
    set +x
done
