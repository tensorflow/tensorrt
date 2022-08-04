#!/usr/bin/bash
set -x

BASE_DIR="/models/huggingface/gpt2"

GPT2_MODELS=(
  "gpt2"
  "gpt2-medium"
  "gpt2-large"
  "gpt2-xl"
)


for model_name in "${GPT2_MODELS[@]}"; do
    echo "Processing: ${model_name} ..."
    sleep 2

    MODEL_DIR="${BASE_DIR}/${model_name}"
    rm -rf ${MODEL_DIR} && mkdir -p ${MODEL_DIR}
    python generate_saved_models.py --output_directory ${MODEL_DIR} --model_name ${model_name}
    saved_model_cli show --dir "${MODEL_DIR}/model" --all 2>&1 | tee ${MODEL_DIR}/model/analysis.txt
done
