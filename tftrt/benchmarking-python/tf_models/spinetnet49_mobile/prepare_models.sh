#!/usr/bin/env bash

pip install tf-models-official==2.9.2

wget https://raw.githubusercontent.com/tensorflow/models/v2.9.2/official/vision/configs/experiments/retinanet/coco_spinenet49_mobile_tpu.yaml \
    -O coco_spinenet49_mobile_tpu_fp16.yaml

sed 's/bfloat16/float32/g' coco_spinenet49_mobile_tpu_fp16.yaml > coco_spinenet49_mobile_tpu_fp32.yaml

BATCH_SIZES=(
  "1"
  "8"
  "16"
  "32"
  "64"
  "128"
)

MODEL_DIR="/models/tf_models/spinetnet49_mobile"

for batch_size in "${BATCH_SIZES[@]}"; do

    python -m official.vision.serving.export_saved_model \
        --experiment="retinanet_mobile_coco" \
        --checkpoint_path="${MODEL_DIR}/checkpoint/" \
        --config_file="coco_spinenet49_mobile_tpu_fp32.yaml" \
        --export_dir="${MODEL_DIR}/" \
        --export_saved_model_subdir="saved_model_bs${batch_size}" \
        --input_image_size=384,384 \
        --batch_size="${batch_size}"

    saved_model_cli show --dir "${MODEL_DIR}/saved_model_bs${batch_size}/" --all 2>&1 \
        | tee "${MODEL_DIR}/saved_model_bs${batch_size}/analysis.txt"

done
