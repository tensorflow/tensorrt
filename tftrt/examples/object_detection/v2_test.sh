OUTPUT_PATH=$PWD

common_args="
    --data_dir /data/coco/coco-2017/coco2017/val2017 \
    --annotation_path /data/coco/coco-2017/coco2017/annotations/instances_val2017.json \
    --gpu_mem_cap 0 \
    --num_images 2048 \
    --num_warmup_iterations 10 \
    --cache \
    --display_every 64 \
    --input_size 640"

SAVED_MODEL_PREFIX=$PWD/saved_models
INPUT_SIZE=640
SCRIPTS_PATH=/tensorflow/qa/inference/object_detection

batch_sizes=(1 8 128)
models=(
    mask_rcnn_resnet50_atrous_coco
    ssd_inception_v2_coco
    ssdlite_mobilenet_v2_coco
    ssd_mobilenet_v1_coco
    ssd_mobilenet_v1_fpn_coco
    ssd_mobilenet_v2_coco
    ssd_resnet_50_fpn_coco
    faster_rcnn_resnet50_coco
    faster_rcnn_nas
)

for model in ${models[@]};
do
  for i in ${batch_sizes[@]};
  do
    SAVED_MODEL_DIR=$SAVED_MODEL_PREFIX/${model}_${INPUT_SIZE}_bs${i}
    python object_detection.py $common_args --batch_size $i \
        --saved_model_dir $SAVED_MODEL_DIR 2>&1 | tee \
        $OUTPUT_PATH/output_tf_fp32_bs${i}_${model}
    python object_detection.py $common_args --batch_size $i \
        --saved_model_dir $SAVED_MODEL_DIR 2>&1 --use_trt --precision FP32 | tee \
        $OUTPUT_PATH/output_tftrt_fp32_bs${i}_${model}
    python object_detection.py $common_args --batch_size $i \
        --saved_model_dir $SAVED_MODEL_DIR 2>&1 --use_trt --precision FP16 | tee \
        $OUTPUT_PATH/output_tftrt_fp16_bs${i}_${model}
    pushd $SCRIPTS_PATH
    python -u check_map.py --input_path $OUTPUT_PATH/output_tf_fp32_bs${i}_${model} \
      --model ${model}
    python -u check_map.py --input_path $OUTPUT_PATH/output_tftrt_fp32_bs${i}_${model} \
      --model ${model}
    python -u check_map.py --input_path $OUTPUT_PATH/output_tftrt_fp16_bs${i}_${model} \
      --model ${model}
    popd
  done
done
