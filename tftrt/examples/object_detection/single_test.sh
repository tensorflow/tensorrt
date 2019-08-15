CUDA_VISIBLE_DEVICES=0
common_args="
    --data_dir /data/coco/coco-2017/coco2017/val2017 \
    --annotation_path /data/coco/coco-2017/coco2017/annotations/instances_val2017.json \
    --gpu_mem_cap 6892 \
    --num_images 2048 \
    --num_warmup_iterations 10 \
    --cache \
    --use_synthetic \
    --display_every 128 \
    --input_size 640"

OUTPUT_PATH=./outputs
model=ssd_mobilenet_v2_coco
SAVED_MODEL_DIR=./saved_models/${model}_640_bs1
#python object_detection.py $common_args --batch_size 1 \
#    --saved_model_dir $SAVED_MODEL_DIR 2>&1 | tee \
#    $OUTPUT_PATH/output_tf_fp32_bs${i}_${model}
#python object_detection.py $common_args --batch_size $i \
#    --saved_model_dir $SAVED_MODEL_DIR 2>&1 --use_trt --precision FP32 | tee \
#    $OUTPUT_PATH/output_tftrt_fp32_bs${i}_${model}
#TF_CPP_VMODULE=segment=2,convert_graph=2,convert_nodes=2,trt_engine=1,trt_logger=1 \
python object_detection.py $common_args --batch_size 1 \
    --saved_model_dir $SAVED_MODEL_DIR --use_trt --precision FP16 2>&1 | tee \
    $OUTPUT_PATH/output_tftrt_fp16_bs${i}_${model}
TF_CPP_VMODULE=segment=2,convert_graph=2,convert_nodes=2,trt_engine=1,trt_logger=1 \
#i=1
#python object_detection.py $common_args --batch_size 1 \
#    --saved_model_dir $SAVED_MODEL_DIR --use_trt --precision INT8 2>&1 | tee \
#    $OUTPUT_PATH/tftrt_int8_bs${i}_${model}
