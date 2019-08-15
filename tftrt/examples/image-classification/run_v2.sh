#export TF_CPP_MIN_VLOG_LEVEL=1 
(python -u image_classification.py \
    --model resnet_v1_50 \
    --precision FP16 \
    --mode benchmark \
    --batch_size 1 \
    --num_iterations 2000 \
    --calib_data_dir /data/imagenet/train-val-tfrecord \
	--gpu_mem_cap 7000 \
    --use_trt \
    --use_trt_dynamic_op \
	--root_saved_model_dir ./saved_models \
	--data_dir /data/imagenet/val-jpeg \
	--engine_dir ./converted \
	--target_duration 120 \
    --num_calib_inputs 8 3>&1 1>&2 2>&3 | grep -v "reference variable") 3>&1 1>&2 2>&3
	#--use_synthetic \
    #--data_dir /data/imagenet/train-val-tfrecord \
	#--saved_model_dir ./saved_models/resnet_v1_50 \

