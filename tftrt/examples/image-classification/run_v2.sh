(python image_classification.py \
    --model mobilenet_v1 \
    --precision INT8 \
    --use_trt_dynamic_op \
    --mode benchmark \
    --batch_size 8 \
    --num_iterations 200 \
    --calib_data_dir /data/imagenet/train-val-tfrecord \
    --use_trt \
    --data_dir /data/imagenet/val-jpeg \
    --num_calib_inputs 8 3>&1 1>&2 2>&3 | grep -v "W0808") 3>&1 1>&2 2>&3
    #--data_dir /data/imagenet/train-val-tfrecord \
    #--cache \ 

