#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BASE_BENCHMARK_DATA_EXPORT_DIR="${BASE_DIR}/benchmark_data"
rm -rf ${BASE_BENCHMARK_DATA_EXPORT_DIR}
mkdir -p ${BASE_BENCHMARK_DATA_EXPORT_DIR}

# List of models. spice is a single implementation
MODELS=(
    "spice"
)

RUN_ARGS="--data_dir=/tmp/ --input_saved_model_dir=/models/tf_hub/spice --display_every=50 --samples_per_input=128 --num_iterations=1000"
TF_TRT_ARGS="--use_tftrt --use_dynamic_shape --num_calib_batches=10"
TF_XLA_ARGS="--use_xla_auto_jit"

export TF_TRT_SHOW_DETAILED_REPORT=1

for model_name in "${MODELS[@]}"; do
    echo "Processing Model: ${model_name} ..."

    MODEL_DATA_EXPORT_DIR="${BASE_BENCHMARK_DATA_EXPORT_DIR}/${model_name}"
    mkdir -p ${MODEL_DATA_EXPORT_DIR}

    # ============================ TF NATIVE ============================ #
    # TF Native - FP32
    echo "Running ${BASE_DIR}/run_inference.sh"
    script -q -c "${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=FP32" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_native_fp32.log

    # TF Native - FP16
    script -q -c "${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=FP16" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_native_fp16.log

    # ============================ TF XLA ============================ #
    # TF XLA - FP32
    script -q -c "${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=FP32 ${TF_XLA_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_xla_fp32.log

    # TF XLA - FP16
    script -q -c "${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=FP16 ${TF_XLA_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_xla_fp16.log

    # ============================ TF-TRT ============================ #
    # TF-TRT - FP32
    script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_fp32.dot ${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=FP32 ${TF_TRT_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_fp32.log

    # TF-TRT - FP16
    script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_fp16.dot ${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=FP16 ${TF_TRT_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_fp16.log

    # TF-TRT - INT8
    script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_int8.dot ${BASE_DIR}/run_inference.sh ${RUN_ARGS} --precision=INT8 ${TF_TRT_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_int8.log
done
