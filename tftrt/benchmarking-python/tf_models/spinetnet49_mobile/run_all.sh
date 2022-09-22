#!/bin/bash

SCRIPT_DIR=""

EXPERIMENT_NAME="spinetnet49_mobile"

BASE_BENCHMARK_DATA_EXPORT_DIR="/workspace/benchmark_data/${EXPERIMENT_NAME}"
rm -rf ${BASE_BENCHMARK_DATA_EXPORT_DIR}
mkdir -p ${BASE_BENCHMARK_DATA_EXPORT_DIR}

EXPERIMENT_FLAG=""

#########################

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BENCHMARK_DATA_EXPORT_DIR="${BASE_BENCHMARK_DATA_EXPORT_DIR}/tf_models/"
mkdir -p ${BENCHMARK_DATA_EXPORT_DIR}

model_name="spinetnet49_mobile"

RUN_ARGS="${EXPERIMENT_FLAG} --data_dir=/tmp --input_saved_model_dir=/models/tf_models/${model_name}/saved_model_bs1/ "
RUN_ARGS="${RUN_ARGS} --debug --batch_size=1 --display_every=5 --use_synthetic_data --num_warmup_iterations=200 --num_iterations=500"
TF_TRT_ARGS="--use_tftrt --use_dynamic_shape --num_calib_batches=10"
TF_XLA_ARGS="--use_xla_auto_jit"

export TF_TRT_SHOW_DETAILED_REPORT=1
# export TF_TRT_BENCHMARK_EARLY_QUIT=1

MODEL_DATA_EXPORT_DIR="${BENCHMARK_DATA_EXPORT_DIR}/${model_name}"
mkdir -p ${MODEL_DATA_EXPORT_DIR}

SCRIPT_PATH="${BASE_DIR}/run_inference.sh"
METRICS_JSON_FLAG="--export_metrics_json_path=${MODEL_DATA_EXPORT_DIR}"

# TF Native
script -q -c "${SCRIPT_PATH} ${RUN_ARGS} --precision=FP32" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_fp32.log
script -q -c "${SCRIPT_PATH} ${RUN_ARGS} --precision=FP16" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_fp16.log

# TF-XLA
script -q -c "${SCRIPT_PATH} ${RUN_ARGS} ${TF_XLA_ARGS} --precision=FP32" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tfxla_fp32.log
script -q -c "${SCRIPT_PATH} ${RUN_ARGS} ${TF_XLA_ARGS} --precision=FP16" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tfxla_fp16.log

# TF-TRT
script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_fp32.dot ${SCRIPT_PATH} ${RUN_ARGS} ${TF_TRT_ARGS} --precision=FP32" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_fp32.log
script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_fp16.dot ${SCRIPT_PATH} ${RUN_ARGS} ${TF_TRT_ARGS} --precision=FP16" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_fp16.log
script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_int8.dot ${SCRIPT_PATH} ${RUN_ARGS} ${TF_TRT_ARGS} --precision=INT8" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_int8.log
