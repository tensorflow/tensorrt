<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
=======
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
#!/bin/bash
=======
#!/bin/#!/usr/bin/env bash
>>>>>>> Modify file endings and permissions

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BASE_BENCHMARK_DATA_EXPORT_DIR="${BASE_DIR}/benchmark_data"
rm -rf ${BASE_BENCHMARK_DATA_EXPORT_DIR}
mkdir -p ${BASE_BENCHMARK_DATA_EXPORT_DIR}

# Default Argument Values
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
SAMPLES_PER_INPUT="128"
BYPASS_ARGUMENTS=""
MODEL_DIR="/models/tf_hub/spice"
<<<<<<< refs/remotes/origin/master
DATA_DIR="/tmp/"
NUM_ITERATIONS="1000"

# List of models. spice is a single implementation
MODELS=(
    "spice"
)

RUN_ARGS="--data_dir=${DATA_DIR} --input_saved_model_dir=${MODEL_DIR} --display_every=50 --samples_per_input=${SAMPLES_PER_INPUT} --num_iterations=${NUM_ITERATIONS}"
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
=======
# Runtime Parameters
MODEL_NAME="spice"

# Default Argument Values
=======
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
SAMPLES_PER_INPUT=128
BYPASS_ARGUMENTS=""
=======
SAMPLES_PER_INPUT="128"
BYPASS_ARGUMENTS=""
>>>>>>> Modify file endings and permissions
MODEL_DIR="/models/tf_hub/"
=======
>>>>>>> Address review comments
DATA_DIR="/tmp/"
NUM_ITERATIONS="1000"

# List of models. spice is a single implementation
MODELS=(
    "spice"
)
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
RUN_ARGS="--data_dir=/workspace/tftrt/benchmarking-python/tf_hub/albert/data --input_saved_model_dir=/models/tf_hub/albert --tokenizer_dir=/models/tf_hub/albert/tokenizer --debug --batch_size=32 --display_every=1"
=======
RUN_ARGS="--data_dir=/tmp --input_saved_model_dir=/models/tf_hub/spice --batch_size=1 --display_every=50"
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
=======
RUN_ARGS="--data_dir=/tmp --input_saved_model_dir=/models/tf_hub/spice --batch_size=1 --display_every=50 --samples_per_input=${SAMPLES_PER_INPUT} --num_iterations=${NUM_ITERATIONS}"
>>>>>>> Modify file endings and permissions
=======

RUN_ARGS="--data_dir=${DATA_DIR} --input_saved_model_dir=${MODEL_DIR} --display_every=50 --samples_per_input=${SAMPLES_PER_INPUT} --num_iterations=${NUM_ITERATIONS}"
>>>>>>> Address review comments
TF_TRT_ARGS="--use_tftrt --use_dynamic_shape --num_calib_batches=10"
TF_XLA_ARGS="--use_xla_auto_jit"

export TF_TRT_SHOW_DETAILED_REPORT=1

<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
for model_name in "${ALBERT_MODELS[@]}"; do
    echo "Processing Model: ${model_name} ..."

    MODEL_DATA_EXPORT_DIR="${BENCHMARK_DATA_EXPORT_DIR}/${model_name}"
    mkdir -p ${MODEL_DATA_EXPORT_DIR}

    MODEL_PROFILE_DIR="${MODEL_DATA_EXPORT_DIR}/tf_profiles"
    mkdir -p ${MODEL_PROFILE_DIR}
    TF_PROFILE_ARG="--tf_profile_export_path=${MODEL_PROFILE_DIR}"

=======
=======
>>>>>>> Modify file endings and permissions
for model_name in "${MODELS[@]}"; do
    echo "Processing Model: ${model_name} ..."

    MODEL_DATA_EXPORT_DIR="${BASE_BENCHMARK_DATA_EXPORT_DIR}/${model_name}"
    mkdir -p ${MODEL_DATA_EXPORT_DIR}

<<<<<<< refs/remotes/origin/master
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
    # ============================ TF NATIVE ============================ #
    # TF Native - FP32
=======
    # ============================ TF NATIVE ============================ #
    # TF Native - FP32
<<<<<<< refs/remotes/origin/master
    echo "${BASE_DIR}/run_inference.sh"  
>>>>>>> Modify file endings and permissions
=======
    echo "Running ${BASE_DIR}/run_inference.sh"  
>>>>>>> Address review comments
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
<<<<<<< refs/remotes/origin/master
<<<<<<< refs/remotes/origin/master
>>>>>>> [Benchmarking-py] add run_all.sh script
=======
#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BASE_BENCHMARK_DATA_EXPORT_DIR="${BASE_DIR}/benchmark_data"
rm -rf ${BASE_BENCHMARK_DATA_EXPORT_DIR}
mkdir -p ${BASE_BENCHMARK_DATA_EXPORT_DIR}

# Default Argument Values
SAMPLES_PER_INPUT=128
BYPASS_ARGUMENTS=""
MODEL_DIR="/models/tf_hub/"
DATA_DIR="/tmp/"
NUM_ITERATIONS="1000"

MODELS=(
    "spice"
)
RUN_ARGS="--data_dir=/tmp --input_saved_model_dir=/models/tf_hub/spice --batch_size=1 --display_every=50"
TF_TRT_ARGS="--use_tftrt --use_dynamic_shape --num_calib_batches=10"
TF_XLA_ARGS="--use_xla_auto_jit"

export TF_TRT_SHOW_DETAILED_REPORT=1

for model_name in "${MODELS[@]}"; do
    echo "Processing Model: ${model_name} ..."

    MODEL_DATA_EXPORT_DIR="${BASE_BENCHMARK_DATA_EXPORT_DIR}/${model_name}"
    mkdir -p ${MODEL_DATA_EXPORT_DIR}

    # ============================ TF NATIVE ============================ #
    # TF Native - FP32
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
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
=======
>>>>>>> [Benchmarking-Py] Spice - run_all.sh
=======
>>>>>>> Modify file endings and permissions
