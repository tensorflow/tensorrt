BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BASE_BENCHMARK_DATA_EXPORT_DIR="${BASE_DIR}/benchmark_data"
rm -rf ${BASE_BENCHMARK_DATA_EXPORT_DIR}
mkdir -p ${BASE_BENCHMARK_DATA_EXPORT_DIR}

#########################

swin_MODELS=(
  "swin_s3_tiny_224"
  "swin_s3_small_224"
  "swin_s3_base_224"
  "swin_tiny_patch4_window7_224"
  "swin_small_patch4_window7_224"
  "swin_base_patch4_window7_224"
  "swin_large_patch4_window7_224"
  "swin_base_patch4_window12_384"
  "swin_large_patch4_window12_384"
)

MODEL_DIR="/models/tf_hub/swin_transformers"
DATA_DIR="/data/imagenet"

RUN_ARGS="--input_saved_model_dir=${MODEL_DIR} --data_dir=${DATA_DIR} --debug --batch_size=32 --display_every=1 --use_synthetic_data --num_iterations=1000"
TF_TRT_ARGS="--use_tftrt --use_dynamic_shape --num_calib_batches=10"
TF_XLA_ARGS="--use_xla_auto_jit"

export TF_TRT_SHOW_DETAILED_REPORT=1

for model_name in "${swin_MODELS[@]}"; do

  echo "Processing Model: ${model_name} ..."

  MODEL_DATA_EXPORT_DIR="${BASE_BENCHMARK_DATA_EXPORT_DIR}/${model_name}"
  mkdir -p ${MODEL_DATA_EXPORT_DIR}

  SUBSCRIPT_DIR="${BASE_DIR}/models/${model_name}/run_inference.sh"

  # ============================ TF NATIVE ============================ #
  # TF Native - FP32
  script -q -c "${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=FP32" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_native_fp32.log

  # TF Native - FP16
  script -q -c "${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=FP16" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_native_fp16.log

  # ============================ TF XLA ============================ #
  # TF XLA - FP32
  script -q -c "${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=FP32 ${TF_XLA_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_xla_fp32.log

  # TF XLA - FP16
  script -q -c "${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=FP16 ${TF_XLA_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tf_xla_fp16.log

  # ============================ TF-TRT ============================ #
  # TF-TRT - FP32
  script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_fp32.dot ${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=FP32 ${TF_TRT_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_fp32.log

  # TF-TRT - FP16
  script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_fp16.dot ${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=FP16 ${TF_TRT_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_fp16.log

  # TF-TRT - INT8
  script -q -c "TF_TRT_EXPORT_GRAPH_VIZ_PATH=${MODEL_DATA_EXPORT_DIR}/tftrt_int8.dot ${SUBSCRIPT_DIR} ${RUN_ARGS} --precision=INT8 ${TF_TRT_ARGS}" /dev/null | tee ${MODEL_DATA_EXPORT_DIR}/inference_tftrt_int8.log

done
