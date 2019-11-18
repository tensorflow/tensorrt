#!/bin/bash
#
# Configure, build, and install Tensorflow
#

# Exit at error
set -e

Usage() {
  echo "Configure, build, and install Tensorflow."
  echo ""
  echo "  Usage: $0 [OPTIONS]"
  echo ""
  echo "    OPTIONS          DESCRIPTION"
  echo "    --python2.7      Build python2.7 package (default)"
  echo "    --python3.6      Build python3.6 package"
  echo "    --configonly     Run configure step only"
  echo "    --noconfig       Skip configure step"
  echo "    --noclean        Retain intermediate build files"
  echo "    --testlist       Build list of python kernel_tests"
  echo "    --v1             Build TensorFlow v1 API"
  echo "    --v2             Build TensorFlow v2 API"
}

PYVER=2.7
CONFIGONLY=0
NOCONFIG=0
NOCLEAN=0
TESTLIST=0
TF_API=1

while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")  Usage; exit 1 ;;
    "--python2.7")  PYVER=2.7 ;;
    "--python3.6")  PYVER=3.6 ;;
    "--configonly") CONFIGONLY=1 ;;
    "--noconfig")   NOCONFIG=1 ;;
    "--noclean")    NOCLEAN=1 ;;
    "--testlist")   TESTLIST=1 ;;
    "--v1")         TF_API=1 ;;
    "--v2")         TF_API=2 ;;
    *)
      echo UNKNOWN OPTION $1
      echo Run $0 -h for help
      exit 1
  esac
  shift 1
done

export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDA_VERSION=$(echo "${CUDA_VERSION}" | cut -d . -f 1-2)
export TF_CUBLAS_VERSION=$(echo "${CUDA_VERSION}" | cut -d . -f 1)
export TF_CUDNN_VERSION=$(echo "${CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)
export TF_TENSORRT_VERSION=$(echo "${TRT_VERSION}" | cut -d . -f 1)
export TF_CUDA_COMPUTE_CAPABILITIES="5.2,6.0,6.1,7.0,7.5"
export TF_ENABLE_XLA=1
export TF_NEED_HDFS=0
export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${THIS_DIR}/tensorflow-source"
export PYTHON_BIN_PATH=/usr/bin/python$PYVER
LIBCUDA_FOUND=$(ldconfig -p | awk '{print $1}' | grep libcuda.so | wc -l)
if [[ $NOCONFIG -eq 0 ]]; then
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
      ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
  yes "" | ./configure
fi

if [[ $CONFIGONLY -eq 1 ]]; then
  exit 0
fi


export OUTPUT_DIRS="tensorflow/python/kernel_tests tensorflow/compiler/tests /tmp/pip"
export BUILD_OPTS="${THIS_DIR}/nvbuildopts"
export IN_CONTAINER="1"
export TESTLIST
export NOCLEAN
export PYVER
export LIBCUDA_FOUND
export TF_API
bash ${THIS_DIR}/image_classification_bazel_build.sh

