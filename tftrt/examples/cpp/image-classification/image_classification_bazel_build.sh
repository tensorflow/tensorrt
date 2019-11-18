#!/bin/bash
# Build the components of tensorflow that require Bazel


# Inputs: 
# 	OUTPUT_DIRS - String of space-delimited directories to store outputs, in order of:
# 			1)kernel test list 
# 			2)xla test list 
# 			3)tensorflow whl
# 	TESTLIST - Determines whether the test lists are built (1 to build, 0 to skip)
# 	NOCLEAN - Determines whether bazel clean is run and the tensorflow whl is 
# 			removed after the build and install (0 to clean, 1 to skip)
# 	PYVER - The version of python
# 	BUILD_OPTS - File containing desired bazel flags for building tensorflow
# 	LIBCUDA_FOUND - Determines whether a libcuda stub was created and needs to be cleaned (0 to clean, 1 to skip)
# 	IN_CONTAINER - Flag for whether Tensorflow is being built within a container (1 for yes, 0 for bare-metal)
#       TF_API - TensorFlow API version: 1 => v1.x, 2 => 2.x
#

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"


read -ra OUTPUT_LIST <<<"$OUTPUT_DIRS"
KERNEL_OUT=${OUTPUT_LIST[0]}
XLA_OUT=${OUTPUT_LIST[1]}
WHL_OUT=${OUTPUT_LIST[2]}

for d in ${OUTPUT_LIST[@]}
do
  mkdir -p ${d}
done

# +
KERNEL_TEST_RETURN=0
XLA_TEST_RETURN=0
BAZEL_BUILD_RETURN=0
if [[ "$TF_API" == "2" ]]; then
  BAZEL_OPTS="--config=v2 $(cat $BUILD_OPTS)"
else
  BAZEL_OPTS="$(cat $BUILD_OPTS)"
fi

bazel build tensorflow/examples/image-classification/...

