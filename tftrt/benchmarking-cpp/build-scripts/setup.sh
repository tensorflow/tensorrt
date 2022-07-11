TF_DIR=/opt/tensorflow
SRC_DIR=$TF_DIR/tensorflow-source/tensorflow/examples/benchmarking-cpp
CUR_DIR=$(dirname $(dirname $(readlink -fm $0)))

ln -s $CUR_DIR $SRC_DIR
patch $TF_DIR/tensorflow-source/tensorflow/core/profiler/rpc/client/BUILD $SRC_DIR/build-scripts/tf-profiler.patch
ln -s $SRC_DIR/build-scripts/tftrt-build.sh $TF_DIR