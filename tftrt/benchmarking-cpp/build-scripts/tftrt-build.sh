# TODO: to programatically determine the python and tf API versions
PYVER=3.8 #TODO get this by parsing `python --version`
TFAPI=2 #TODO get this by parsing tf.__version__

/opt/tensorflow/nvbuild.sh --configonly --python$PYVER --v$TFAPI

BUILD_OPTS="$(cat /opt/tensorflow/nvbuildopts)"
if [[ "$TFAPI" == "2" ]]; then
  BUILD_OPTS="--config=v2 $BUILD_OPTS"
fi

cd tensorflow-source
bazel build $BUILD_OPTS tensorflow/examples/benchmarking-cpp/...
