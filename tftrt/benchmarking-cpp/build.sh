#! /bin/bash

BUILD_DIR="./build"

if [ -d "${BUILD_DIR}" ]; then
  echo "Found old cpp benchmark build directory, deleting it..."
  rm -rf ${BUILD_DIR};
  echo "Done."
fi


echo "Building TFTRT CPP benchmark..."
mkdir build
cd build
cmake ..
make
echo "Done"
