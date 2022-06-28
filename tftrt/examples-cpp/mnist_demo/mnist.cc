/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mnist.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <stdint.h>

namespace mnist {

namespace {

// This function interprets a 4 byte array as an unsigned,
// big-endian integer. sizeof(data) must be 4.
// From Rob Pike's Byte Order Fallacy:
// https://commandcenter.blogspot.com/2012/04/byte-order-fallacy.html
uint32_t ConvertBigEndian(unsigned char data[]) {
  uint32_t result =
      (data[3] << 0) | (data[2] << 8) | (data[1] << 16) | (data[0] << 24);
  return result;
}

} // namespace

std::ostream &operator<<(std::ostream &os, const MNISTImage &image) {
  for (int row = 0; row < MNISTImage::kSize; ++row) {
    for (int column = 0; column < MNISTImage::kSize; ++column) {
      os << (image.buf[column + MNISTImage::kSize * row] > 0 ? "X " : "  ");
    }
    os << "\n";
  }
  return os;
}

tensorflow::Tensor MNISTImageReader::MNISTImageToTensor(int offset,
                                                        int batch_size) {
  // https://github.com/tensorflow/tensorflow/issues/8033#issuecomment-520977062
  tensorflow::Tensor input_image(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape(
          {batch_size, MNISTImage::kSize, MNISTImage::kSize}));
  float *img_tensor_flat = input_image.flat<float>().data();
  constexpr int N = MNISTImage::kSize * MNISTImage::kSize;
  for (int i = 0; i < batch_size; i++) {
    std::copy_n(images[i].buf.data() + offset * N, N, img_tensor_flat + i * N);
  }
  return input_image;
}

MNISTImageReader::MNISTImageReader(const std::string &path)
    : mnist_path_(path) {}

// MNIST's file format is documented here: http://yann.lecun.com/exdb/mnist/
// Note(bmzhao): The serialized integers are in big endian, and the magic
// number is documented to be 2051.
tensorflow::Status MNISTImageReader::ReadMnistImages() {
  std::ifstream image_file(mnist_path_, std::ios::binary);
  if (!image_file.is_open()) {
    return tensorflow::errors::NotFound("Error opening file", mnist_path_, ": ",
                                        std::strerror(errno));
  }

  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_columns;

  auto ReadBigEndian = [](std::ifstream &image_file) {
    unsigned char buf[4];
    image_file.read(reinterpret_cast<char *>(&buf[0]), sizeof(buf));
    return ConvertBigEndian(buf);
  };

  // Read the magic number.
  uint32_t magic_number = ReadBigEndian(image_file);
  if (magic_number != 2051) {
    return tensorflow::errors::Internal("Magic Number of Mnist Data File ",
                                        mnist_path_, " was ", magic_number,
                                        " expected 2051");
  }

  // Read the number of images.
  num_images = ReadBigEndian(image_file);
  std::cout << "Number of images " << num_images << std::endl;
  // Read the number of rows.
  num_rows = ReadBigEndian(image_file);
  if (num_rows != MNISTImage::kSize) {
    return tensorflow::errors::FailedPrecondition(
        "Num Rows of Mnist Data File was ", num_rows, " expected 28");
  }

  // Read the number of columns
  num_columns = ReadBigEndian(image_file);
  if (num_columns != MNISTImage::kSize) {
    return tensorflow::errors::FailedPrecondition(
        "Num Columns of Mnist Data File was ", num_columns, " expected 28");
  }

  // Iterate through the images, and create an MNISTImage struct for each
  for (int i = 0; i < num_images; ++i) {
    images.emplace_back();
    uint8_t img_buf[MNISTImage::kSize * MNISTImage::kSize];
    image_file.read(reinterpret_cast<char *>(&img_buf[0]), sizeof(img_buf));

    // Convert the buffer into float MNISTImage
    std::copy_n(img_buf, MNISTImage::kSize * MNISTImage::kSize,
                images[i].buf.data());
  }
  if (images.empty()) {
    return tensorflow::errors::Internal("Error reading MNIST images");
  }
  return tensorflow::Status::OK();
}

} // namespace mnist
