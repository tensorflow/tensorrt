/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <google/protobuf/map.h>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mnist.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// Returns the name of nodes listed in the signature definition.
std::vector<std::string>
GetNodeNames(const google::protobuf::Map<std::string, tensorflow::TensorInfo>
                 &signature) {
  std::vector<std::string> names;
  for (auto const &item : signature) {
    absl::string_view name = item.second.name();
    // Remove tensor suffix like ":0".
    size_t last_colon = name.find_last_of(':');
    if (last_colon != absl::string_view::npos) {
      name.remove_suffix(name.size() - last_colon);
    }
    names.push_back(std::string(name));
  }
  return names;
}

// Loads a SavedModel from export_dir into the SavedModelBundle.
tensorflow::Status LoadModel(const std::string &export_dir,
                             tensorflow::SavedModelBundle *bundle,
                             std::vector<std::string> *input_names,
                             std::vector<std::string> *output_names) {

  tensorflow::RunOptions run_options;
  TF_RETURN_IF_ERROR(tensorflow::LoadSavedModel(tensorflow::SessionOptions(),
                                                run_options, export_dir,
                                                {"serve"}, bundle));

  // Print the signature defs.
  auto signature_map = bundle->GetSignatures();
  for (const auto &name_and_signature_def : signature_map) {
    const auto &name = name_and_signature_def.first;
    const auto &signature_def = name_and_signature_def.second;
    std::cerr << "Name: " << name << std::endl;
    std::cerr << "SignatureDef: " << signature_def.DebugString() << std::endl;
  }

  // Extract input and output tensor names from the signature def.
  const tensorflow::SignatureDef &signature = signature_map["serving_default"];
  *input_names = GetNodeNames(signature.inputs());
  *output_names = GetNodeNames(signature.outputs());

  return tensorflow::Status::OK();
}

// We prepare to input sets, with batch size 1 and batch size 4.
tensorflow::Status
LoadInputs(const std::string &mnist_path,
           std::vector<std::vector<tensorflow::Tensor>> *inputs) {
  // Load the MNIST images from the given path.
  mnist::MNISTImageReader image_reader(mnist_path);
  TF_RETURN_IF_ERROR(image_reader.ReadMnistImages());

  // Convert the first image to a tensorflow::Tensor.
  tensorflow::Tensor input_image = image_reader.MNISTImageToTensor(0, 1);
  std::cout << "Created input tensor with shape "
            << input_image.shape().DebugString() << std::endl;
  inputs->push_back({input_image});

  // Create another tensor with batch size 4.
  input_image = image_reader.MNISTImageToTensor(0, 4);
  std::cout << "Created input tensor with shape "
            << input_image.shape().DebugString() << std::endl;
  inputs->push_back({input_image});

  for (int i = 0; i < 4; i++) {
    std::cout << "Input image " << i << std::endl;
    std::cout << image_reader.images[i] << std::endl;
  }
  return tensorflow::Status::OK();
}

#define TFTRT_ENSURE_OK(x)                                                     \
  do {                                                                         \
    tensorflow::Status s = x;                                                  \
    if (!s.ok()) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << s.error_message()     \
                << std::endl;                                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  // One can use a command line arg to specify the saved model directory.
  // Note that the model has to be trained with input size [None, 28, 28].
  // Currently, it is assumed that the model is already frozen (variables are
  // converted to constants).
  std::string export_dir =
      "/workspace/tensorflow-source/tf_trt_cpp_example/mnist_model_frozen";
  std::string mnist_data_path =
      "/workspace/tensorflow-source/tf_trt_cpp_example/"
      "t10k-images.idx3-ubyte";
  bool frozen_graph = false;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("saved_model_dir", &export_dir,
                       "Path to saved model directory"),
      tensorflow::Flag("mnist_data", &mnist_data_path, "Path to MNIST images"),
      tensorflow::Flag("frozen_graph", &frozen_graph,
                       "Assume graph is frozen and use TF-TRT API for frozen "
                       "graphs")};
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    LOG(ERROR) << usage;
    return -1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::SavedModelBundle bundle;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Load the saved model from the provided path.
  TFTRT_ENSURE_OK(LoadModel(export_dir, &bundle, &input_names, &output_names));

  // Prepare input tensors
  std::vector<std::vector<tensorflow::Tensor>> inputs;
  TFTRT_ENSURE_OK(LoadInputs(mnist_data_path, &inputs));

  // Run TF-TRT conversion
  tensorflow::tensorrt::TfTrtConversionParams params;
  params.use_dynamic_shape = true;
  params.profile_strategy = tensorflow::tensorrt::ProfileStrategy::kOptimal;
  tensorflow::StatusOr<tensorflow::GraphDef> status_or_gdef;
  if (frozen_graph) {
    status_or_gdef = tensorflow::tensorrt::ConvertAndBuild(
        bundle.meta_graph_def.graph_def(), input_names, output_names, inputs,
        params);
  } else {
    status_or_gdef = tensorflow::tensorrt::ConvertAndBuild(
        &bundle, "serving_default", inputs, params);
  }
  if (!status_or_gdef.ok()) {
    std::cerr << "Error converting the graph" << status_or_gdef.status()
              << std::endl;
    return 1;
  }
  tensorflow::GraphDef &converted_graph_def = status_or_gdef.ValueOrDie();
  tensorflow::Session *session = nullptr;
  TFTRT_ENSURE_OK(NewSession(tensorflow::SessionOptions(), &session));
  bundle.session.reset(session);
  TFTRT_ENSURE_OK(bundle.session->Create(converted_graph_def));

  // Infer the converted model
  for (auto const &input : inputs) {
    std::vector<std::pair<std::string, tensorflow::Tensor>> input_pairs;
    for (int i = 0; i < input_names.size(); i++) {
      input_pairs.push_back({input_names.at(i), input.at(i)});
    }
    std::vector<tensorflow::Tensor> output_tensors;
    for (int i = 0; i < output_names.size(); i++) {
      output_tensors.push_back({});
    }
    std::cout << "Inferring the model" << std::endl;
    TFTRT_ENSURE_OK(
        session->Run(input_pairs, output_names, {}, &output_tensors));

    for (const auto &output_tensor : output_tensors) {
      const auto &vec = output_tensor.flat_inner_dims<float>();
      int m = output_tensor.dim_size(1);
      for (int k = 0; k < output_tensor.dim_size(0); k++) {
        float max = 0;
        int argmax = 0;
        for (int i = 0; i < m; ++i) {
          if (vec(i + k * m) > max) {
            argmax = i;
          }
        }
        std::cerr << "Sample " << k << ", Predicted Number: " << argmax
                  << std::endl;
      }
    }
  }
}
