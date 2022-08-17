#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

// Some helper routines
#define TFTRT_ENSURE_OK(x)                                                 \
  do {                                                                     \
    Status s = x;                                                          \
    if (!s.ok()) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << s.error_message() \
                << std::endl;                                              \
      return 1;                                                            \
    }                                                                      \
  } while (0)

// Returns info for nodes listed in the signature definition.
std::vector<tensorflow::TensorInfo> GetNodeInfo(
    const google::protobuf::Map<string, tensorflow::TensorInfo>& signature) {
  std::vector<tensorflow::TensorInfo> info;
  for (const auto& item : signature) {
    info.push_back(item.second);
  }
  return info;
}

// Load the `SavedModel` located at `model_dir`.
Status LoadModel(const string& model_dir, const string& signature_key,
                 tensorflow::SavedModelBundle* bundle,
                 std::vector<tensorflow::TensorInfo>* input_info,
                 std::vector<tensorflow::TensorInfo>* output_info) {
  tensorflow::RunOptions run_options;
  tensorflow::SessionOptions sess_options;

  sess_options.config.mutable_gpu_options()->force_gpu_compatible();
  TF_RETURN_IF_ERROR(tensorflow::LoadSavedModel(sess_options, run_options,
                                                model_dir, {"serve"}, bundle));

  // Get input and output names
  auto signature_map = bundle->GetSignatures();
  const tensorflow::SignatureDef& signature = signature_map[signature_key];
  *input_info = GetNodeInfo(signature.inputs());
  *output_info = GetNodeInfo(signature.outputs());

  return Status::OK();
}

// Create random inputs matching `input_info`
Status SetupInputs(int32_t batch_size,
                   int32_t input_size,
                   std::vector<tensorflow::TensorInfo>& input_info,
                   std::vector<std::pair<std::string, tensorflow::Tensor>>* inputs) {

  //std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& info : input_info) {
      // Set input batch size
      auto* shape = info.mutable_tensor_shape();
      shape->mutable_dim(0)->set_size(batch_size);

      // Set dynamic dims to static size
      for (size_t i = 1; i < shape->dim_size(); i++) {
          auto* dim = shape->mutable_dim(i);
          if (dim->size() < 0) {
            dim->set_size(input_size);
          }
      }
      
      // Allocate memory and fill host tensor
      Tensor input_tensor(info.dtype(), *shape);
      std::fill_n((uint8_t*)input_tensor.data(), input_tensor.AllocatedBytes(), 1);

      inputs->push_back({info.name(), input_tensor});
  }

  return Status::OK();
}

// Get output tensor names based on `output_info`.
Status SetupOutputs(std::vector<tensorflow::TensorInfo>& output_info,
                   std::vector<string>* output_names,
                   std::vector<Tensor>* outputs) {
  for (auto& info : output_info) {
    output_names->push_back(info.name());
    outputs->push_back({});
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {
  // Parse arguments
  string model_path = "/path/to/model/";
  string signature_key = "serving_default";
  int32_t batch_size = 64;
  int32_t input_size = 128;
  std::vector<Flag> flag_list = {
      Flag("model_path", &model_path, "graph to be executed"),
      Flag("signature_key", &signature_key, "the serving signature to use"),
      Flag("batch_size", &batch_size, "batch size to use for inference"),
      Flag("input_size", &input_size, "shape to use for -1 input dims"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // Setup TF session
  tensorflow::SavedModelBundle bundle;
  std::vector<tensorflow::TensorInfo> input_info;
  std::vector<tensorflow::TensorInfo> output_info;
  TFTRT_ENSURE_OK(
      LoadModel(model_path, signature_key, &bundle, &input_info, &output_info));

  // Setup inputs
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  TFTRT_ENSURE_OK(SetupInputs(batch_size, input_size, input_info, &inputs));

  // Setup outputs
  std::vector<string> output_names;
  std::vector<Tensor> outputs;
  TFTRT_ENSURE_OK(SetupOutputs(output_info, &output_names, &outputs));

  int num_iterations = 10;
  for (int i = 0; i < num_iterations; i++) {
    LOG(INFO) << "Step: " << i;

    TFTRT_ENSURE_OK(
        bundle.session->Run(inputs, output_names, {}, &outputs));
  }

  return 0;
}