#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

#define TFTRT_ENSURE_OK(x)                                                 \
  do {                                                                     \
    Status s = x;                                                          \
    if (!s.ok()) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << s.error_message() \
                << std::endl;                                              \
      return 1;                                                            \
    }                                                                      \
  } while (0)

// Get the name of the GPU.
const string GetDeviceName(std::unique_ptr<tensorflow::Session>& session) {
  string device_name = "";
  std::vector<tensorflow::DeviceAttributes> devices;
  Status status = session->ListDevices(&devices);
  if (!status.ok()) {
    return device_name;
  }
  for (const auto& d : devices) {
    if (d.device_type() == "GPU" || d.device_type() == "gpu") {
      device_name = d.name();
    }
  }
  return device_name;
}

// Move from the host to the device with `device_name`.
Status MoveToDevice(const string& device_name, Tensor& tensor_host,
                    Tensor* tensor_device) {
  // Create identity graph
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto x = tensorflow::ops::Placeholder(root, tensor_host.dtype());
  auto y = tensorflow::ops::Identity(root, x);

  tensorflow::GraphDef graphDef;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graphDef));

  // Create session with identity graph
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graphDef));

  // Configure to return output on device
  tensorflow::Session::CallableHandle handle;
  tensorflow::CallableOptions opts;
  opts.add_feed("Placeholder:0");
  opts.add_fetch("Identity:0");
  opts.mutable_fetch_devices()->insert({"Identity:0", device_name});
  opts.set_fetch_skip_sync(true);
  TF_RETURN_IF_ERROR(session->MakeCallable(opts, &handle));

  // Execute graph
  std::vector<Tensor> tensors_device;
  Status status =
      session->RunCallable(handle, {tensor_host}, &tensors_device, nullptr);
  *tensor_device = tensors_device.front();
  session->ReleaseCallable(handle);
  return status;
}

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

// Create arbitrary inputs matching `input_info` and load them on the device.
Status SetupInputs(const string& device_name, int32_t batch_size,
                   std::vector<tensorflow::TensorInfo>& input_info,
                   std::vector<Tensor>* inputs) {
  std::vector<Tensor> inputs_device;
  for (const auto& info : input_info) {
    auto shape = info.tensor_shape();
    shape.mutable_dim(0)->set_size(batch_size);
    Tensor input_host(info.dtype(), shape);
    Tensor input_device;
    TF_RETURN_IF_ERROR(MoveToDevice(device_name, input_host, &input_device));
    inputs_device.push_back(input_device);
  }
  *inputs = inputs_device;
  return Status::OK();
}

// Configure a `CallableHandle` that feeds from and fetches to a device.
Status SetupCallable(std::unique_ptr<tensorflow::Session>& session,
                     std::vector<tensorflow::TensorInfo>& input_info,
                     std::vector<tensorflow::TensorInfo>& output_info,
                     const string& device_name,
                     tensorflow::Session::CallableHandle* handle) {
  tensorflow::CallableOptions opts;
  for (const auto& info : input_info) {
    const string& name = info.name();
    opts.add_feed(name);
    opts.mutable_feed_devices()->insert({name, device_name});
  }
  for (const auto& info : output_info) {
    const string& name = info.name();
    opts.add_fetch(name);
    opts.mutable_fetch_devices()->insert({name, device_name});
  }
  opts.set_fetch_skip_sync(true);
  return session->MakeCallable(opts, handle);
}

int main(int argc, char* argv[]) {
  // Parse arguments
  string model_path = "/path/to/model/";
  string signature_key = "serving_default";
  int32_t batch_size = 64;
  int32_t warmup_iters = 50;
  int32_t eval_iters = 1000;
  std::vector<Flag> flag_list = {
      Flag("model_path", &model_path, "graph to be executed"),
      Flag("signature_key", &signature_key, "the serving signature to use"),
      Flag("batch_size", &batch_size, "batch size to use for inference"),
      Flag("warmup_iters", &warmup_iters, "number of warmup iterations to run"),
      Flag("eval_iters", &eval_iters, "number of timed iterations to run"),
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

  // Setup session
  tensorflow::SavedModelBundle bundle;
  std::vector<tensorflow::TensorInfo> input_info;
  std::vector<tensorflow::TensorInfo> output_info;
  TFTRT_ENSURE_OK(
      LoadModel(model_path, signature_key, &bundle, &input_info, &output_info));

  // Create inputs and move to device
  const string device_name = GetDeviceName(bundle.session);
  std::vector<Tensor> inputs_device;
  TFTRT_ENSURE_OK(
      SetupInputs(device_name, batch_size, input_info, &inputs_device));

  // Configure to feed and fetch from device
  tensorflow::Session::CallableHandle handle;
  TFTRT_ENSURE_OK(SetupCallable(bundle.session, input_info, output_info,
                                device_name, &handle));

  // Run benchmarking
  std::vector<Tensor> outputs;
  std::vector<double> infer_time;
  std::chrono::steady_clock::time_point eval_start_time;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  for (int i = 0; i < warmup_iters + eval_iters; i++) {
    if (i == warmup_iters) {
      LOG(INFO) << "Warmup done";
      eval_start_time = std::chrono::steady_clock::now();
    }

    start_time = std::chrono::steady_clock::now();
    Status status =
        bundle.session->RunCallable(handle, inputs_device, &outputs, nullptr);
    end_time = std::chrono::steady_clock::now();

    TFTRT_ENSURE_OK(status);
    double duration = (end_time - start_time).count() / 1e6;
    infer_time.push_back(duration);
  }
  TFTRT_ENSURE_OK(bundle.session->ReleaseCallable(handle));

  // Print results
  std::sort(infer_time.begin() + warmup_iters, infer_time.end());
  double total_compute_time =
      std::accumulate(infer_time.begin() + warmup_iters, infer_time.end(), 0.0);
  double total_wall_time = (end_time - eval_start_time).count() / 1e6;
  int32_t m = warmup_iters + eval_iters / 2;
  LOG(INFO) << "Total wall time (s): " << total_wall_time / 1e3;
  LOG(INFO) << "Total GPU compute time (s): " << total_compute_time / 1e3;
  LOG(INFO) << "Mean GPU compute time (ms): " << total_compute_time / eval_iters;
  LOG(INFO) << "Median GPU compute time (ms): " << (eval_iters % 2 ? infer_time[m]
                                                                   : (infer_time[m - 1] + infer_time[m]) / 2);
  // Note: Throughput using GPU inference time, rather than wall time
  LOG(INFO) << "Throughput (samples/s): " << 1e3 * eval_iters * batch_size / total_compute_time;
  LOG(INFO) << "First inference latency (ms): " << infer_time.front();

  return 0;
}