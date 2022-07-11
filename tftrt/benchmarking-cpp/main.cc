#include <chrono>
#include <iostream>
#include <numeric>
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
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
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

// Get the GPU and its default context.
Status GetDevice(std::unique_ptr<tensorflow::Session>& session,
                 tensorflow::Device** device,
                 tensorflow::DeviceContext** device_context) {
  const tensorflow::DeviceMgr* mgr;
  TF_RETURN_IF_ERROR(session->LocalDeviceManager(&mgr));
  std::vector<tensorflow::Device*> devices = mgr->ListDevices();
  for (const auto& d : devices) {
    if (d->device_type() == "GPU" || d->device_type() == "gpu") {
      auto* device_info = d->tensorflow_gpu_device_info();
      *device_context = device_info->default_context;
      *device = d;
    }
  }
  return Status::OK();
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

  tensorflow::OptimizerOptions* optimizer_options =
      sess_options.config.mutable_graph_options()->mutable_optimizer_options();
  optimizer_options->set_opt_level(tensorflow::OptimizerOptions::L0);
  optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);

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
Status SetupInputs(tensorflow::Device* device,
                   tensorflow::DeviceContext* device_context,
                   int32_t batch_size,
                   std::vector<tensorflow::TensorInfo>& input_info,
                   std::vector<Tensor>* inputs) {
  tensorflow::AllocatorAttributes attr;
  tensorflow::Allocator* allocator = device->GetAllocator(attr);

  std::vector<Tensor> inputs_device;
  for (const auto& info : input_info) {
    // Set input batch size
    auto shape = info.tensor_shape();
    shape.mutable_dim(0)->set_size(batch_size);

    // Allocate memory and fill host tensor
    Tensor input_host(info.dtype(), shape);
    Tensor input_device(allocator, info.dtype(), shape);
    std::fill_n((uint8_t*)input_host.data(), input_host.AllocatedBytes(), 1);

    // Copy from host to device
    TF_RETURN_IF_ERROR(device_context->CopyCPUTensorToDeviceSync(
        &input_host, device, &input_device));
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
                     bool input_from_device,
                     bool output_to_host,
                     tensorflow::Session::CallableHandle* handle) {
  tensorflow::CallableOptions opts;
  for (const auto& info : input_info) {
    const string& name = info.name();
    opts.add_feed(name);
    if (input_from_device) {
      opts.mutable_feed_devices()->insert({name, device_name});
    }
  }
  for (const auto& info : output_info) {
    const string& name = info.name();
    opts.add_fetch(name);
    if (!output_to_host) {
      opts.mutable_fetch_devices()->insert({name, device_name});
    }
  }
  opts.set_fetch_skip_sync(true);
  return session->MakeCallable(opts, handle);
}

// Start the profiling session.
Status StartProfiling(std::unique_ptr<tensorflow::ProfilerSession>& profiler) {
  profiler = tensorflow::ProfilerSession::Create(
      tensorflow::ProfilerSession::DefaultOptions()
  );
  return profiler->Status();
}

// Tear down the profiler and export tensorboard logs.
Status StopProfiling(std::unique_ptr<tensorflow::ProfilerSession>& profiler,
                     const string& out_dir) {
  tensorflow::profiler::XSpace xspace;
  TF_RETURN_IF_ERROR(profiler->CollectData(&xspace));
  tensorflow::profiler::ExportToTensorBoard(xspace, out_dir);
  profiler.reset();
  return Status::OK();
}

int main(int argc, char* argv[]) {
  // Parse arguments
  string model_path = "/path/to/model/";
  string signature_key = "serving_default";
  int32_t batch_size = 64;
  int32_t warmup_iters = 200;
  int32_t eval_iters = 800;
  bool input_from_device = true;
  bool output_to_host = true;
  string out_dir = "";
  std::vector<Flag> flag_list = {
      Flag("model_path", &model_path, "graph to be executed"),
      Flag("signature_key", &signature_key, "the serving signature to use"),
      Flag("batch_size", &batch_size, "batch size to use for inference"),
      Flag("warmup_iters", &warmup_iters, "number of warmup iterations to run"),
      Flag("eval_iters", &eval_iters, "number of timed iterations to run"),
      Flag("input_from_device", &input_from_device, "use inputs from device, rather than host"),
      Flag("output_to_host", &output_to_host, "copy outputs to host after inference"),
      Flag("out_dir", &out_dir, "if set, runs the profiler and exports to this directory"),
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

  // TODO: Convert model w/ TRT and add flag for this behavior

  // Get device
  tensorflow::Device* device;
  tensorflow::DeviceContext* device_context;
  TFTRT_ENSURE_OK(GetDevice(bundle.session, &device, &device_context));

  // Create inputs and move to device
  // TODO: Measure H2D times over repeated calls and report metrics
  std::vector<Tensor> inputs_device;
  TFTRT_ENSURE_OK(SetupInputs(device, device_context, batch_size, input_info,
                              &inputs_device));

  // Configure to feed and fetch from device
  tensorflow::Session::CallableHandle handle;
  TFTRT_ENSURE_OK(SetupCallable(bundle.session, input_info, output_info,
                                device->name(), input_from_device, output_to_host, &handle));

  // Run benchmarking
  std::vector<Tensor> outputs;
  std::vector<double> infer_time;
  std::chrono::steady_clock::time_point eval_start_time;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  std::unique_ptr<tensorflow::ProfilerSession> profiler;
  for (int i = 0; i < warmup_iters + eval_iters; i++) {
    if (i == warmup_iters) {
      LOG(INFO) << "Warmup done";
      if (!out_dir.empty()) {
        StartProfiling(profiler);
      }
      eval_start_time = std::chrono::steady_clock::now();
    }

    {
      tensorflow::profiler::TraceMe trace([&i, &warmup_iters]() {
        return tensorflow::profiler::TraceMeEncode(
          "gpu_compute", {{"iter", i - warmup_iters}}
        );
      }, 1);
      start_time = std::chrono::steady_clock::now();
      TFTRT_ENSURE_OK(
          bundle.session->RunCallable(handle, inputs_device, &outputs, nullptr));
      // Sync, as `set_fetch_skip_sync(false)` is currently not implemented
      TFTRT_ENSURE_OK(device->Sync());
      end_time = std::chrono::steady_clock::now();
    }

    if ((i % 10) == 0) {
      LOG(INFO) << "step: " << i;
    }

    double duration = (end_time - start_time).count() / 1e6;
    infer_time.push_back(duration);
  }
  if (!out_dir.empty()) {
    StopProfiling(profiler, out_dir);
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
  LOG(INFO) << "Engine build time + first inference latency (ms): " << infer_time.front();

  return 0;
}
