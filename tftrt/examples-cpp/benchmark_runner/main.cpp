#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::Flag;

#define TFTRT_ENSURE_OK(x)                                                     \
  do {                                                                         \
    tensorflow::Status s = x;                                                  \
    if (!s.ok()) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << s.error_message()     \
                << std::endl;                                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Get the name of the GPU.
string getDeviceName(std::unique_ptr<tensorflow::Session>& session) {
    string device_name = "";
    std::vector<tensorflow::DeviceAttributes> devices;
    Status status = session->ListDevices(&devices);
    if (!status.ok()) { return device_name; }
    for (const auto& d : devices) {
        if (d.device_type() == "GPU" || d.device_type() == "gpu") {
            device_name = d.name();
        }
    }
    return device_name;
}

// Move tensors from the host to the device with `device_name`.
Status moveToDevice(string& device_name,
                    std::vector<Tensor>& tensors_host,
                    std::vector<Tensor>* tensors_device) {
    // Create identity graph
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    auto x = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
    auto y = tensorflow::ops::Identity(root, x);

    tensorflow::GraphDef graphDef;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graphDef));

    // Create session with identity graph
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions())
    );
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
    Status status = session->RunCallable(handle, tensors_host, tensors_device, nullptr);
    session->ReleaseCallable(handle);
    return status;
}

// Returns the name of nodes listed in the signature definition.
std::vector<std::string> getNodeNames(
    const google::protobuf::Map<std::string, tensorflow::TensorInfo>& signature) {
  std::vector<std::string> names;
  for (auto const &item : signature) {
    names.push_back(item.second.name());
  }
  return names;
}

Status loadModel(const std::string& model_dir,
                 tensorflow::SavedModelBundle* bundle,
                 std::vector<string>* input_names,
                 std::vector<string>* output_names) {
    tensorflow::RunOptions run_options;
    tensorflow::SessionOptions sess_options;
    sess_options.config.mutable_gpu_options()->force_gpu_compatible();
    TF_RETURN_IF_ERROR(tensorflow::LoadSavedModel(sess_options, run_options, model_dir, {"serve"}, bundle));

    // Get input and output names
    auto signature_map = bundle->GetSignatures();
    const tensorflow::SignatureDef& signature = signature_map["serving_default"];
    *input_names = getNodeNames(signature.inputs());
    *output_names = getNodeNames(signature.outputs());

    return tensorflow::Status::OK();
}

// Configure a `CallableHandle` that feeds from and fetches to a device.
Status setupCallable(std::unique_ptr<tensorflow::Session>& session,
                     std::vector<string>& input_names,
                     std::vector<string>& output_names,
                     string& device_name,
                     tensorflow::Session::CallableHandle* handle) {
    tensorflow::CallableOptions opts;
    for (const auto& name : input_names) {
        opts.add_feed(name);
        opts.mutable_feed_devices()->insert({name, device_name});
    }
    for (const auto& name : output_names) {
        opts.add_fetch(name);
        opts.mutable_fetch_devices()->insert({name, device_name});
    }
    opts.set_fetch_skip_sync(true);
    return session->MakeCallable(opts, handle);
}

int main(int argc, char* argv[]) {
    // Parse arguments
    string model_path = "/path/to/model/";
    int32_t batch_size = 64;
    int32_t image_size = 224;
    int32_t image_channels = 3;
    int32_t warmup_iters = 50;
    int32_t eval_iters = 1000;
    std::vector<Flag> flag_list = {
        Flag("model_path", &model_path, "graph to be executed"),
        Flag("batch_size", &batch_size, "batch size to use for inference"),
        Flag("image_size", &image_size, "size of the input image"),
        Flag("image_channels", &image_channels, "number of channels for the input image"),
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
    std::vector<string> input_names;
    std::vector<string> output_names;
    TFTRT_ENSURE_OK(loadModel(model_path, &bundle, &input_names, &output_names));

    // Create inputs and move to device
    Tensor input(tensorflow::DT_FLOAT, {batch_size, image_size, image_size, image_channels});
    std::vector<Tensor> inputs_host = {input};
    std::vector<Tensor> inputs_device;
    std::vector<Tensor> outputs;
    string device_name = getDeviceName(bundle.session);
    TFTRT_ENSURE_OK(moveToDevice(device_name, inputs_host, &inputs_device));

    // Configure to feed and fetch from device
    tensorflow::Session::CallableHandle handle;
    TFTRT_ENSURE_OK(setupCallable(bundle.session, input_names, output_names, device_name, &handle));

    // Run benchmarking
    std::vector<double> infer_time;

    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    for (int i = 0; i < warmup_iters + eval_iters; i++) {
        if (i == warmup_iters) {
            LOG(INFO) << "Warmup done";
        }

        start_time = std::chrono::steady_clock::now();
        Status status = bundle.session->RunCallable(handle, inputs_device, &outputs, nullptr);
        end_time = std::chrono::steady_clock::now();
        
        TFTRT_ENSURE_OK(status);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        infer_time.push_back((double)duration);
    }
    TFTRT_ENSURE_OK(bundle.session->ReleaseCallable(handle));

    // Print results
    LOG(INFO) << "First inference time (ms): " << infer_time.front();
    LOG(INFO) << "Last inference time (ms): " << infer_time.back();
    LOG(INFO) << "Mean GPU time (ms): " << accumulate(infer_time.begin() + warmup_iters, infer_time.end(), 0.0) / eval_iters;
    LOG(INFO) << "Throughput (ims/s): " << 1000 * eval_iters * batch_size / accumulate(infer_time.begin() + warmup_iters, infer_time.end(), 0.0);

    return 0;
}