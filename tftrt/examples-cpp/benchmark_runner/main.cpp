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
#include "runner.h"

using tensorflow::Flag;

int main(int argc, char* argv[]) {
    // Parse arguments
    string graph =
        "/opt/tensorflow/tensorflow-source/tensorflow/examples/benchmark_runner/toy_model/frozen/frozen_model.pb";
    int32_t batch_size = 32;
    string input_layer = "x";
    string output_layer = "Identity";
    int32_t warmup_iters = 50;
    int32_t eval_iters = 1000;
    bool trace = false;
    std::vector<Flag> flag_list = {
        Flag("graph", &graph, "graph to be executed"),
        Flag("batch_size", &batch_size, "batch size to use for inference"),
        Flag("input_layer", &input_layer, "name of input layer"),
        Flag("output_layer", &output_layer, "name of output layer"),
        Flag("warmup_iters", &warmup_iters, "number of warmup iterations to run"),
        Flag("eval_iters", &eval_iters, "number of timed iterations to run"),
        Flag("trace", &trace, "whether or not to profile the eval iterations"),
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

    // Setup runner
    Runner runner;
    Status status = runner.loadSavedModel(graph);
    if (!status.ok()) {
        LOG(ERROR) << status << std::endl;
        return -1;
    }

    // Synthetic data
    tensorflow::Input::Initializer inputTensor(1.0f, tensorflow::TensorShape({batch_size, 28, 28, 1}));
    Tensor input = inputTensor.tensor;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        status = runner.runInference(input, input_layer, output_layer);
        if (!status.ok()) {
            LOG(ERROR) << status << std::endl;
            return -1;
        }
    }

    // Run benchmarking
    std::vector<double> elapsed_time;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

    if (trace) {
        runner.startTrace();
    }
    for (int i = 0; i < eval_iters; i++) {
        start_time = std::chrono::steady_clock::now();
        status = runner.runInference(input, input_layer, output_layer);
        end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        elapsed_time.push_back(double(duration));
        if (!status.ok()) {
            LOG(ERROR) << status << std::endl;
            return -1;
        }
    }
    if (trace) {
        runner.stopTrace();
        runner.printRunStats();
    }
    LOG(INFO) << "Throughput: " << 1000 * eval_iters * batch_size / accumulate(elapsed_time.begin(), elapsed_time.end(), 0.0) << " images/s" << std::endl;

    return 0;
}