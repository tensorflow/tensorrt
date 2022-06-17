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
#include "datasets/synthetic.h"

using tensorflow::Flag;

template <typename F, typename ... Args>
double timed(F&& f, Args&&...args) {
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::forward<F>(f)(std::forward<Args>(args)...);
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    return static_cast<double>(duration);
}

void dequeue(datasets::BaseDataset& dataset, int offset, int batch_size, Tensor* input) {
    *input = dataset.getTensor(offset, batch_size);
}

void infer(Runner& runner, Tensor& input, const string& input_layer, const string& output_layer, Status* result) {
    *result = runner.runInference(input, input_layer, output_layer);
}

int main(int argc, char* argv[]) {
    // Parse arguments
    string graph =
        "/opt/tensorflow/tensorflow-source/tensorflow/examples/benchmark_runner/toy_model/frozen/frozen_model.pb";
    int32_t batch_size = 1024;
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
        LOG(ERROR) << status;
        return -1;
    }

    // Setup dataset
    datasets::SyntheticDataset dataset = {batch_size, 28, 28, 1};

    // Run benchmarking
    std::vector<double> dequeue_time;
    std::vector<double> memcpyHtoD_time;
    std::vector<double> infer_time;
    std::vector<double> wall_time;

    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    for (int i = 0; i < warmup_iters + eval_iters; i++) {
        if (i >= warmup_iters && trace) {
            runner.startTrace();
        }
        start_time = std::chrono::steady_clock::now();

        Tensor input;
        dequeue_time.push_back(timed(dequeue, dataset, i * batch_size, batch_size, &input));

        Status inferStatus;
        infer_time.push_back(timed(infer, runner, input, input_layer, output_layer, &inferStatus));

        end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        wall_time.push_back((double)duration);
        
        if (!inferStatus.ok()) {
            LOG(ERROR) << inferStatus;
            return -1;
        }
    }
    if (trace) {
        runner.stopTrace();
        runner.printRunStats();
    }
    LOG(INFO) << "Mean wall clock time (ms): " << accumulate(wall_time.begin() + warmup_iters, wall_time.end(), 0.0) / eval_iters;
    LOG(INFO) << "Mean dequeue time (ms): " << accumulate(dequeue_time.begin() + warmup_iters, dequeue_time.end(), 0.0) / eval_iters;
    LOG(INFO) << "Mean GPU time (ms): " << accumulate(infer_time.begin() + warmup_iters, infer_time.end(), 0.0) / eval_iters;
    LOG(INFO) << "Throughput (ims/s): " << 1000 * eval_iters * batch_size / accumulate(infer_time.begin() + warmup_iters, infer_time.end(), 0.0);

    return 0;
}