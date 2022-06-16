#include <vector>
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "runner.h"

Status Runner::loadSavedModel(const string& modelPath) {
    Status loadGraphStatus = ReadBinaryProto(tensorflow::Env::Default(), modelPath, &mGraphDef);
    if (!loadGraphStatus.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", modelPath, "'");
    }
    auto options = tensorflow::SessionOptions();
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.8);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    mSession.reset(tensorflow::NewSession(options));
    return mSession->Create(mGraphDef);
}

Status Runner::runInference(Tensor& input, const string& inputLayer, const string& outputLayer) {
    std::vector<Tensor> outputs;
    std::vector<std::pair<string, Tensor>> inputs = {{inputLayer, input}};
    return mSession->Run(mRunOptions, inputs, {outputLayer}, {}, &outputs, &mRunMetadata);
}

void Runner::startTrace() {
    mRunOptions.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
}

void Runner::stopTrace() {
    mRunOptions.set_trace_level(tensorflow::RunOptions::NO_TRACE);
}

void Runner::printRunStats() {
    tensorflow::StatSummarizer statSummarizer(mGraphDef);
    statSummarizer.ProcessStepStats(mRunMetadata.step_stats());
    statSummarizer.PrintStepStats();
}
