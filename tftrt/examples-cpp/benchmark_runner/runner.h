#ifndef TFTRT_RUNNER_H
#define TFTRT_RUNNER_H

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

/**
 * @brief Manages the running and profiling of
 * serialized DL models using the TF C++ API.
 */
class Runner {
public:
    /**
     * @brief Load the frozen graph stored at `modelPath`.
     * 
     * @param modelPath Path to the serialized model
     * @return Success status
     */
    Status loadSavedModel(const string& modelPath);

    /**
     * @brief Evaluates the graph on a single batch.
     * 
     * @param input The batch to evaluate
     * @param inputLayer Name of the input tensor
     * @param outputLayer Name of the output tensor
     * @return Success status
     */
    Status runInference(Tensor& input, const string& inputLayer, const string& outputLayer);

    /**
     * @brief Start profiling each inference run.
     */
    void startTrace();
    
    /**
     * @brief Stop profiling each inference run.
     */
    void stopTrace();

    /**
     * @brief Print aggregated stats from profiled runs.
     */
    void printRunStats();
private:
    tensorflow::GraphDef mGraphDef;
    std::unique_ptr<tensorflow::Session> mSession;

    tensorflow::RunOptions mRunOptions;
    tensorflow::RunMetadata mRunMetadata;
};

#endif // TFTRT_RUNNER_H