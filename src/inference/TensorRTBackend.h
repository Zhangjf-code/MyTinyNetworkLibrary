#pragma once

#include "IInferenceBackend.h"
#include "YoloV8Processor.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>

class TensorRTBackend final : public IInferenceBackend
{
public:
    TensorRTBackend();
    ~TensorRTBackend() override;

    bool load(const ModelConfig& config, std::string* error) override;
    bool warmup(std::string* error) override;
    InferenceResult infer(const InferenceRequest& request) override;
    bool healthy() const override { return loaded_; }
    BackendType type() const override { return BackendType::kTensorRT; }

private:
    class Logger final : public nvinfer1::ILogger
    {
        void log(Severity severity, const char* message) noexcept override;
    };

    bool run(const std::vector<float>& nhwcInput,
             std::vector<float>* output, std::string* error);
    void close();

    Logger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    void* deviceInput_;
    void* deviceOutput_;
    size_t inputBytes_;
    size_t outputBytes_;
    std::atomic_bool loaded_;
    std::mutex executionMutex_;
    ModelConfig config_;
    YoloV8Processor processor_;
};
