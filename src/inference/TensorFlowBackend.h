#pragma once

#include "IInferenceBackend.h"
#include "YoloV8Processor.h"

#include <tensorflow/c/c_api.h>

#include <atomic>
#include <mutex>

class TensorFlowBackend final : public IInferenceBackend
{
public:
    TensorFlowBackend();
    ~TensorFlowBackend() override;

    bool load(const ModelConfig& config, std::string* error) override;
    bool warmup(std::string* error) override;
    InferenceResult infer(const InferenceRequest& request) override;
    bool healthy() const override { return loaded_; }
    BackendType type() const override { return BackendType::kTensorFlow; }

private:
    bool run(const std::vector<float>& input, std::vector<float>* output,
             std::string* error);
    void close();

    TF_Graph* graph_;
    TF_Session* session_;
    TF_Status* status_;
    TF_Output input_;
    TF_Output output_;
    std::atomic_bool loaded_;
    std::mutex sessionMutex_;
    ModelConfig config_;
    YoloV8Processor processor_;
};
