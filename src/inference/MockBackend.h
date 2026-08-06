#pragma once

#include "IInferenceBackend.h"

#include <atomic>
#include <chrono>

class MockBackend final : public IInferenceBackend
{
public:
    explicit MockBackend(std::chrono::milliseconds delay = std::chrono::milliseconds(20));

    bool load(const ModelConfig& config, std::string* error) override;
    bool warmup(std::string* error) override;
    InferenceResult infer(const InferenceRequest& request) override;
    bool healthy() const override;
    BackendType type() const override { return BackendType::kAuto; }

private:
    std::chrono::milliseconds delay_;
    std::atomic_bool loaded_;
};
