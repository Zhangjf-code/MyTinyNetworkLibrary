#pragma once

#include "InferenceTypes.h"

#include <string>

class IInferenceBackend
{
public:
    virtual ~IInferenceBackend() = default;

    virtual bool load(const ModelConfig& config, std::string* error) = 0;
    virtual bool warmup(std::string* error) = 0;
    virtual InferenceResult infer(const InferenceRequest& request) = 0;
    virtual bool healthy() const = 0;
    virtual BackendType type() const = 0;
};
