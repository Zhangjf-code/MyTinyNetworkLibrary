#pragma once

#include "IInferenceBackend.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

class ModelRegistry
{
public:
    struct Registration
    {
        ModelConfig config;
        std::shared_ptr<IInferenceBackend> backend;
        int priority = 0;
        bool available = false;
        std::string loadError;
    };

    bool loadFromFile(const std::string& path, std::string* error);
    bool add(ModelConfig config, std::shared_ptr<IInferenceBackend> backend,
             int priority, std::string* error);
    bool initialize(std::string* error);

    std::shared_ptr<IInferenceBackend> resolve(std::string* modelName,
                                               BackendType requested) const;
    bool hasModel(const std::string& modelName) const;
    bool hasBackend(const std::string& modelName, BackendType backend) const;
    size_t availableBackendCount() const;
    size_t healthyBackendCount() const;
    const std::string& defaultModel() const { return defaultModel_; }
    const std::vector<Registration>& registrations() const { return registrations_; }

private:
    std::string defaultModel_;
    std::vector<Registration> registrations_;
};
