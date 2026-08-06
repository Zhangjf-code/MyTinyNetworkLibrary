#include "ModelRegistry.h"

#include "MockBackend.h"
#ifdef MYTINYMUDUO_ENABLE_TENSORFLOW
#include "TensorFlowBackend.h"
#endif
#ifdef MYTINYMUDUO_ENABLE_TENSORRT
#include "TensorRTBackend.h"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>

namespace
{
std::string directoryOf(const std::string& path)
{
    const std::string::size_type slash = path.find_last_of('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

bool absolutePath(const std::string& path)
{
    return !path.empty() && path[0] == '/';
}

std::shared_ptr<IInferenceBackend> createBackend(BackendType type)
{
#ifdef MYTINYMUDUO_ENABLE_TENSORFLOW
    if (type == BackendType::kTensorFlow)
        return std::shared_ptr<IInferenceBackend>(new TensorFlowBackend);
#endif
#ifdef MYTINYMUDUO_ENABLE_TENSORRT
    if (type == BackendType::kTensorRT)
        return std::shared_ptr<IInferenceBackend>(new TensorRTBackend);
#endif
    return std::shared_ptr<IInferenceBackend>();
}
}

bool ModelRegistry::loadFromFile(const std::string& path, std::string* error)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        if (error) *error = "cannot open model registry config: " + path;
        return false;
    }
    nlohmann::json document;
    try { input >> document; }
    catch (const std::exception& exception)
    {
        if (error) *error = std::string("invalid model registry JSON: ") + exception.what();
        return false;
    }
    if (!document.is_object() || !document.contains("default_model") ||
        !document["default_model"].is_string() || !document.contains("models") ||
        !document["models"].is_array())
    {
        if (error) *error = "registry requires string default_model and array models";
        return false;
    }

    defaultModel_ = document["default_model"].get<std::string>();
    registrations_.clear();
    const std::string baseDirectory = directoryOf(path);
    for (const nlohmann::json& item : document["models"])
    {
        try
        {
            if (item.contains("enabled") && !item["enabled"].get<bool>()) continue;
            ModelConfig config;
            config.name = item.at("name").get<std::string>();
            config.version = item.at("version").get<std::string>();
            config.path = item.at("path").get<std::string>();
            std::string backendName = item.at("backend").get<std::string>();
            if (!parseBackendType(backendName, &config.backend) || config.backend == BackendType::kAuto)
                throw std::runtime_error("backend must be tensorflow or tensorrt");
            config.deviceId = item.value("device_id", 0);
            config.maxConcurrency = item.value("max_concurrency", static_cast<size_t>(1));
            if (config.name.empty() || config.version.empty() || config.path.empty())
                throw std::runtime_error("name, version, and path must not be empty");
            if (!absolutePath(config.path)) config.path = baseDirectory + "/" + config.path;
            const int priority = item.value("priority", 0);
            std::shared_ptr<IInferenceBackend> backend = createBackend(config.backend);
            if (!backend)
            {
                if (!add(config, backend, priority, error)) return false;
                registrations_.back().loadError = "backend was disabled at build time";
            }
            else if (!add(config, backend, priority, error)) return false;
        }
        catch (const std::exception& exception)
        {
            if (error) *error = std::string("invalid model entry: ") + exception.what();
            registrations_.clear();
            return false;
        }
    }
    if (registrations_.empty())
    {
        if (error) *error = "model registry contains no enabled entries";
        return false;
    }
    if (!hasModel(defaultModel_))
    {
        if (error) *error = "default_model does not reference an enabled model";
        registrations_.clear();
        return false;
    }
    return true;
}

bool ModelRegistry::add(ModelConfig config,
                        std::shared_ptr<IInferenceBackend> backend,
                        int priority, std::string* error)
{
    for (const Registration& registration : registrations_)
    {
        if (registration.config.name == config.name &&
            registration.config.backend == config.backend)
        {
            if (error) *error = "duplicate model/backend registration";
            return false;
        }
    }
    Registration registration;
    registration.config = std::move(config);
    registration.backend = std::move(backend);
    registration.priority = priority;
    registrations_.push_back(std::move(registration));
    if (defaultModel_.empty()) defaultModel_ = registrations_.back().config.name;
    return true;
}

bool ModelRegistry::initialize(std::string* error)
{
    size_t available = 0;
    std::ostringstream failures;
    for (Registration& registration : registrations_)
    {
        if (!registration.backend) continue;
        std::string backendError;
        registration.available = registration.backend->load(registration.config, &backendError) &&
                                 registration.backend->warmup(&backendError);
        if (registration.available) ++available;
        else
        {
            registration.loadError = backendError.empty() ? "backend initialization failed" : backendError;
            failures << registration.config.name << '/' << backendTypeName(registration.config.backend)
                     << ": " << registration.loadError << "; ";
        }
    }
    if (available == 0)
    {
        if (error) *error = "no inference backend is available: " + failures.str();
        return false;
    }
    return true;
}

std::shared_ptr<IInferenceBackend> ModelRegistry::resolve(
    std::string* modelName, BackendType requested) const
{
    if (*modelName == "default") *modelName = defaultModel_;
    const Registration* selected = nullptr;
    for (const Registration& registration : registrations_)
    {
        if (!registration.available || registration.config.name != *modelName) continue;
        if (requested != BackendType::kAuto && registration.config.backend != requested) continue;
        if (!selected || registration.priority > selected->priority) selected = &registration;
    }
    return selected ? selected->backend : std::shared_ptr<IInferenceBackend>();
}

bool ModelRegistry::hasModel(const std::string& modelName) const
{
    const std::string resolved = modelName == "default" ? defaultModel_ : modelName;
    for (const Registration& registration : registrations_)
        if (registration.config.name == resolved) return true;
    return false;
}

bool ModelRegistry::hasBackend(const std::string& modelName, BackendType backend) const
{
    const std::string resolved = modelName == "default" ? defaultModel_ : modelName;
    for (const Registration& registration : registrations_)
        if (registration.config.name == resolved && registration.config.backend == backend &&
            registration.available) return true;
    return false;
}

size_t ModelRegistry::availableBackendCount() const
{
    size_t count = 0;
    for (const Registration& registration : registrations_)
        if (registration.available) ++count;
    return count;
}

size_t ModelRegistry::healthyBackendCount() const
{
    size_t count = 0;
    for (const Registration& registration : registrations_)
        if (registration.available && registration.backend && registration.backend->healthy()) ++count;
    return count;
}
