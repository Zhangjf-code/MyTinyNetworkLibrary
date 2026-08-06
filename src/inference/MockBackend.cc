#include "MockBackend.h"

#include <thread>

MockBackend::MockBackend(std::chrono::milliseconds delay)
    : delay_(delay), loaded_(false)
{
}

bool MockBackend::load(const ModelConfig&, std::string*)
{
    loaded_ = true;
    return true;
}

bool MockBackend::warmup(std::string* error)
{
    if (!loaded_)
    {
        if (error) *error = "mock backend is not loaded";
        return false;
    }
    return true;
}

bool MockBackend::healthy() const
{
    return loaded_;
}

InferenceResult MockBackend::infer(const InferenceRequest& request)
{
    const auto started = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(delay_);

    InferenceResult result;
    result.requestId = request.requestId;
    result.modelName = request.modelName;
    result.modelVersion = "mock-1";
    result.backend = BackendType::kAuto;
    if (!loaded_)
    {
        result.errorCode = "BACKEND_NOT_READY";
        result.errorMessage = "mock backend is not loaded";
        return result;
    }

    result.success = true;
    result.imageWidth = 640;
    result.imageHeight = 480;
    Detection detection;
    detection.classId = 0;
    detection.label = "mock-object";
    detection.confidence = 0.90f;
    detection.x1 = 64.0f;
    detection.y1 = 48.0f;
    detection.x2 = 320.0f;
    detection.y2 = 240.0f;
    result.detections.push_back(detection);
    result.latency.inferenceMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}
