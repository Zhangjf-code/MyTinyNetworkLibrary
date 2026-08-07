#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class BackendType
{
    kAuto,
    kTensorFlow,
    kTensorRT,
};

inline const char* backendTypeName(BackendType type)
{
    switch (type)
    {
    case BackendType::kTensorFlow: return "tensorflow";
    case BackendType::kTensorRT: return "tensorrt";
    default: return "auto";
    }
}

inline bool parseBackendType(const std::string& value, BackendType* result)
{
    if (value.empty() || value == "auto") *result = BackendType::kAuto;
    else if (value == "tensorflow") *result = BackendType::kTensorFlow;
    else if (value == "tensorrt") *result = BackendType::kTensorRT;
    else return false;
    return true;
}

struct ModelConfig
{
    std::string name;
    std::string version;
    std::string path;
    BackendType backend = BackendType::kAuto;
    int deviceId = 0;
    size_t maxConcurrency = 1;
};

struct ImageData
{
    std::shared_ptr<std::vector<uint8_t>> encoded;
    std::string fileName;
    std::string contentType;
};

struct InferenceRequest
{
    std::string requestId;
    std::string modelName;
    BackendType backend = BackendType::kAuto;
    float confidenceThreshold = 0.25f;
    ImageData image;
};

struct Detection
{
    int classId = -1;
    std::string label;
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct LatencyInfo
{
    double queueMs = 0.0;
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double totalMs = 0.0;
};

struct InferenceResult
{
    bool success = false;
    std::string requestId;
    std::string modelName;
    std::string modelVersion;
    BackendType backend = BackendType::kAuto;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<Detection> detections;
    LatencyInfo latency;
    std::string errorCode;
    std::string errorMessage;
};
