#include "InferenceJson.h"

#include <nlohmann/json.hpp>

using nlohmann::json;

std::string serializeApiError(const std::string& requestId,
                              const std::string& code,
                              const std::string& message)
{
    return json{
        {"success", false},
        {"request_id", requestId},
        {"error", {{"code", code}, {"message", message}}}
    }.dump();
}

std::string serializeInferenceResult(const InferenceResult& result)
{
    json output = {
        {"success", result.success},
        {"request_id", result.requestId},
        {"model", result.modelName},
        {"model_version", result.modelVersion},
        {"backend", backendTypeName(result.backend)},
        {"image", {{"width", result.imageWidth}, {"height", result.imageHeight}}},
        {"latency_ms", {
            {"queue", result.latency.queueMs},
            {"preprocess", result.latency.preprocessMs},
            {"inference", result.latency.inferenceMs},
            {"postprocess", result.latency.postprocessMs},
            {"total", result.latency.totalMs}
        }},
        {"detections", json::array()}
    };

    for (const Detection& detection : result.detections)
    {
        output["detections"].push_back({
            {"class_id", detection.classId},
            {"label", detection.label},
            {"score", detection.confidence},
            {"bbox", {detection.x1, detection.y1, detection.x2, detection.y2}}
        });
    }
    if (!result.success)
    {
        output["error"] = {
            {"code", result.errorCode},
            {"message", result.errorMessage}
        };
    }
    return output.dump();
}
