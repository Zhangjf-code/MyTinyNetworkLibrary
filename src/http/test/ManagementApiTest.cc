#include "ManagementController.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <memory>

class HealthyBackend final : public IInferenceBackend
{
public:
    explicit HealthyBackend(bool loads) : loads_(loads), healthy_(false) {}
    bool load(const ModelConfig&, std::string* error) override
    {
        healthy_ = loads_;
        if (!healthy_ && error) *error = "unavailable for test";
        return healthy_;
    }
    bool warmup(std::string*) override { return healthy_; }
    InferenceResult infer(const InferenceRequest&) override { return InferenceResult(); }
    bool healthy() const override { return healthy_; }
    BackendType type() const override { return BackendType::kTensorFlow; }

private:
    bool loads_;
    bool healthy_;
};

HttpRequest request(const char* method, size_t length)
{
    HttpRequest result;
    assert(result.setMethod(method, method + length));
    return result;
}

int main()
{
    std::shared_ptr<ModelRegistry> registry(new ModelRegistry);
    ModelConfig ready;
    ready.name = "detector";
    ready.version = "1";
    ready.path = "unused";
    ready.backend = BackendType::kTensorFlow;
    std::string error;
    assert(registry->add(ready, std::shared_ptr<IInferenceBackend>(new HealthyBackend(true)), 10, &error));
    ModelConfig unavailable = ready;
    unavailable.backend = BackendType::kTensorRT;
    assert(registry->add(unavailable, std::shared_ptr<IInferenceBackend>(new HealthyBackend(false)), 100, &error));
    assert(registry->initialize(&error));
    std::shared_ptr<InferenceMetrics> metrics(new InferenceMetrics);
    metrics->setQueueCapacity(128);
    metrics->setQueueDepth(3);
    InferenceResult completed;
    completed.success = true;
    completed.modelName = "detector";
    completed.backend = BackendType::kTensorFlow;
    completed.latency.queueMs = 2.0;
    completed.latency.inferenceMs = 12.0;
    completed.latency.totalMs = 20.0;
    metrics->recordCompletion(completed);
    metrics->recordSubmissionFailure("detector", BackendType::kTensorRT,
                                     "INFERENCE_QUEUE_FULL");
    ManagementController controller(registry, metrics);

    HttpResponse health(false);
    controller.handleHealth(request("GET", 3), &health);
    assert(health.statusCode() == HttpResponse::k200Ok);
    nlohmann::json healthJson = nlohmann::json::parse(health.body());
    assert(healthJson["status"] == "degraded");
    assert(healthJson["backends"]["healthy"] == 1);

    HttpResponse models(false);
    controller.handleModels(request("GET", 3), &models);
    nlohmann::json modelsJson = nlohmann::json::parse(models.body());
    assert(modelsJson["default_model"] == "detector");
    assert(modelsJson["models"].size() == 2);
    assert(modelsJson["models"][0]["available"] == true);
    assert(!modelsJson["models"][0].contains("path"));

    HttpResponse methodNotAllowed(false);
    controller.handleModels(request("POST", 4), &methodNotAllowed);
    assert(methodNotAllowed.statusCode() == HttpResponse::k405MethodNotAllowed);
    assert(methodNotAllowed.getHeader("Allow") == "GET");

    HttpResponse metricsResponse(false);
    controller.handleMetrics(request("GET", 3), &metricsResponse);
    assert(metricsResponse.statusCode() == HttpResponse::k200Ok);
    assert(metricsResponse.getHeader("Content-Type") ==
           "text/plain; version=0.0.4; charset=utf-8");
    assert(metricsResponse.body().find("mywebserver_inference_requests_total") != std::string::npos);
    assert(metricsResponse.body().find("backend=\"tensorflow\"") != std::string::npos);
    assert(metricsResponse.body().find("mywebserver_inference_queue_depth 3") != std::string::npos);
    assert(metricsResponse.body().find("mywebserver_inference_request_duration_seconds_bucket") != std::string::npos);

    HttpResponse metricsMethod(false);
    controller.handleMetrics(request("POST", 4), &metricsMethod);
    assert(metricsMethod.statusCode() == HttpResponse::k405MethodNotAllowed);
    return 0;
}
