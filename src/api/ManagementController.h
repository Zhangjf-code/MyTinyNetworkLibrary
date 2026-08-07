#pragma once

#include "HttpRequest.h"
#include "HttpResponse.h"
#include "ModelRegistry.h"
#include "InferenceMetrics.h"

#include <memory>
#include <utility>

class ManagementController
{
public:
    explicit ManagementController(std::shared_ptr<ModelRegistry> registry,
                                  std::shared_ptr<InferenceMetrics> metrics = nullptr)
        : registry_(std::move(registry)), metrics_(std::move(metrics)) {}

    void handleHealth(const HttpRequest& request, HttpResponse* response) const;
    void handleModels(const HttpRequest& request, HttpResponse* response) const;
    void handleMetrics(const HttpRequest& request, HttpResponse* response) const;

private:
    bool requireGet(const HttpRequest& request, HttpResponse* response) const;
    std::shared_ptr<ModelRegistry> registry_;
    std::shared_ptr<InferenceMetrics> metrics_;
};
