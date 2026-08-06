#include "ManagementController.h"

#include <nlohmann/json.hpp>

bool ManagementController::requireGet(const HttpRequest& request,
                                      HttpResponse* response) const
{
    if (request.method() == HttpRequest::kGet) return true;
    response->setStatusCode(HttpResponse::k405MethodNotAllowed);
    response->setStatusMessage("Method Not Allowed");
    response->setContentType("application/json; charset=utf-8");
    response->addHeader("Allow", "GET");
    response->setBody(nlohmann::json({
        {"success", false},
        {"error", {{"code", "METHOD_NOT_ALLOWED"}, {"message", "GET method is required"}}}
    }).dump());
    return false;
}

void ManagementController::handleHealth(const HttpRequest& request,
                                        HttpResponse* response) const
{
    if (!requireGet(request, response)) return;
    const size_t configured = registry_ ? registry_->registrations().size() : 0;
    const size_t available = registry_ ? registry_->availableBackendCount() : 0;
    const size_t healthy = registry_ ? registry_->healthyBackendCount() : 0;
    const bool serving = healthy > 0;
    const bool degraded = serving && healthy < configured;
    response->setStatusCode(serving ? HttpResponse::k200Ok
                                    : HttpResponse::k503ServiceUnavailable);
    response->setStatusMessage(serving ? "OK" : "Service Unavailable");
    response->setContentType("application/json; charset=utf-8");
    response->setBody(nlohmann::json({
        {"status", serving ? (degraded ? "degraded" : "healthy") : "unavailable"},
        {"serving", serving},
        {"backends", {{"configured", configured}, {"available", available}, {"healthy", healthy}}}
    }).dump());
}

void ManagementController::handleModels(const HttpRequest& request,
                                        HttpResponse* response) const
{
    if (!requireGet(request, response)) return;
    nlohmann::json models = nlohmann::json::array();
    if (registry_)
    {
        for (const ModelRegistry::Registration& registration : registry_->registrations())
        {
            const bool healthy = registration.available && registration.backend &&
                                 registration.backend->healthy();
            models.push_back({
                {"name", registration.config.name},
                {"version", registration.config.version},
                {"backend", backendTypeName(registration.config.backend)},
                {"priority", registration.priority},
                {"available", registration.available},
                {"healthy", healthy}
            });
        }
    }
    response->setStatusCode(HttpResponse::k200Ok);
    response->setStatusMessage("OK");
    response->setContentType("application/json; charset=utf-8");
    response->setBody(nlohmann::json({
        {"default_model", registry_ ? registry_->defaultModel() : ""},
        {"models", models}
    }).dump());
}

void ManagementController::handleMetrics(const HttpRequest& request,
                                         HttpResponse* response) const
{
    if (!requireGet(request, response)) return;
    response->setStatusCode(HttpResponse::k200Ok);
    response->setStatusMessage("OK");
    response->setContentType("text/plain; version=0.0.4; charset=utf-8");
    response->setBody(metrics_ ? metrics_->renderPrometheus() : "");
}
