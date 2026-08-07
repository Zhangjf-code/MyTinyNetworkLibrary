#pragma once

#include "HttpRequest.h"
#include "HttpResponse.h"
#include "InferenceTypes.h"
#include "InferenceScheduler.h"
#include "InferenceMetrics.h"
#include "TcpConnection.h"

#include <atomic>
#include <string>

class InferenceController
{
public:
    explicit InferenceController(InferenceScheduler* scheduler = nullptr,
                                 std::shared_ptr<InferenceMetrics> metrics = nullptr)
        : scheduler_(scheduler), metrics_(std::move(metrics)) {}

    void handle(const HttpRequest& request, HttpResponse* response) const;
    bool handleAsync(const TcpConnectionPtr& connection, const HttpRequest& request) const;

private:
    bool buildRequest(const HttpRequest& httpRequest,
                      InferenceRequest* request,
                      std::string* errorCode,
                      std::string* errorMessage) const;
    static std::string nextRequestId();
    static void setJsonError(HttpResponse* response,
                             HttpResponse::HttpStatusCode status,
                             const std::string& statusMessage,
                             const std::string& requestId,
                             const std::string& code,
                             const std::string& message);
    static void sendResponseAndClose(const TcpConnectionPtr& connection,
                                     const HttpResponse& response);

    InferenceScheduler* scheduler_;
    std::shared_ptr<InferenceMetrics> metrics_;
};
