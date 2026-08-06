#include "InferenceController.h"

#include "InferenceJson.h"
#include "Buffer.h"
#include "EventLoop.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <sstream>

std::string InferenceController::nextRequestId()
{
    static std::atomic<unsigned long long> sequence(0);
    const unsigned long long now = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::ostringstream output;
    output << std::hex << now << '-' << sequence.fetch_add(1);
    return output.str();
}

void InferenceController::setJsonError(HttpResponse* response,
                                       HttpResponse::HttpStatusCode status,
                                       const std::string& statusMessage,
                                       const std::string& requestId,
                                       const std::string& code,
                                       const std::string& message)
{
    response->setStatusCode(status);
    response->setStatusMessage(statusMessage);
    response->setContentType("application/json; charset=utf-8");
    response->setBody(serializeApiError(requestId, code, message));
}

bool InferenceController::buildRequest(const HttpRequest& httpRequest,
                                       InferenceRequest* request,
                                       std::string* errorCode,
                                       std::string* errorMessage) const
{
    if (httpRequest.method() != HttpRequest::kPost)
    {
        *errorCode = "METHOD_NOT_ALLOWED";
        *errorMessage = "POST method is required";
        return false;
    }
    if (httpRequest.files().size() != 1)
    {
        *errorCode = "INVALID_IMAGE_COUNT";
        *errorMessage = "exactly one image file is required";
        return false;
    }

    const HttpRequest::UploadedFile& file = httpRequest.files().front();
    if (file.fieldName != "image" || file.content.empty())
    {
        *errorCode = "INVALID_IMAGE";
        *errorMessage = "a non-empty multipart file field named image is required";
        return false;
    }
    if (file.contentType != "image/jpeg" && file.contentType != "image/png")
    {
        *errorCode = "UNSUPPORTED_IMAGE_TYPE";
        *errorMessage = "only image/jpeg and image/png are supported";
        return false;
    }

    request->modelName = httpRequest.getBodyForm("model");
    if (request->modelName.empty()) request->modelName = "default";

    if (!parseBackendType(httpRequest.getBodyForm("backend"), &request->backend))
    {
        *errorCode = "INVALID_BACKEND";
        *errorMessage = "backend must be auto, tensorflow, or tensorrt";
        return false;
    }

    const std::string threshold = httpRequest.getBodyForm("threshold");
    if (!threshold.empty())
    {
        errno = 0;
        char* end = nullptr;
        const float parsed = std::strtof(threshold.c_str(), &end);
        if (errno != 0 || end == threshold.c_str() || *end != '\0' || parsed < 0.0f || parsed > 1.0f)
        {
            *errorCode = "INVALID_THRESHOLD";
            *errorMessage = "threshold must be a number between 0 and 1";
            return false;
        }
        request->confidenceThreshold = parsed;
    }

    request->image.fileName = file.fileName;
    request->image.contentType = file.contentType;
    request->image.encoded = std::make_shared<std::vector<uint8_t>>(
        file.content.begin(), file.content.end());
    return true;
}

void InferenceController::handle(const HttpRequest& httpRequest, HttpResponse* response) const
{
    InferenceRequest request;
    request.requestId = nextRequestId();
    std::string errorCode;
    std::string errorMessage;
    if (!buildRequest(httpRequest, &request, &errorCode, &errorMessage))
    {
        if (metrics_) metrics_->recordValidationFailure(errorCode);
        HttpResponse::HttpStatusCode status = HttpResponse::k400BadRequest;
        std::string statusMessage = "Bad Request";
        if (errorCode == "METHOD_NOT_ALLOWED")
        {
            status = HttpResponse::k405MethodNotAllowed;
            statusMessage = "Method Not Allowed";
            response->addHeader("Allow", "POST");
        }
        else if (errorCode == "UNSUPPORTED_IMAGE_TYPE")
        {
            status = HttpResponse::k415UnsupportedMediaType;
            statusMessage = "Unsupported Media Type";
        }
        setJsonError(response, status, statusMessage,
                     request.requestId, errorCode, errorMessage);
        return;
    }

    setJsonError(response, HttpResponse::k501NotImplemented, "Not Implemented",
                 request.requestId, "BACKEND_NOT_IMPLEMENTED",
                 "the inference API is valid, but no inference backend is connected yet");
}

void InferenceController::sendResponseAndClose(const TcpConnectionPtr& connection,
                                               const HttpResponse& response)
{
    Buffer output;
    response.appendToBuffer(&output);
    connection->send(output.retrieveAllAsString());
    connection->shutdown();
}

bool InferenceController::handleAsync(const TcpConnectionPtr& connection,
                                      const HttpRequest& httpRequest) const
{
    if (httpRequest.path() != "/v1/infer") return false;

    InferenceRequest request;
    request.requestId = nextRequestId();
    std::string errorCode;
    std::string errorMessage;
    if (!buildRequest(httpRequest, &request, &errorCode, &errorMessage))
    {
        HttpResponse response(true);
        HttpResponse::HttpStatusCode status = HttpResponse::k400BadRequest;
        std::string statusMessage = "Bad Request";
        if (errorCode == "METHOD_NOT_ALLOWED")
        {
            status = HttpResponse::k405MethodNotAllowed;
            statusMessage = "Method Not Allowed";
            response.addHeader("Allow", "POST");
        }
        else if (errorCode == "UNSUPPORTED_IMAGE_TYPE")
        {
            status = HttpResponse::k415UnsupportedMediaType;
            statusMessage = "Unsupported Media Type";
        }
        setJsonError(&response, status, statusMessage, request.requestId,
                     errorCode, errorMessage);
        sendResponseAndClose(connection, response);
        return true;
    }

    if (!scheduler_)
    {
        if (metrics_) metrics_->recordSubmissionFailure(request.modelName, request.backend,
                                                        "scheduler_unavailable");
        HttpResponse response(true);
        setJsonError(&response, HttpResponse::k503ServiceUnavailable,
                     "Service Unavailable", request.requestId,
                     "SCHEDULER_UNAVAILABLE", "inference scheduler is not running");
        sendResponseAndClose(connection, response);
        return true;
    }

    const std::string requestId = request.requestId;
    const std::string requestedModel = request.modelName;
    const BackendType requestedBackend = request.backend;
    WeakTcpConnectionPtr weakConnection(connection);
    const InferenceScheduler::SubmitResult submitted = scheduler_->submit(
        std::move(request),
        [weakConnection](InferenceResult result) mutable {
            TcpConnectionPtr connection = weakConnection.lock();
            if (!connection) return;
            EventLoop* loop = connection->getLoop();
            std::shared_ptr<InferenceResult> resultPtr(
                new InferenceResult(std::move(result)));
            loop->queueInLoop([connection, resultPtr]() {
                if (!connection->connected()) return;
                HttpResponse response(true);
                response.setContentType("application/json; charset=utf-8");
                if (resultPtr->success)
                {
                    response.setStatusCode(HttpResponse::k200Ok);
                    response.setStatusMessage("OK");
                }
                else
                {
                    const bool invalidImage = resultPtr->errorCode == "IMAGE_DECODE_FAILED";
                    response.setStatusCode(invalidImage ? HttpResponse::k422UnprocessableEntity
                                                        : HttpResponse::k500InternalServerError);
                    response.setStatusMessage(invalidImage ? "Unprocessable Entity"
                                                           : "Internal Server Error");
                }
                response.setBody(serializeInferenceResult(*resultPtr));
                InferenceController::sendResponseAndClose(connection, response);
            });
        });

    if (submitted != InferenceScheduler::SubmitResult::kAccepted)
    {
        HttpResponse response(true);
        const bool backendUnavailable =
            submitted == InferenceScheduler::SubmitResult::kBackendUnavailable;
        const bool modelUnavailable =
            submitted == InferenceScheduler::SubmitResult::kModelUnavailable;
        const bool unprocessable = backendUnavailable || modelUnavailable;
        const std::string code = backendUnavailable ? "BACKEND_UNAVAILABLE" :
            (modelUnavailable ? "MODEL_UNAVAILABLE" :
            (submitted == InferenceScheduler::SubmitResult::kQueueFull
                ? "INFERENCE_QUEUE_FULL" : "SCHEDULER_UNAVAILABLE"));
        const std::string message = backendUnavailable ? "requested inference backend is not available" :
            (modelUnavailable ? "requested model is not available" :
            (submitted == InferenceScheduler::SubmitResult::kQueueFull
                ? "inference queue is full" : "inference scheduler is not running"));
        if (metrics_) metrics_->recordSubmissionFailure(requestedModel, requestedBackend, code);
        setJsonError(&response,
                     unprocessable ? HttpResponse::k422UnprocessableEntity
                                   : HttpResponse::k503ServiceUnavailable,
                     unprocessable ? "Unprocessable Entity" : "Service Unavailable",
                     requestId, code, message);
        if (!unprocessable) response.addHeader("Retry-After", "1");
        sendResponseAndClose(connection, response);
        return true;
    }

    connection->stopRead();
    return true;
}
