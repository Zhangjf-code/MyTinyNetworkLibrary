#include "InferenceController.h"
#include "InferenceJson.h"
#include "Buffer.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <iostream>

namespace
{
HttpRequest makeValidRequest()
{
    HttpRequest request;
    const char method[] = "POST";
    assert(request.setMethod(method, method + 4));
    request.addBodyForm("model", "yolov8");
    request.addBodyForm("backend", "tensorrt");
    request.addBodyForm("threshold", "0.4");
    HttpRequest::UploadedFile file;
    file.fieldName = "image";
    file.fileName = "test.jpg";
    file.contentType = "image/jpeg";
    file.content.assign("jpeg-data", 9);
    request.addFile(std::move(file));
    return request;
}

void testDetectionSerialization()
{
    InferenceResult result;
    result.success = true;
    result.requestId = "request-1";
    result.modelName = "yolov8";
    result.modelVersion = "1";
    result.backend = BackendType::kTensorRT;
    result.imageWidth = 640;
    result.imageHeight = 480;
    Detection detection;
    detection.classId = 0;
    detection.label = "person";
    detection.confidence = 0.95f;
    detection.x1 = 1.0f;
    detection.y1 = 2.0f;
    detection.x2 = 3.0f;
    detection.y2 = 4.0f;
    result.detections.push_back(detection);

    const nlohmann::json output = nlohmann::json::parse(serializeInferenceResult(result));
    assert(output["success"] == true);
    assert(output["backend"] == "tensorrt");
    assert(output["detections"][0]["label"] == "person");
    assert(output["detections"][0]["bbox"].size() == 4);
}

void testValidProtocolBeforeBackendExists()
{
    InferenceController controller;
    HttpRequest request = makeValidRequest();
    HttpResponse response(false);
    controller.handle(request, &response);

    assert(response.statusCode() == HttpResponse::k501NotImplemented);
    assert(response.getHeader("Content-Type") == "application/json; charset=utf-8");
    const nlohmann::json output = nlohmann::json::parse(response.body());
    assert(output["error"]["code"] == "BACKEND_NOT_IMPLEMENTED");
    assert(!output["request_id"].get<std::string>().empty());
}

void testValidationErrors()
{
    InferenceController controller;
    HttpRequest request = makeValidRequest();
    request.addBodyForm("threshold", "1.5");
    HttpResponse response(false);
    controller.handle(request, &response);
    assert(response.statusCode() == HttpResponse::k400BadRequest);
    const nlohmann::json output = nlohmann::json::parse(response.body());
    assert(output["error"]["code"] == "INVALID_THRESHOLD");
    assert(!output["request_id"].get<std::string>().empty());
}

void testClosedResponseHasContentLength()
{
    HttpResponse response(true);
    response.setStatusCode(HttpResponse::k200Ok);
    response.setStatusMessage("OK");
    response.setBody("12345");
    Buffer output(64);
    response.appendToBuffer(&output);
    const std::string wire = output.retrieveAllAsString();
    assert(wire.find("Connection: close\r\n") != std::string::npos);
    assert(wire.find("Content-Length: 5\r\n") != std::string::npos);
}
}

int main()
{
    testDetectionSerialization();
    testValidProtocolBeforeBackendExists();
    testValidationErrors();
    testClosedResponseHasContentLength();
    std::cout << "Inference API tests passed" << std::endl;
    return 0;
}
