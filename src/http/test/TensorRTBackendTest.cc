#include "TensorRTBackend.h"

#include <cassert>
#include <fstream>
#include <iterator>
#include <memory>

int main()
{
    std::ifstream imageFile(TEST_IMAGE_PATH, std::ios::binary);
    assert(imageFile);
    std::shared_ptr<std::vector<uint8_t>> bytes(new std::vector<uint8_t>(
        std::istreambuf_iterator<char>(imageFile), std::istreambuf_iterator<char>()));

    TensorRTBackend backend;
    ModelConfig config;
    config.name = "yolov8n";
    config.version = "trt-10.0.1-fp16";
    config.path = TEST_ENGINE_PATH;
    config.backend = BackendType::kTensorRT;
    config.deviceId = 0;
    std::string error;
    assert(backend.load(config, &error));
    assert(backend.warmup(&error));

    InferenceRequest request;
    request.requestId = "tensorrt-test";
    request.modelName = "yolov8n";
    request.backend = BackendType::kTensorRT;
    request.confidenceThreshold = 0.25f;
    request.image.encoded = bytes;
    request.image.fileName = "bus.jpg";
    request.image.contentType = "image/jpeg";
    const InferenceResult result = backend.infer(request);
    assert(result.success);
    assert(result.backend == BackendType::kTensorRT);
    assert(result.imageWidth == 810 && result.imageHeight == 1080);
    assert(!result.detections.empty());
    bool foundBus = false;
    for (const Detection& detection : result.detections)
        if (detection.label == "bus") foundBus = true;
    assert(foundBus);
    return 0;
}
