#include "TensorFlowBackend.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>

int main()
{
    TensorFlowBackend backend;
    ModelConfig config;
    config.name = "yolov8n";
    config.version = "tf-2.18.0";
    config.path = TEST_MODEL_PATH;
    config.backend = BackendType::kTensorFlow;
    std::string error;
    assert(backend.load(config, &error));
    assert(backend.warmup(&error));

    std::ifstream input(TEST_IMAGE_PATH, std::ios::binary);
    assert(input);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    InferenceRequest request;
    request.requestId = "tensorflow-test";
    request.modelName = "yolov8n";
    request.backend = BackendType::kTensorFlow;
    request.confidenceThreshold = 0.25f;
    request.image.encoded.reset(new std::vector<uint8_t>(std::move(bytes)));
    request.image.contentType = "image/png";

    const InferenceResult result = backend.infer(request);
    if (!result.success) std::cerr << result.errorCode << ": " << result.errorMessage << std::endl;
    assert(result.success);
    assert(result.backend == BackendType::kTensorFlow);
    assert(result.imageWidth > 0 && result.imageHeight > 0);
    assert(!result.detections.empty());
    std::cout << "TensorFlow backend test passed with "
              << result.detections.size() << " detections" << std::endl;
    return 0;
}
