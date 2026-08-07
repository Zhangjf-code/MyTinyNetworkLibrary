#include "TensorFlowBackend.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace
{
void tensorDeallocator(void* data, size_t, void*) { std::free(data); }

std::string statusMessage(TF_Status* status)
{
    return TF_Message(status) ? TF_Message(status) : "unknown TensorFlow error";
}
}

TensorFlowBackend::TensorFlowBackend()
    : graph_(nullptr), session_(nullptr), status_(TF_NewStatus()),
      input_{nullptr, 0}, output_{nullptr, 0}, loaded_(false)
{
}

TensorFlowBackend::~TensorFlowBackend()
{
    close();
    TF_DeleteStatus(status_);
}

void TensorFlowBackend::close()
{
    loaded_ = false;
    if (session_)
    {
        TF_CloseSession(session_, status_);
        TF_DeleteSession(session_, status_);
        session_ = nullptr;
    }
    if (graph_) { TF_DeleteGraph(graph_); graph_ = nullptr; }
}

bool TensorFlowBackend::load(const ModelConfig& config, std::string* error)
{
    std::lock_guard<std::mutex> lock(sessionMutex_);
    close();
    config_ = config;
    graph_ = TF_NewGraph();
    TF_SessionOptions* options = TF_NewSessionOptions();
    const char* tags[] = {"serve"};
    session_ = TF_LoadSessionFromSavedModel(options, nullptr, config.path.c_str(),
                                            tags, 1, graph_, nullptr, status_);
    TF_DeleteSessionOptions(options);
    if (TF_GetCode(status_) != TF_OK)
    {
        if (error) *error = statusMessage(status_);
        close();
        return false;
    }

    input_.oper = TF_GraphOperationByName(graph_, "serving_default_images");
    output_.oper = TF_GraphOperationByName(graph_, "PartitionedCall");
    if (!input_.oper || !output_.oper)
    {
        if (error) *error = "SavedModel does not contain expected serving_default input/output operations";
        close();
        return false;
    }
    loaded_ = true;
    return true;
}

bool TensorFlowBackend::run(const std::vector<float>& input,
                            std::vector<float>* output, std::string* error)
{
    const int64_t dimensions[] = {1, 640, 640, 3};
    const size_t bytes = input.size() * sizeof(float);
    void* tensorData = std::malloc(bytes);
    if (!tensorData) { if (error) *error = "failed to allocate input tensor"; return false; }
    std::memcpy(tensorData, input.data(), bytes);
    TF_Tensor* inputTensor = TF_NewTensor(TF_FLOAT, dimensions, 4, tensorData,
                                          bytes, tensorDeallocator, nullptr);
    if (!inputTensor) { std::free(tensorData); if (error) *error = "failed to create input tensor"; return false; }

    TF_Tensor* outputTensor = nullptr;
    {
        std::lock_guard<std::mutex> lock(sessionMutex_);
        TF_SessionRun(session_, nullptr, &input_, &inputTensor, 1,
                      &output_, &outputTensor, 1, nullptr, 0, nullptr, status_);
    }
    TF_DeleteTensor(inputTensor);
    if (TF_GetCode(status_) != TF_OK || !outputTensor)
    {
        if (error) *error = statusMessage(status_);
        if (outputTensor) TF_DeleteTensor(outputTensor);
        return false;
    }
    const size_t count = TF_TensorByteSize(outputTensor) / sizeof(float);
    const float* values = static_cast<const float*>(TF_TensorData(outputTensor));
    output->assign(values, values + count);
    TF_DeleteTensor(outputTensor);
    return true;
}

bool TensorFlowBackend::warmup(std::string* error)
{
    if (!loaded_) { if (error) *error = "TensorFlow backend is not loaded"; return false; }
    std::vector<float> input(640 * 640 * 3, 114.0f / 255.0f);
    std::vector<float> output;
    return run(input, &output, error) && output.size() == 84U * 8400U;
}

InferenceResult TensorFlowBackend::infer(const InferenceRequest& request)
{
    InferenceResult result;
    result.requestId = request.requestId;
    result.modelName = request.modelName;
    result.modelVersion = config_.version;
    result.backend = BackendType::kTensorFlow;
    if (!loaded_)
    {
        result.errorCode = "BACKEND_NOT_READY";
        result.errorMessage = "TensorFlow backend is not loaded";
        return result;
    }

    PreprocessedImage image;
    std::string error;
    auto started = std::chrono::steady_clock::now();
    if (!processor_.preprocess(request.image, &image, &error))
    {
        result.errorCode = "IMAGE_DECODE_FAILED";
        result.errorMessage = error;
        return result;
    }
    auto preprocessed = std::chrono::steady_clock::now();
    std::vector<float> rawOutput;
    if (!run(image.tensor, &rawOutput, &error))
    {
        result.errorCode = "TENSORFLOW_INFERENCE_FAILED";
        result.errorMessage = error;
        return result;
    }
    auto inferred = std::chrono::steady_clock::now();
    result.detections = processor_.postprocess(rawOutput.data(), rawOutput.size(), image,
                                               request.confidenceThreshold);
    auto finished = std::chrono::steady_clock::now();
    result.success = true;
    result.imageWidth = image.originalWidth;
    result.imageHeight = image.originalHeight;
    result.latency.preprocessMs = std::chrono::duration<double, std::milli>(preprocessed - started).count();
    result.latency.inferenceMs = std::chrono::duration<double, std::milli>(inferred - preprocessed).count();
    result.latency.postprocessMs = std::chrono::duration<double, std::milli>(finished - inferred).count();
    return result;
}
