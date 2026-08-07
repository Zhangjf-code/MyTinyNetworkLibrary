#include "TensorRTBackend.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace
{
const char* kInputName = "images";
const char* kOutputName = "output0";
const size_t kInputElements = 1U * 3U * 640U * 640U;
const size_t kOutputElements = 1U * 84U * 8400U;

bool cudaOk(cudaError_t status, const char* operation, std::string* error)
{
    if (status == cudaSuccess) return true;
    if (error)
    {
        std::ostringstream message;
        message << operation << ": " << cudaGetErrorString(status);
        *error = message.str();
    }
    return false;
}

bool dimensionsEqual(const nvinfer1::Dims& dimensions,
                     const std::vector<int>& expected)
{
    if (dimensions.nbDims != static_cast<int>(expected.size())) return false;
    for (int i = 0; i < dimensions.nbDims; ++i)
        if (dimensions.d[i] != expected[static_cast<size_t>(i)]) return false;
    return true;
}
}

TensorRTBackend::TensorRTBackend()
    : runtime_(nullptr), engine_(nullptr), context_(nullptr), stream_(nullptr),
      deviceInput_(nullptr), deviceOutput_(nullptr), inputBytes_(0),
      outputBytes_(0), loaded_(false)
{
}

TensorRTBackend::~TensorRTBackend()
{
    close();
}

void TensorRTBackend::Logger::log(Severity severity, const char* message) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cerr << "[TensorRT] " << (message ? message : "") << std::endl;
}

void TensorRTBackend::close()
{
    loaded_ = false;
    if (deviceOutput_) { cudaFree(deviceOutput_); deviceOutput_ = nullptr; }
    if (deviceInput_) { cudaFree(deviceInput_); deviceInput_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    delete context_; context_ = nullptr;
    delete engine_; engine_ = nullptr;
    delete runtime_; runtime_ = nullptr;
}

bool TensorRTBackend::load(const ModelConfig& config, std::string* error)
{
    std::lock_guard<std::mutex> lock(executionMutex_);
    close();
    config_ = config;
    if (!cudaOk(cudaSetDevice(config.deviceId), "cudaSetDevice", error)) return false;

    std::ifstream file(config.path.c_str(), std::ios::binary | std::ios::ate);
    if (!file)
    {
        if (error) *error = "cannot open TensorRT engine: " + config.path;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0)
    {
        if (error) *error = "TensorRT engine is empty";
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> serialized(static_cast<size_t>(size));
    if (!file.read(serialized.data(), size))
    {
        if (error) *error = "failed to read TensorRT engine";
        return false;
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_) engine_ = runtime_->deserializeCudaEngine(serialized.data(), serialized.size());
    if (engine_) context_ = engine_->createExecutionContext();
    if (!runtime_ || !engine_ || !context_)
    {
        if (error) *error = "failed to deserialize TensorRT engine or create execution context";
        close();
        return false;
    }

    if (engine_->getTensorIOMode(kInputName) != nvinfer1::TensorIOMode::kINPUT ||
        engine_->getTensorIOMode(kOutputName) != nvinfer1::TensorIOMode::kOUTPUT ||
        engine_->getTensorDataType(kInputName) != nvinfer1::DataType::kFLOAT ||
        engine_->getTensorDataType(kOutputName) != nvinfer1::DataType::kFLOAT ||
        !dimensionsEqual(engine_->getTensorShape(kInputName), {1, 3, 640, 640}) ||
        !dimensionsEqual(engine_->getTensorShape(kOutputName), {1, 84, 8400}))
    {
        if (error) *error = "TensorRT engine does not match images/output0 FP32 YOLOv8 tensor contract";
        close();
        return false;
    }

    inputBytes_ = kInputElements * sizeof(float);
    outputBytes_ = kOutputElements * sizeof(float);
    if (!cudaOk(cudaStreamCreate(&stream_), "cudaStreamCreate", error) ||
        !cudaOk(cudaMalloc(&deviceInput_, inputBytes_), "cudaMalloc(input)", error) ||
        !cudaOk(cudaMalloc(&deviceOutput_, outputBytes_), "cudaMalloc(output)", error))
    {
        close();
        return false;
    }
    if (!context_->setTensorAddress(kInputName, deviceInput_) ||
        !context_->setTensorAddress(kOutputName, deviceOutput_))
    {
        if (error) *error = "failed to bind TensorRT input/output buffers";
        close();
        return false;
    }
    loaded_ = true;
    return true;
}

bool TensorRTBackend::run(const std::vector<float>& nhwcInput,
                          std::vector<float>* output, std::string* error)
{
    if (nhwcInput.size() != kInputElements)
    {
        if (error) *error = "unexpected TensorRT input element count";
        return false;
    }
    std::vector<float> nchw(kInputElements);
    for (size_t pixel = 0; pixel < 640U * 640U; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            nchw[channel * 640U * 640U + pixel] = nhwcInput[pixel * 3U + channel];
    output->resize(kOutputElements);

    std::lock_guard<std::mutex> lock(executionMutex_);
    if (!loaded_)
    {
        if (error) *error = "TensorRT backend is not loaded";
        return false;
    }
    if (!cudaOk(cudaMemcpyAsync(deviceInput_, nchw.data(), inputBytes_,
                                cudaMemcpyHostToDevice, stream_), "copy input to GPU", error) ||
        !context_->enqueueV3(stream_) ||
        !cudaOk(cudaMemcpyAsync(output->data(), deviceOutput_, outputBytes_,
                                cudaMemcpyDeviceToHost, stream_), "copy output to host", error) ||
        !cudaOk(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", error))
    {
        if (error && error->empty()) *error = "TensorRT enqueueV3 failed";
        return false;
    }
    return true;
}

bool TensorRTBackend::warmup(std::string* error)
{
    if (!loaded_) { if (error) *error = "TensorRT backend is not loaded"; return false; }
    std::vector<float> input(kInputElements, 114.0f / 255.0f);
    std::vector<float> output;
    return run(input, &output, error) && output.size() == kOutputElements;
}

InferenceResult TensorRTBackend::infer(const InferenceRequest& request)
{
    InferenceResult result;
    result.requestId = request.requestId;
    result.modelName = request.modelName;
    result.modelVersion = config_.version;
    result.backend = BackendType::kTensorRT;
    if (!loaded_)
    {
        result.errorCode = "BACKEND_NOT_READY";
        result.errorMessage = "TensorRT backend is not loaded";
        return result;
    }

    PreprocessedImage image;
    std::string error;
    const auto started = std::chrono::steady_clock::now();
    if (!processor_.preprocess(request.image, &image, &error))
    {
        result.errorCode = "IMAGE_DECODE_FAILED";
        result.errorMessage = error;
        return result;
    }
    const auto preprocessed = std::chrono::steady_clock::now();
    std::vector<float> rawOutput;
    if (!run(image.tensor, &rawOutput, &error))
    {
        result.errorCode = "TENSORRT_INFERENCE_FAILED";
        result.errorMessage = error;
        return result;
    }
    const auto inferred = std::chrono::steady_clock::now();
    result.detections = processor_.postprocess(rawOutput.data(), rawOutput.size(), image,
                                               request.confidenceThreshold);
    const auto finished = std::chrono::steady_clock::now();
    result.success = true;
    result.imageWidth = image.originalWidth;
    result.imageHeight = image.originalHeight;
    result.latency.preprocessMs = std::chrono::duration<double, std::milli>(preprocessed - started).count();
    result.latency.inferenceMs = std::chrono::duration<double, std::milli>(inferred - preprocessed).count();
    result.latency.postprocessMs = std::chrono::duration<double, std::milli>(finished - inferred).count();
    return result;
}
