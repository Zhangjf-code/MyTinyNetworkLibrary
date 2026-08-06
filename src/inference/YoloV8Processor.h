#pragma once

#include "InferenceTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct PreprocessedImage
{
    std::vector<float> tensor;
    int originalWidth = 0;
    int originalHeight = 0;
    float scale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};

class YoloV8Processor
{
public:
    static const int kInputWidth = 640;
    static const int kInputHeight = 640;
    static const int kClassCount = 80;
    static const int kCandidateCount = 8400;

    bool preprocess(const ImageData& image, PreprocessedImage* output,
                    std::string* error) const;

    std::vector<Detection> postprocess(const float* output,
                                       size_t outputElements,
                                       const PreprocessedImage& image,
                                       float confidenceThreshold,
                                       float nmsThreshold = 0.45f) const;
};
