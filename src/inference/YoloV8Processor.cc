#include "YoloV8Processor.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <algorithm>
#include <cmath>

namespace
{
const char* const kCocoLabels[YoloV8Processor::kClassCount] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

float intersectionOverUnion(const Detection& a, const Detection& b)
{
    const float left = std::max(a.x1, b.x1);
    const float top = std::max(a.y1, b.y1);
    const float right = std::min(a.x2, b.x2);
    const float bottom = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float total = areaA + areaB - intersection;
    return total > 0.0f ? intersection / total : 0.0f;
}
}

bool YoloV8Processor::preprocess(const ImageData& image,
                                 PreprocessedImage* output,
                                 std::string* error) const
{
    if (!image.encoded || image.encoded->empty())
    {
        if (error) *error = "encoded image is empty";
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    const stbi_uc* bytes = image.encoded->data();
    const int byteCount = static_cast<int>(image.encoded->size());
    if (!stbi_info_from_memory(bytes, byteCount, &width, &height, &channels) ||
        width <= 0 || height <= 0 || width > 8192 || height > 8192 ||
        static_cast<int64_t>(width) * height > 40000000)
    {
        if (error) *error = "invalid image or image dimensions exceed limits";
        return false;
    }

    stbi_uc* pixels = stbi_load_from_memory(bytes, byteCount, &width, &height, &channels, 3);
    if (!pixels)
    {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "image decode failed";
        return false;
    }

    output->originalWidth = width;
    output->originalHeight = height;
    output->scale = std::min(static_cast<float>(kInputWidth) / width,
                             static_cast<float>(kInputHeight) / height);
    const int resizedWidth = static_cast<int>(std::round(width * output->scale));
    const int resizedHeight = static_cast<int>(std::round(height * output->scale));
    const int padLeft = (kInputWidth - resizedWidth) / 2;
    const int padTop = (kInputHeight - resizedHeight) / 2;
    output->padX = static_cast<float>(padLeft);
    output->padY = static_cast<float>(padTop);
    output->tensor.assign(kInputWidth * kInputHeight * 3, 114.0f / 255.0f);

    for (int y = 0; y < resizedHeight; ++y)
    {
        const float sourceY = (y + 0.5f) / output->scale - 0.5f;
        const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(sourceY))));
        const int y1 = std::min(height - 1, y0 + 1);
        const float fy = sourceY - std::floor(sourceY);
        for (int x = 0; x < resizedWidth; ++x)
        {
            const float sourceX = (x + 0.5f) / output->scale - 0.5f;
            const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(sourceX))));
            const int x1 = std::min(width - 1, x0 + 1);
            const float fx = sourceX - std::floor(sourceX);
            for (int c = 0; c < 3; ++c)
            {
                const float top = pixels[(y0 * width + x0) * 3 + c] * (1.0f - fx) +
                                  pixels[(y0 * width + x1) * 3 + c] * fx;
                const float bottom = pixels[(y1 * width + x0) * 3 + c] * (1.0f - fx) +
                                     pixels[(y1 * width + x1) * 3 + c] * fx;
                const size_t target = ((padTop + y) * kInputWidth + padLeft + x) * 3 + c;
                output->tensor[target] = (top * (1.0f - fy) + bottom * fy) / 255.0f;
            }
        }
    }
    stbi_image_free(pixels);
    return true;
}

std::vector<Detection> YoloV8Processor::postprocess(
    const float* output, size_t outputElements, const PreprocessedImage& image,
    float confidenceThreshold, float nmsThreshold) const
{
    std::vector<Detection> candidates;
    if (!output || outputElements != static_cast<size_t>(84 * kCandidateCount)) return candidates;

    for (int i = 0; i < kCandidateCount; ++i)
    {
        int bestClass = 0;
        float bestScore = output[4 * kCandidateCount + i];
        for (int c = 1; c < kClassCount; ++c)
        {
            const float score = output[(4 + c) * kCandidateCount + i];
            if (score > bestScore) { bestScore = score; bestClass = c; }
        }
        if (bestScore < confidenceThreshold) continue;

        const float centerX = output[i];
        const float centerY = output[kCandidateCount + i];
        const float width = output[2 * kCandidateCount + i];
        const float height = output[3 * kCandidateCount + i];
        Detection detection;
        detection.classId = bestClass;
        detection.label = kCocoLabels[bestClass];
        detection.confidence = bestScore;
        detection.x1 = std::max(0.0f, (centerX - width * 0.5f - image.padX) / image.scale);
        detection.y1 = std::max(0.0f, (centerY - height * 0.5f - image.padY) / image.scale);
        detection.x2 = std::min(static_cast<float>(image.originalWidth),
                                (centerX + width * 0.5f - image.padX) / image.scale);
        detection.y2 = std::min(static_cast<float>(image.originalHeight),
                                (centerY + height * 0.5f - image.padY) / image.scale);
        if (detection.x2 > detection.x1 && detection.y2 > detection.y1)
            candidates.push_back(std::move(detection));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<Detection> selected;
    for (const Detection& candidate : candidates)
    {
        bool suppressed = false;
        for (const Detection& kept : selected)
        {
            if (candidate.classId == kept.classId && intersectionOverUnion(candidate, kept) > nmsThreshold)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) selected.push_back(candidate);
    }
    return selected;
}
