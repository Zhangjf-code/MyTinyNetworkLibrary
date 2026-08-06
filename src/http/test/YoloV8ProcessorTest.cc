#include "YoloV8Processor.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>

ImageData loadImage()
{
    std::ifstream input(TEST_IMAGE_PATH, std::ios::binary);
    assert(input);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    ImageData image;
    image.encoded.reset(new std::vector<uint8_t>(std::move(bytes)));
    image.contentType = "image/png";
    return image;
}

int main()
{
    YoloV8Processor processor;
    PreprocessedImage image;
    std::string error;
    assert(processor.preprocess(loadImage(), &image, &error));
    assert(image.originalWidth > 0 && image.originalHeight > 0);
    assert(image.tensor.size() == 640U * 640U * 3U);

    std::vector<float> output(84U * 8400U, 0.0f);
    const auto setBox = [&](int index, int classId, float score,
                            float x, float y, float width, float height) {
        output[index] = x;
        output[8400 + index] = y;
        output[2 * 8400 + index] = width;
        output[3 * 8400 + index] = height;
        output[(4 + classId) * 8400 + index] = score;
    };
    setBox(0, 0, 0.9f, 320, 320, 200, 200);
    setBox(1, 0, 0.8f, 322, 322, 200, 200);
    setBox(2, 2, 0.7f, 322, 322, 200, 200);
    const std::vector<Detection> detections = processor.postprocess(
        output.data(), output.size(), image, 0.25f, 0.45f);
    assert(detections.size() == 2);
    assert(detections[0].classId == 0);
    assert(detections[1].classId == 2);
    std::cout << "YOLOv8 processor tests passed" << std::endl;
    return 0;
}
