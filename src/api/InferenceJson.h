#pragma once

#include "InferenceTypes.h"

#include <string>

std::string serializeInferenceResult(const InferenceResult& result);
std::string serializeApiError(const std::string& requestId,
                              const std::string& code,
                              const std::string& message);
