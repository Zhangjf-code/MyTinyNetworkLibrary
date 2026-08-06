#pragma once

#include "InferenceTypes.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

class InferenceMetrics
{
public:
    InferenceMetrics();

    void recordValidationFailure(const std::string& reason);
    void recordSubmissionFailure(const std::string& model, BackendType backend,
                                 const std::string& reason);
    void recordCompletion(const InferenceResult& result);
    void setQueueDepth(size_t depth);
    void setQueueCapacity(size_t capacity);

    std::string renderPrometheus() const;

private:
    struct Histogram
    {
        std::vector<unsigned long long> buckets;
        unsigned long long count = 0;
        double sum = 0.0;
    };

    static std::string escapeLabel(const std::string& value);
    static std::string labels(const std::string& model,
                              const std::string& backend,
                              const std::string& status);
    void observe(std::map<std::string, Histogram>* histograms,
                 const std::string& key, double seconds);

    mutable std::mutex mutex_;
    std::map<std::string, unsigned long long> requests_;
    std::map<std::string, unsigned long long> rejections_;
    std::map<std::string, Histogram> totalDuration_;
    std::map<std::string, Histogram> queueDuration_;
    std::map<std::string, Histogram> backendDuration_;
    size_t queueDepth_ = 0;
    size_t queueCapacity_ = 0;
    const std::vector<double> buckets_;
};
