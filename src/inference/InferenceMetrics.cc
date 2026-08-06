#include "InferenceMetrics.h"

#include <iomanip>
#include <sstream>

InferenceMetrics::InferenceMetrics()
    : buckets_({0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0})
{
}

std::string InferenceMetrics::escapeLabel(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value)
    {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        if (ch == '\n') escaped += "\\n";
        else escaped.push_back(ch);
    }
    return escaped;
}

std::string InferenceMetrics::labels(const std::string& model,
                                     const std::string& backend,
                                     const std::string& status)
{
    return "model=\"" + escapeLabel(model) + "\",backend=\"" +
           escapeLabel(backend) + "\",status=\"" + escapeLabel(status) + "\"";
}

void InferenceMetrics::recordValidationFailure(const std::string& reason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ++requests_[labels("unknown", "unknown", "invalid_request")];
    ++rejections_[reason];
}

void InferenceMetrics::recordSubmissionFailure(const std::string& model,
                                                BackendType backend,
                                                const std::string& reason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ++requests_[labels(model, backendTypeName(backend), "rejected")];
    ++rejections_[reason];
}

void InferenceMetrics::observe(std::map<std::string, Histogram>* histograms,
                               const std::string& key, double seconds)
{
    Histogram& histogram = (*histograms)[key];
    if (histogram.buckets.empty()) histogram.buckets.resize(buckets_.size(), 0);
    ++histogram.count;
    histogram.sum += seconds;
    for (size_t i = 0; i < buckets_.size(); ++i)
        if (seconds <= buckets_[i]) ++histogram.buckets[i];
}

void InferenceMetrics::recordCompletion(const InferenceResult& result)
{
    const std::string key = labels(result.modelName, backendTypeName(result.backend),
                                   result.success ? "success" : "error");
    const std::string durationLabels = labels(result.modelName,
                                              backendTypeName(result.backend), "all");
    std::lock_guard<std::mutex> lock(mutex_);
    ++requests_[key];
    observe(&totalDuration_, durationLabels, result.latency.totalMs / 1000.0);
    observe(&queueDuration_, durationLabels, result.latency.queueMs / 1000.0);
    observe(&backendDuration_, durationLabels, result.latency.inferenceMs / 1000.0);
}

void InferenceMetrics::setQueueDepth(size_t depth)
{
    std::lock_guard<std::mutex> lock(mutex_);
    queueDepth_ = depth;
}

void InferenceMetrics::setQueueCapacity(size_t capacity)
{
    std::lock_guard<std::mutex> lock(mutex_);
    queueCapacity_ = capacity;
}

std::string InferenceMetrics::renderPrometheus() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream output;
    output << std::setprecision(10);
    output << "# HELP mywebserver_inference_requests_total Inference requests by final outcome.\n"
           << "# TYPE mywebserver_inference_requests_total counter\n";
    for (const auto& item : requests_)
        output << "mywebserver_inference_requests_total{" << item.first << "} " << item.second << '\n';

    output << "# HELP mywebserver_inference_rejections_total Requests rejected before inference.\n"
           << "# TYPE mywebserver_inference_rejections_total counter\n";
    for (const auto& item : rejections_)
        output << "mywebserver_inference_rejections_total{reason=\""
               << escapeLabel(item.first) << "\"} " << item.second << '\n';

    output << "# HELP mywebserver_inference_queue_depth Current queued inference tasks.\n"
           << "# TYPE mywebserver_inference_queue_depth gauge\n"
           << "mywebserver_inference_queue_depth " << queueDepth_ << '\n'
           << "# HELP mywebserver_inference_queue_capacity Maximum queued inference tasks.\n"
           << "# TYPE mywebserver_inference_queue_capacity gauge\n"
           << "mywebserver_inference_queue_capacity " << queueCapacity_ << '\n';

    const auto renderHistogram = [this, &output](const char* name, const char* help,
                                                 const std::map<std::string, Histogram>& values) {
        output << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " histogram\n";
        for (const auto& item : values)
        {
            for (size_t i = 0; i < buckets_.size(); ++i)
                output << name << "_bucket{" << item.first << ",le=\"" << buckets_[i]
                       << "\"} " << item.second.buckets[i] << '\n';
            output << name << "_bucket{" << item.first << ",le=\"+Inf\"} "
                   << item.second.count << '\n'
                   << name << "_sum{" << item.first << "} " << item.second.sum << '\n'
                   << name << "_count{" << item.first << "} " << item.second.count << '\n';
        }
    };
    renderHistogram("mywebserver_inference_request_duration_seconds",
                    "End-to-end inference request duration in seconds.", totalDuration_);
    renderHistogram("mywebserver_inference_queue_duration_seconds",
                    "Time spent waiting in the scheduler queue in seconds.", queueDuration_);
    renderHistogram("mywebserver_inference_backend_duration_seconds",
                    "Model execution duration in seconds.", backendDuration_);
    return output.str();
}
