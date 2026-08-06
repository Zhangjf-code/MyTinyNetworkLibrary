#pragma once

#include "IInferenceBackend.h"
#include "ModelRegistry.h"
#include "InferenceMetrics.h"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class InferenceScheduler
{
public:
    using Completion = std::function<void(InferenceResult)>;

    enum class SubmitResult
    {
        kAccepted,
        kQueueFull,
        kStopped,
        kBackendUnavailable,
        kModelUnavailable,
    };

    InferenceScheduler(std::shared_ptr<IInferenceBackend> backend,
                       size_t workerCount,
                       size_t queueCapacity,
                       ModelConfig modelConfig = ModelConfig());
    InferenceScheduler(std::shared_ptr<ModelRegistry> registry,
                       size_t workerCount,
                       size_t queueCapacity,
                       std::shared_ptr<InferenceMetrics> metrics = nullptr);
    ~InferenceScheduler();

    bool start(std::string* error);
    void stop();
    SubmitResult submit(InferenceRequest request, Completion completion);

    size_t pendingTasks() const;
    bool running() const;

private:
    struct Task
    {
        InferenceRequest request;
        Completion completion;
        std::shared_ptr<IInferenceBackend> backend;
        std::chrono::steady_clock::time_point enqueuedAt;
    };

    void workerLoop();

    std::shared_ptr<ModelRegistry> registry_;
    std::shared_ptr<InferenceMetrics> metrics_;
    const size_t workerCount_;
    const size_t queueCapacity_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::deque<Task> tasks_;
    std::vector<std::thread> workers_;
    bool running_;
    bool stopping_;
};
