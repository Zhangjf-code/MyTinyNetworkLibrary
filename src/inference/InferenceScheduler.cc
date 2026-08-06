#include "InferenceScheduler.h"

#include <utility>

InferenceScheduler::InferenceScheduler(std::shared_ptr<IInferenceBackend> backend,
                                       size_t workerCount,
                                       size_t queueCapacity,
                                       ModelConfig modelConfig)
    : registry_(new ModelRegistry), workerCount_(workerCount),
      queueCapacity_(queueCapacity),
      running_(false),
      stopping_(false)
{
    std::string ignored;
    registry_->add(std::move(modelConfig), std::move(backend), 0, &ignored);
}

InferenceScheduler::InferenceScheduler(std::shared_ptr<ModelRegistry> registry,
                                       size_t workerCount,
                                       size_t queueCapacity,
                                       std::shared_ptr<InferenceMetrics> metrics)
    : registry_(std::move(registry)), metrics_(std::move(metrics)), workerCount_(workerCount),
      queueCapacity_(queueCapacity), running_(false), stopping_(false)
{
    if (metrics_) metrics_->setQueueCapacity(queueCapacity_);
}

InferenceScheduler::~InferenceScheduler()
{
    stop();
}

bool InferenceScheduler::start(std::string* error)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) return true;
    if (!registry_ || workerCount_ == 0 || queueCapacity_ == 0)
    {
        if (error) *error = "scheduler requires a backend, workers, and queue capacity";
        return false;
    }
    if (!registry_->initialize(error)) return false;

    stopping_ = false;
    running_ = true;
    for (size_t i = 0; i < workerCount_; ++i)
        workers_.emplace_back(&InferenceScheduler::workerLoop, this);
    return true;
}

void InferenceScheduler::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        stopping_ = true;
    }
    notEmpty_.notify_all();
    for (std::thread& worker : workers_)
        if (worker.joinable()) worker.join();
    workers_.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    stopping_ = false;
}

InferenceScheduler::SubmitResult InferenceScheduler::submit(
    InferenceRequest request, Completion completion)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ || stopping_) return SubmitResult::kStopped;
    if (!registry_->hasModel(request.modelName)) return SubmitResult::kModelUnavailable;
    std::shared_ptr<IInferenceBackend> backend =
        registry_->resolve(&request.modelName, request.backend);
    if (!backend) return SubmitResult::kBackendUnavailable;
    if (tasks_.size() >= queueCapacity_) return SubmitResult::kQueueFull;
    Task task;
    task.request = std::move(request);
    task.completion = std::move(completion);
    task.backend = std::move(backend);
    task.enqueuedAt = std::chrono::steady_clock::now();
    tasks_.emplace_back(std::move(task));
    if (metrics_) metrics_->setQueueDepth(tasks_.size());
    notEmpty_.notify_one();
    return SubmitResult::kAccepted;
}

size_t InferenceScheduler::pendingTasks() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

bool InferenceScheduler::running() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return running_ && !stopping_;
}

void InferenceScheduler::workerLoop()
{
    while (true)
    {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            notEmpty_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
            if (metrics_) metrics_->setQueueDepth(tasks_.size());
        }

        const auto started = std::chrono::steady_clock::now();
        InferenceResult result = task.backend->infer(task.request);
        result.latency.queueMs = std::chrono::duration<double, std::milli>(
            started - task.enqueuedAt).count();
        result.latency.totalMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - task.enqueuedAt).count();
        if (metrics_) metrics_->recordCompletion(result);
        if (task.completion) task.completion(std::move(result));
    }
}
