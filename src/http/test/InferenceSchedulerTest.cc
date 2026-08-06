#include "InferenceScheduler.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

class BlockingBackend final : public IInferenceBackend
{
public:
    bool load(const ModelConfig&, std::string*) override { loaded_ = true; return true; }
    bool warmup(std::string*) override { return loaded_; }
    bool healthy() const override { return loaded_; }
    BackendType type() const override { return BackendType::kAuto; }

    InferenceResult infer(const InferenceRequest& request) override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            started_ = true;
        }
        condition_.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return released_; });
        }
        InferenceResult result;
        result.success = true;
        result.requestId = request.requestId;
        result.modelName = request.modelName;
        return result;
    }

    void waitUntilStarted()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return started_; });
    }

    void release()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        condition_.notify_all();
    }

private:
    bool loaded_ = false;
    bool started_ = false;
    bool released_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

int main()
{
    std::shared_ptr<BlockingBackend> backend(new BlockingBackend);
    InferenceScheduler scheduler(backend, 1, 1);
    std::string error;
    const bool started = scheduler.start(&error);
    assert(started);

    const std::thread::id mainThread = std::this_thread::get_id();
    std::mutex completedMutex;
    std::condition_variable completedCondition;
    int completed = 0;
    bool ranOutsideMainThread = false;
    const auto completion = [&](InferenceResult result) {
        assert(result.success);
        {
            std::lock_guard<std::mutex> lock(completedMutex);
            ++completed;
            ranOutsideMainThread = ranOutsideMainThread || std::this_thread::get_id() != mainThread;
        }
        completedCondition.notify_all();
    };

    InferenceRequest first;
    first.requestId = "first";
    const InferenceScheduler::SubmitResult firstSubmit = scheduler.submit(first, completion);
    assert(firstSubmit == InferenceScheduler::SubmitResult::kAccepted);
    backend->waitUntilStarted();

    InferenceRequest second;
    second.requestId = "second";
    const InferenceScheduler::SubmitResult secondSubmit = scheduler.submit(second, completion);
    assert(secondSubmit == InferenceScheduler::SubmitResult::kAccepted);
    InferenceRequest third;
    third.requestId = "third";
    const InferenceScheduler::SubmitResult thirdSubmit = scheduler.submit(third, completion);
    assert(thirdSubmit == InferenceScheduler::SubmitResult::kQueueFull);

    backend->release();
    {
        std::unique_lock<std::mutex> lock(completedMutex);
        const bool finished = completedCondition.wait_for(
            lock, std::chrono::seconds(2), [&] { return completed == 2; });
        assert(finished);
    }
    assert(ranOutsideMainThread);
    scheduler.stop();
    const InferenceScheduler::SubmitResult stoppedSubmit = scheduler.submit(third, completion);
    assert(stoppedSubmit == InferenceScheduler::SubmitResult::kStopped);

    std::cout << "Inference scheduler tests passed" << std::endl;
    return 0;
}
