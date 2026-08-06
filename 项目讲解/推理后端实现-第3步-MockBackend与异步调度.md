# 推理后端实现：第 3 步——MockBackend 与异步调度

## 目标

本步骤在不依赖 TensorFlow、TensorRT、CUDA 和图片解码库的情况下，验证完整推理服务链路：

```text
上传图片
→ HTTP 参数校验
→ 有界任务队列
→ 推理工作线程
→ MockBackend
→ 回到连接所属 EventLoop
→ 返回 JSON
```

最关键的验证目标是：模拟推理期间，EventLoop 线程不会被阻塞。

## 为什么先实现 MockBackend

如果一开始同时接入网络、线程调度、CUDA 和真实模型，出现问题时很难判断属于哪一层。MockBackend 遵循与真实后端相同的 `IInferenceBackend` 接口，但只执行：

1. 等待 20 ms，模拟推理耗时。
2. 返回一个固定的 `mock-object` 检测框。
3. 填写请求 ID、模型名和推理延迟。

因此可以先独立验证并发、队列、连接生命周期和 JSON 回包。

## InferenceScheduler

调度器位于：

```text
src/inference/InferenceScheduler.h
src/inference/InferenceScheduler.cc
```

当前配置：

```text
工作线程数：2
等待队列容量：128
Mock 推理时间：20 ms
```

其核心数据结构为：

```cpp
struct Task {
    InferenceRequest request;
    Completion completion;
    steady_clock::time_point enqueuedAt;
};

std::deque<Task> tasks_;
std::mutex mutex_;
std::condition_variable notEmpty_;
std::vector<std::thread> workers_;
```

任务提交结果是显式枚举：

```cpp
enum class SubmitResult {
    kAccepted,
    kQueueFull,
    kStopped,
};
```

调用者不需要阻塞等待队列空间。队列已满或调度器停止时立即得到结果。

## 有界队列与背压

上传图片可能接近 10 MiB。如果使用无限队列，客户端提交速度大于 GPU 处理速度时，进程内存会持续增长。

当前容量 128 指等待执行的任务数量，不包括正在两个工作线程中执行的任务。队满后返回：

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 1
Content-Type: application/json
```

```json
{
  "success": false,
  "request_id": "...",
  "error": {
    "code": "INFERENCE_QUEUE_FULL",
    "message": "inference queue is full"
  }
}
```

这叫作背压：服务明确拒绝超出处理能力的任务，而不是接受后无限积压。

## 工作线程流程

工作线程执行：

```text
等待 condition_variable
→ 从队首移动一个 Task
→ 释放队列锁
→ backend->infer(request)
→ 记录 queue_ms 和 total_ms
→ 执行 completion(result)
```

调用 `backend->infer()` 时不持有队列互斥锁，因此不同工作线程可以并行推理，HTTP 线程也可以继续提交任务。

`stop()` 采用平滑停止：停止接受新任务，但会处理完队列中已有的任务，再 join 所有工作线程。

## 异步 HTTP 入口

`HttpServer` 增加了 `AsyncHttpCallback`：

```cpp
using AsyncHttpCallback =
    std::function<bool(const TcpConnectionPtr&, const HttpRequest&)>;
```

返回 `true` 表示该请求已经被异步处理，不应创建栈上的同步响应；返回 `false` 时继续执行原来的同步 HTTP 回调。

目前只有 `/v1/infer` 使用异步入口，主页、favicon 和 `/upload` 仍使用原来的同步路由。

## 从推理线程回到 EventLoop

推理完成回调发生在工作线程。工作线程不直接操作 socket，而是先尝试锁定连接弱引用：

```text
weak_ptr<TcpConnection>
→ lock 失败：客户端已断开，丢弃结果
→ lock 成功：获取连接所属 EventLoop
→ queueInLoop(发送响应任务)
```

最终的 JSON 序列化响应通过 `EventLoop::queueInLoop()` 回到 I/O 线程发送。第 1 步修复的字符串所有权保证数据在延迟发送期间有效。

`InferenceRequest` 和 completion 都按移动语义进入队列；图片由 `shared_ptr<vector<uint8_t>>` 持有，不依赖已经重置的 `HttpRequest`。

## 每连接一个推理请求

本阶段不处理同一个 HTTP/1.1 连接上的多个并发推理请求。任务成功提交后调用：

```cpp
connection->stopRead();
```

这会取消该连接的 EPOLLIN 关注。推理完成后响应包含：

```http
Connection: close
Content-Length: ...
```

然后执行半关闭。这样不会发生同一连接上两个异步响应顺序颠倒的问题。

如果客户端已经把多个流水线请求一次性写进输入 Buffer，`HttpServer` 在异步回调接管第一个请求后也会立即停止解析剩余数据，确保不会绕过 `stopRead()` 重复提交。

未来如果要复用 keep-alive，需要为每个连接维护请求序号和有序响应队列。

## Mock 响应示例

```json
{
  "success": true,
  "request_id": "65863b10acf50-0",
  "model": "yolov8",
  "model_version": "mock-1",
  "backend": "auto",
  "image": {
    "width": 640,
    "height": 480
  },
  "latency_ms": {
    "queue": 0.02,
    "preprocess": 0.0,
    "inference": 20.5,
    "postprocess": 0.0,
    "total": 20.56
  },
  "detections": [
    {
      "class_id": 0,
      "label": "mock-object",
      "score": 0.9,
      "bbox": [64.0, 48.0, 320.0, 240.0]
    }
  ]
}
```

`backend` 为 `auto` 是因为结果来自 MockBackend，而不是真正的 TensorRT。真实后端接入后会填写实际执行后端。

## 测试

新增 `InferenceSchedulerTest`，使用一个可控制释放时间的测试 Backend，确定性验证：

1. 推理发生在非主线程。
2. 一个任务执行、一个任务等待时，第三个任务得到 `kQueueFull`。
3. 队列中的两个任务最终都执行 completion。
4. `stop()` 后提交得到 `kStopped`。

此外进行了真实端到端测试：启动服务器，使用 curl 向 `/v1/infer` 上传仓库中的 PNG 图片，得到 HTTP 200、结构化检测 JSON 和约 20 ms 的模拟推理延迟。

## 本步骤边界

本步骤尚未进行真实图片解码，MockBackend 返回的 `640×480` 和检测框都是固定值。下一步将接入 TensorFlow Backend；开始之前需要确定模型格式、TensorFlow 版本、运行设备以及模型的输入输出签名。
