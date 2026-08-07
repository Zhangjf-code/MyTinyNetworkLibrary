# 推理后端实现：第 9 步——Prometheus 指标与性能统计

## 1. 本步目标

第 8 步的 `/healthz` 能判断服务是否可用，但不能回答“处理了多少请求”、“哪个后端更慢”或“调度队列是否积压”。本步新增进程内指标注册表和 `GET /metrics` 接口，直接输出 Prometheus 0.0.4 文本格式，不增加第三方运行库。

## 2. 采集链路

`src/inference/InferenceMetrics.h` 和 `.cc` 实现线程安全的指标容器。同一个 `shared_ptr<InferenceMetrics>` 被传给：

- `InferenceController`：记录 HTTP 校验失败、模型/后端不可用、队列满和调度器停止等提交失败。
- `InferenceScheduler`：入队和出队时更新队列深度，推理完成后使用结果中的实际模型和实际后端记录成功/失败及延迟。
- `ManagementController`：对指标快照进行 Prometheus 序列化。

所有修改操作和抓取快照都受互斥锁保护，不会在推理工作线程与 HTTP EventLoop 并发访问时产生数据竞争。

## 3. `/metrics` 接口

请求：

```bash
curl http://127.0.0.1:10000/metrics
```

响应类型为：

```text
text/plain; version=0.0.4; charset=utf-8
```

接口仅允许 `GET`，其他方法返回 HTTP 405。

## 4. 指标说明

### 4.1 请求计数

```text
mywebserver_inference_requests_total{model="yolov8n",backend="tensorrt",status="success"} 1
```

`status` 可为 `success`、`error`、`invalid_request` 或 `rejected`。已进入调度器的请求使用推理结果中的实际 backend，所以客户端传入 `auto` 但路由到 TensorRT 时，指标标签是 `tensorrt`。

### 4.2 提交拒绝

```text
mywebserver_inference_rejections_total{reason="INFERENCE_QUEUE_FULL"} 1
```

该计数器区分请求参数无效、模型不存在、指定后端不可用、队列满和调度器未运行等原因。

### 4.3 队列状态

```text
mywebserver_inference_queue_depth 0
mywebserver_inference_queue_capacity 128
```

`queue_depth` 是正在等待工作线程的任务数，不包含已经被工作线程取出、正在执行的任务。

### 4.4 延迟直方图

提供三组单位为秒的 Prometheus Histogram：

- `mywebserver_inference_request_duration_seconds`：入队到推理完成的端到端耗时。
- `mywebserver_inference_queue_duration_seconds`：在调度队列中等待的时间。
- `mywebserver_inference_backend_duration_seconds`：模型后端报告的纯推理执行耗时。

固定桶边界从 1 ms 到 10 s，每组都输出累计 `_bucket`、`_sum` 和 `_count`，可以使用 PromQL 计算 P95：

```promql
histogram_quantile(0.95,
  sum by (le, backend) (rate(mywebserver_inference_request_duration_seconds_bucket[5m])))
```

## 5. 标签与基数控制

指标只使用模型名、后端类型、有限的结果状态和有限的拒绝原因。`request_id`、文件名、错误文本和客户端地址不会进入标签，避免每次请求创建新时序导致 Prometheus 内存持续增长。标签值也会对反斜杠、双引号和换行符进行转义。

## 6. Prometheus 抓取示例

```yaml
scrape_configs:
  - job_name: mywebserver-inference
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:10000"]
```

`/metrics` 当前没有认证，部署时应仅向监控网络开放，或由反向代理限制访问来源。

## 7. 测试与真实验证

`ManagementApiTest` 新增了以下检查：

- Prometheus Content-Type 正确。
- 成功请求与拒绝请求能生成指标。
- 队列深度和延迟直方图存在。
- 非 `GET` 请求返回 405。

TensorFlow + TensorRT Debug 联合构建的 8 项测试全部通过。真实上传 YOLOv8 测试图片后，`auto` 路由到 TensorRT，端到端耗时约 57.4 ms、模型执行约 30.9 ms，且 `/metrics` 中的计数、队列 gauge 和直方图全部更新。

## 8. 当前边界

- 指标只保存在内存中，进程重启后从零开始，历史数据由 Prometheus 保存。
- 当前没有单独统计“正在执行”的 worker 数量。
- 还没有进程 CPU、内存和 GPU 显存指标；这些通常由 node_exporter 和 NVIDIA DCGM Exporter 提供。
