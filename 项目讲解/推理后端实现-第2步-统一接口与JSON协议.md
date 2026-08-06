# 推理后端实现：第 2 步——统一接口与 JSON 协议

## 目标

本步骤在 HTTP 层和具体推理框架之间建立稳定边界：

1. HTTP 层将 multipart 表单转换为统一的 `InferenceRequest`。
2. TensorFlow 和 TensorRT 后端遵守同一个 `IInferenceBackend` 接口。
3. 推理结果使用统一的 `InferenceResult` 表达。
4. HTTP 响应使用固定 JSON Schema，不暴露框架内部对象。

本步骤还没有执行真实推理。合法请求返回 `501 Not Implemented`，表示协议验证成功、推理后端尚未连接。

## JSON 依赖

项目使用 `nlohmann/json` v3.12.0，单头文件位于：

```text
third_party/nlohmann/json.hpp
```

文件 SHA-256：

```text
aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
```

依赖随源码一起构建，目标机器无需预装 JSON 开发包，也不需要在 CMake 配置阶段联网。

推理服务中的 JSON 通常只有几 KB，主要耗时来自图片解码和模型推理，所以这里优先考虑 API 清晰度、类型转换和可测试性，而不是极限 JSON 吞吐。

## 统一数据类型

数据结构定义在 `src/inference/InferenceTypes.h`。

### 后端类型

```cpp
enum class BackendType {
    kAuto,
    kTensorFlow,
    kTensorRT,
};
```

`auto` 表示由后续的模型注册中心选择后端。HTTP 和模型配置中使用小写字符串：`auto`、`tensorflow`、`tensorrt`。

### 推理请求

```cpp
struct InferenceRequest {
    std::string requestId;
    std::string modelName;
    BackendType backend;
    float confidenceThreshold;
    ImageData image;
};
```

图片内容使用：

```cpp
std::shared_ptr<std::vector<uint8_t>> encoded;
```

这样请求从 HTTP 层进入异步队列时拥有独立的数据生命周期，也允许调度器和推理工作线程共享图片而不反复复制。

### 检测结果

每个检测框包含：

```cpp
struct Detection {
    int classId;
    std::string label;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};
```

坐标统一使用原图像素坐标下的 `[x1, y1, x2, y2]`。无论模型原始输出是中心点坐标、归一化坐标还是特征图坐标，都由后处理层转换成这种形式。

`InferenceResult` 还包含模型名称、版本、实际后端、图像尺寸、各阶段延迟和结构化错误。

## 后端抽象

`src/inference/IInferenceBackend.h` 定义：

```cpp
class IInferenceBackend {
public:
    virtual bool load(const ModelConfig&, std::string* error) = 0;
    virtual bool warmup(std::string* error) = 0;
    virtual InferenceResult infer(const InferenceRequest&) = 0;
    virtual bool healthy() const = 0;
    virtual BackendType type() const = 0;
};
```

生命周期为：

```text
创建 Backend
→ load 模型
→ warmup 预热
→ 多次 infer
→ 销毁 Backend
```

网络层、JSON 层和未来的任务调度器只依赖该接口。TensorFlow 的 Session、TensorRT 的 Engine、CUDA stream 等类型不会渗透到公共接口。

## HTTP API

端点：

```http
POST /v1/infer
Content-Type: multipart/form-data
```

multipart 字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `image` | 文件 | 是 | 无 | 只允许一个文件 |
| `model` | 文本 | 否 | `default` | 模型逻辑名称 |
| `backend` | 文本 | 否 | `auto` | `auto`/`tensorflow`/`tensorrt` |
| `threshold` | 浮点文本 | 否 | `0.25` | 范围 `[0, 1]` |

当前只接受 `image/jpeg` 和 `image/png`。这里校验的是 multipart 声明的媒体类型；后续图片解码阶段还需要校验真实文件内容，不能只信任客户端 Header。

示例请求：

```bash
curl -X POST http://127.0.0.1:10000/v1/infer \
  -F 'model=yolov8' \
  -F 'backend=tensorrt' \
  -F 'threshold=0.4' \
  -F 'image=@test.jpg;type=image/jpeg'
```

## 成功响应 Schema

```json
{
  "success": true,
  "request_id": "64f0b23a-1",
  "model": "yolov8",
  "model_version": "1",
  "backend": "tensorrt",
  "image": {
    "width": 1280,
    "height": 720
  },
  "latency_ms": {
    "queue": 0.5,
    "preprocess": 2.1,
    "inference": 7.8,
    "postprocess": 1.2,
    "total": 11.6
  },
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "score": 0.95,
      "bbox": [10.0, 20.0, 200.0, 400.0]
    }
  ]
}
```

## 错误响应 Schema

所有 API 错误都使用相同结构：

```json
{
  "success": false,
  "request_id": "64f0b23a-2",
  "error": {
    "code": "INVALID_THRESHOLD",
    "message": "threshold must be a number between 0 and 1"
  }
}
```

当前状态码映射：

| 状态码 | 场景 |
| --- | --- |
| `400` | 字段缺失、后端名称非法、阈值非法 |
| `405` | `/v1/infer` 使用非 POST 方法，并返回 `Allow: POST` |
| `415` | 图片媒体类型不是 JPEG/PNG |
| `501` | 请求合法，但真实推理后端尚未接入 |

错误码面向程序处理，保持稳定；`message` 面向开发者阅读，可以进一步优化措辞。

## 请求 ID

每次进入推理 API 都生成请求 ID，包括参数校验失败的请求。当前 ID 由微秒时间戳和进程内原子递增序号组成，满足单进程内并发唯一性。

未来如果需要跨机器全局唯一和按时间排序，可以替换为 UUIDv7；替换过程不会影响其他数据结构。

## 测试

`InferenceApiTest` 覆盖：

1. 检测框、坐标、后端和图像尺寸的 JSON 序列化。
2. 合法 multipart 推理请求在未接后端时返回结构化 `501`。
3. 非法阈值返回 `400` 和稳定错误码。
4. 所有 API 响应都能被 JSON 解析器重新解析。

执行：

```bash
cmake -S . -B build
cmake --build build --target InferenceApiTest
ctest --test-dir build --output-on-failure
```

## 本步骤边界

当前 Controller 会把上传文件复制进 `InferenceRequest`，然后返回 `501`。下一步加入 MockBackend 和异步推理调度器后，这个请求将被送入有界队列，工作线程完成处理后再把 JSON 响应投递回连接所属的 EventLoop。
