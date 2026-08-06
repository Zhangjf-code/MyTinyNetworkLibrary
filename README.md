# MyWebServer：C++ 多后端目标检测推理服务

MyWebServer 是一个基于 Linux Epoll 和主从 Reactor 网络模型实现的 C++11 在线目标检测服务。浏览器或其他 HTTP 客户端通过 `multipart/form-data` 上传 JPEG/PNG 图片，服务端异步调度 TensorFlow SavedModel 或 TensorRT Engine 完成 YOLOv8 目标检测，并返回统一 JSON 结果。

当前仓库已包含完整的 Web 推理控制台、模型注册与路由、健康检查、Prometheus 指标以及 NVIDIA GPU 容器部署配置。

## 主要能力

- 自研非阻塞 HTTP WebServer：Epoll、LT 模式、主从 Reactor、one loop per thread。
- 增量解析 HTTP 请求，支持 `multipart/form-data` 图片上传及请求大小限制。
- 有界异步推理队列，模型计算不会阻塞网络 EventLoop。
- TensorFlow 2.18 C API CPU 推理，模型格式为原生 SavedModel。
- TensorRT 10.0.1.6 FP16 GPU 推理，使用 CUDA 12.4。
- 统一的图片预处理、YOLOv8 后处理、置信度过滤和 NMS。
- 模型注册中心支持同一模型的多个后端版本及优先级路由。
- `auto` 默认优先选择 TensorRT，启动加载失败时可使用 TensorFlow；显式指定后端时严格路由。
- Web 控制台支持图片预览、模型/后端选择、检测框绘制和耗时展示。
- 提供健康检查、模型目录及 Prometheus 指标。
- 支持环境变量配置、SIGINT/SIGTERM 优雅停机和 NVIDIA GPU 容器部署。

## 整体架构

```mermaid
flowchart LR
    Client[浏览器 / HTTP 客户端] -->|multipart 图片| HTTP[HTTP Server<br/>主从 Reactor]
    HTTP --> API[InferenceController<br/>参数校验与 JSON 协议]
    API --> Queue[InferenceScheduler<br/>有界队列 + worker]
    Queue --> Registry[ModelRegistry<br/>模型与后端路由]
    Registry --> TRT[TensorRT Backend<br/>CUDA / FP16]
    Registry --> TF[TensorFlow Backend<br/>SavedModel / CPU]
    TRT --> Result[YOLOv8 后处理<br/>检测结果]
    TF --> Result
    Result --> Client
    HTTP --> Management[/healthz · /v1/models · /metrics]
```

网络线程负责连接、HTTP 解析和响应发送；推理 worker 负责图片解码、预处理、模型执行和后处理。推理完成后，结果会重新投递到连接所属的 EventLoop，保证连接操作的线程归属正确。

## 目录结构

```text
src/http/          HTTP 协议、服务器入口和测试
src/muduo_core/    Reactor、EventLoop、TCP 连接等网络核心
src/inference/     模型接口、注册中心、调度器及 TF/TRT 后端
src/api/           推理与管理 HTTP Controller、JSON 序列化
root/              Web 推理控制台和示例图片
models/            模型注册表及 YOLOv8 模型产物
tools/             依赖下载、模型导出和 Engine 构建脚本
项目讲解/          网络模块与推理后端的分步实现文档
```

## 快速开始

### 1. 基础依赖

以下命令以 Ubuntu/Debian 为例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake curl
```

仅验证网络和异步调度框架时，可以使用不依赖 TensorFlow、CUDA 或 TensorRT 的 MockBackend：

```bash
cmake -S . -B build \
  -DENABLE_TENSORFLOW=OFF \
  -DENABLE_TENSORRT=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
./src/http/HttpServer
```

服务默认监听 `10000` 端口。打开 <http://127.0.0.1:10000/> 即可访问 Web 控制台。

### 2. TensorFlow CPU 后端

下载经过 SHA256 校验的 TensorFlow C 2.18.0：

```bash
./tools/fetch_tensorflow_c.sh
```

构建并启动：

```bash
cmake -S . -B build-tf \
  -DENABLE_TENSORFLOW=ON \
  -DTENSORFLOW_ROOT="$PWD/.deps/tensorflow-2.18.0" \
  -DENABLE_TENSORRT=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-tf -j"$(nproc)"

LD_LIBRARY_PATH="$PWD/.deps/tensorflow-2.18.0/lib" \
  ./src/http/HttpServer
```

仓库中的默认 SavedModel 位于 `models/yolov8n_tf218_saved_model/`。

### 3. TensorRT GPU 后端

要求宿主机具有 NVIDIA 驱动、CUDA 12.4 Toolkit，并且 `nvidia-smi` 可正常执行。下载 TensorRT 10.0.1.6：

```bash
./tools/fetch_tensorrt.sh
```

同时启用 TensorFlow 和 TensorRT：

```bash
cmake -S . -B build-gpu \
  -DENABLE_TENSORFLOW=ON \
  -DTENSORFLOW_ROOT="$PWD/.deps/tensorflow-2.18.0" \
  -DENABLE_TENSORRT=ON \
  -DTENSORRT_ROOT="$PWD/.deps/tensorrt-10.0.1.6" \
  -DCUDA_ROOT=/usr/local/cuda-12.4 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j"$(nproc)"

LD_LIBRARY_PATH="$PWD/.deps/tensorrt-10.0.1.6/lib:$PWD/.deps/tensorflow-2.18.0/lib:/usr/local/cuda-12.4/lib64" \
  ./src/http/HttpServer
```

默认 FP16 Engine 是在 RTX 4060 Ti、TensorRT 10.0.1.6 上生成的。TensorRT Engine 通常不能跨 TensorRT 版本或 GPU 架构直接复用；部署硬件变化时，应从 `models/yolov8n.onnx` 在目标机器重新构建：

```bash
./tools/build_tensorrt_engine.sh
```

模型文件、来源、校验值及重新导出说明参见 [models/README.md](models/README.md)。

## HTTP API

### 执行推理

```http
POST /v1/infer
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `image` | 是 | 单个 JPEG 或 PNG 文件 |
| `model` | 否 | 模型名称，默认使用注册表的默认模型 |
| `backend` | 否 | `auto`、`tensorflow` 或 `tensorrt`，默认 `auto` |
| `threshold` | 否 | 0～1 的置信度阈值，默认 `0.25` |

示例：

```bash
curl -X POST http://127.0.0.1:10000/v1/infer \
  -F image=@root/picture/ultralytics_bus.jpg \
  -F model=yolov8n \
  -F backend=auto \
  -F threshold=0.25
```

成功响应示例：

```json
{
  "success": true,
  "request_id": "658651551d4a8-0",
  "model": "yolov8n",
  "model_version": "trt-10.0.1-fp16",
  "backend": "tensorrt",
  "image": {"width": 810, "height": 1080},
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "score": 0.89,
      "bbox": [670.2, 380.3, 809.7, 879.4]
    }
  ],
  "latency_ms": {
    "queue": 0.03,
    "preprocess": 25.42,
    "inference": 30.94,
    "postprocess": 0.97,
    "total": 57.38
  }
}
```

`bbox` 顺序为 `[x1, y1, x2, y2]`，坐标对应上传图片的原始尺寸。

### 服务管理

| 接口 | 作用 |
| --- | --- |
| `GET /healthz` | 返回 `healthy`、`degraded` 或 `unavailable`；完全不可服务时为 HTTP 503 |
| `GET /v1/models` | 返回默认模型及所有注册后端的版本、优先级和健康状态 |
| `GET /metrics` | 返回 Prometheus 0.0.4 文本指标 |

```bash
curl http://127.0.0.1:10000/healthz
curl http://127.0.0.1:10000/v1/models
curl http://127.0.0.1:10000/metrics
```

Prometheus 指标包括请求结果计数、拒绝原因、调度队列深度/容量，以及端到端、排队和后端执行延迟直方图。

## 模型注册与路由

默认配置位于 [models/registry.json](models/registry.json)。同一模型可以注册多个后端：

```json
{
  "default_model": "yolov8n",
  "models": [
    {
      "name": "yolov8n",
      "backend": "tensorrt",
      "path": "yolov8n_fp16.engine",
      "priority": 100,
      "enabled": true
    },
    {
      "name": "yolov8n",
      "backend": "tensorflow",
      "path": "yolov8n_tf218_saved_model",
      "priority": 10,
      "enabled": true
    }
  ]
}
```

相对模型路径以注册表文件所在目录为基准。`auto` 路由选择健康且优先级最高的后端；显式请求 `tensorflow` 或 `tensorrt` 时不会静默切换到另一后端。当前回退发生在启动加载阶段，不会在单次推理失败后自动重试其他后端。

## 运行参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MYWEBSERVER_PORT` | `10000` | HTTP 监听端口 |
| `MYWEBSERVER_INFERENCE_WORKERS` | `2` | 推理 worker 数量 |
| `MYWEBSERVER_QUEUE_CAPACITY` | `128` | 有界推理队列容量 |
| `MYWEBSERVER_MODEL_REGISTRY` | `models/registry.json` 的编译期绝对路径 | 模型注册表路径 |
| `MYWEBSERVER_HTML_PATH` | `root/index4.html` 的编译期绝对路径 | Web 首页路径 |

示例：

```bash
MYWEBSERVER_PORT=8080 \
MYWEBSERVER_INFERENCE_WORKERS=4 \
MYWEBSERVER_QUEUE_CAPACITY=256 \
./src/http/HttpServer
```

数字配置会做范围校验。服务收到 SIGINT 或 SIGTERM 后会停止 EventLoop、排空并停止推理调度器，然后释放模型资源。

## Docker GPU 部署

要求安装 Docker Engine、Docker Compose 和 NVIDIA Container Toolkit：

```bash
docker compose up --build -d
docker compose logs -f inference
```

Compose 会申请全部 NVIDIA GPU、映射宿主机 `10000` 端口，并使用 `/healthz` 进行容器健康检查：

```bash
docker compose ps
curl http://127.0.0.1:10000/healthz
```

停止服务：

```bash
docker compose down
```

Dockerfile 会在构建阶段下载并校验 TensorFlow/TensorRT，不会复制本机 `.deps`。详细说明和 Engine 可移植性注意事项参见[第 10 步文档](项目讲解/推理后端实现-第10步-容器化与部署.md)。

## 测试

构建完成后运行：

```bash
ctest --test-dir build --output-on-failure
```

启用 TensorRT 的测试需要访问 NVIDIA GPU，并确保动态库可以被找到：

```bash
LD_LIBRARY_PATH="$PWD/.deps/tensorrt-10.0.1.6/lib:$PWD/.deps/tensorflow-2.18.0/lib:/usr/local/cuda-12.4/lib64" \
  ctest --test-dir build-gpu --output-on-failure
```

当前完整联合构建包含 8 项测试，覆盖 HTTP 增量解析、推理 API、异步调度、模型注册、管理接口、YOLOv8 处理、TensorFlow 和 TensorRT 后端。

## 分步实现文档

1. [HTTP 基础改造](项目讲解/推理后端实现-第1步-HTTP基础改造.md)
2. [统一接口与 JSON 协议](项目讲解/推理后端实现-第2步-统一接口与JSON协议.md)
3. [MockBackend 与异步调度](项目讲解/推理后端实现-第3步-MockBackend与异步调度.md)
4. [TensorFlow 后端](项目讲解/推理后端实现-第4步-TensorFlow后端.md)
5. [TensorRT 后端](项目讲解/推理后端实现-第5步-TensorRT后端.md)
6. [模型注册中心与后端路由](项目讲解/推理后端实现-第6步-模型注册中心与后端路由.md)
7. [Web 推理控制台](项目讲解/推理后端实现-第7步-Web推理控制台.md)
8. [服务管理与可观测性](项目讲解/推理后端实现-第8步-服务管理与可观测性.md)
9. [Prometheus 指标与性能统计](项目讲解/推理后端实现-第9步-Prometheus指标与性能统计.md)
10. [容器化与部署](项目讲解/推理后端实现-第10步-容器化与部署.md)

网络基础模块文档位于 [`项目讲解/`](项目讲解/)：定时器、时间轮、异步日志、HTTP、内存池和数据库连接池等。

## 当前边界

- 模型注册表只在进程启动时加载，尚不支持热更新。
- 单次推理执行失败后不会自动切换后端重试，避免隐式增加延迟及产生重复执行语义。
- `/healthz` 使用后端内部健康状态，尚未定时执行主动推理探针。
- `/healthz`、`/v1/models` 和 `/metrics` 尚未提供认证，生产环境应通过内网或反向代理限制访问。
- TensorRT Engine 需要根据部署 GPU、CUDA 和 TensorRT 版本评估并重新生成。
- Prometheus 指标保存在进程内存中，服务重启后从零开始。

## 模型许可

仓库中的 YOLOv8 模型来源及导出信息记录在 [models/README.md](models/README.md)。Ultralytics 软件与模型资产适用其自身许可条款；用于分发或商业场景前，请自行确认对应许可要求。
