# 推理后端实现：第 5 步——TensorRT GPU 后端

## 目标

本步骤在第 4 步统一接口和 YOLOv8 前后处理之上增加真实 TensorRT GPU 推理：

```text
JPEG/PNG
→ CPU 解码与 letterbox
→ NHWC 转 NCHW
→ CUDA H2D
→ TensorRT enqueueV3
→ CUDA D2H
→ YOLOv8 NMS
→ HTTP JSON
```

TensorRTBackend 继续实现 `IInferenceBackend`，因此 HTTP Controller、JSON 协议和异步调度器不依赖 CUDA/TensorRT 类型。

## 开发环境

本步骤实际验证环境：

```text
GPU: NVIDIA GeForce RTX 4060 Ti 16 GB
Compute Capability: 8.9
CUDA Toolkit: 12.4
TensorRT: 10.0.1.6
Engine precision: FP16
```

TensorRT 采用 NVIDIA Linux x86_64、CUDA 12.4 tar 包，解压到：

```text
.deps/tensorrt-10.0.1.6/
```

`.deps/` 不提交到 Git。安装脚本 `tools/fetch_tensorrt.sh` 会下载官方包、校验 SHA-256，并只提取 C++ Header、动态库和 `trtexec` 等必要目录。

## 从 PT 到 Engine

模型转换链路：

```text
models/yolov8n.pt
→ Ultralytics ONNX export
→ models/yolov8n.onnx
→ trtexec FP16 build
→ models/yolov8n_fp16.engine
```

导出固定 Batch 1 ONNX：

```bash
python tools/export_yolov8_onnx.py \
  --weights models/yolov8n.pt \
  --imgsz 640 \
  --opset 12
```

生成 FP16 Engine：

```bash
./tools/build_tensorrt_engine.sh
```

Engine 构建阶段会针对当前 GPU 搜索执行 tactic。本机首次构建约 130 秒，生成的 Engine 约 9.34 MiB。

TensorRT Engine 通常绑定 TensorRT 版本和 GPU 架构。部署机器的版本或 GPU 架构改变时，应从 ONNX 重新构建 Engine，而不是直接复制现有 `.engine`。

## Engine 张量契约

加载时严格验证名称、数据类型和维度：

```text
input name:  images
input dtype: float32
input shape: [1, 3, 640, 640]

output name:  output0
output dtype: float32
output shape: [1, 84, 8400]
```

FP16 表示 Engine 内部可以选择 FP16 tactic，并不要求网络边界也改为 FP16。本 Engine 的输入和输出仍是 float32，减少与公共前后处理的耦合。

## NHWC 与 NCHW

第 4 步 TensorFlow SavedModel 使用 NHWC：

```text
[1, 640, 640, 3]
```

Ultralytics ONNX/TensorRT 使用 NCHW：

```text
[1, 3, 640, 640]
```

公共 `YoloV8Processor` 保留 NHWC RGB 输出。TensorRTBackend 在上传 GPU 前转换为 NCHW，因此 TensorFlow 后端行为不受影响，letterbox 参数和后处理仍能完全共享。

## TensorRTBackend 生命周期

加载过程：

```text
cudaSetDevice
→ 读取序列化 Engine
→ createInferRuntime
→ deserializeCudaEngine
→ createExecutionContext
→ 校验 images/output0
→ 创建 CUDA Stream
→ 分配持久化 input/output GPU Buffer
→ setTensorAddress
```

GPU Buffer 和 CUDA Stream 在模型加载后复用，避免每个 HTTP 请求重复执行 `cudaMalloc`。析构时依次释放 Buffer、Stream、ExecutionContext、Engine 和 Runtime。

## 单次推理

```text
NHWC 转 NCHW
→ cudaMemcpyAsync(H2D)
→ enqueueV3
→ cudaMemcpyAsync(D2H)
→ cudaStreamSynchronize
```

当前一个 TensorRTBackend 只有一个 ExecutionContext 和一个 CUDA Stream。两个调度线程可以并行做图片解码和后处理，但 GPU 执行区由互斥锁保护。这是第一版明确且安全的并发策略；后续可按吞吐需求改为 ExecutionContext/Stream 池。

## 可选构建

默认构建不要求 TensorRT：

```bash
cmake -S . -B build
cmake --build build -j
```

启用 TensorRT：

```bash
cmake -S . -B build \
  -DENABLE_TENSORRT=ON \
  -DTENSORRT_ROOT="$PWD/.deps/tensorrt-10.0.1.6" \
  -DCUDA_ROOT=/usr/local/cuda-12.4
cmake --build build -j
```

该仓库原有 CMake 把共享库和服务器二进制输出到源码目录。不同配置的多个 build 目录会写入同一个 `lib/libmytinymuduo.so`，所以不要并行构建 TensorFlow、TensorRT 和默认配置；切换配置后应重新完整构建对应目录。

启动代码在 `ENABLE_TENSORRT=ON` 时加载 `models/yolov8n_fp16.engine`。如果 TensorFlow 和 TensorRT 同时打开，当前阶段启动代码优先选择 TensorRT；多模型、多后端同时注册和请求级路由将在后续模型注册中心步骤完成。

## 测试结果

`TensorRTBackendTest` 完成以下真实验证：

- 反序列化 FP16 Engine。
- 创建真实 CUDA ExecutionContext。
- warmup。
- 推理 Ultralytics 官方 `bus.jpg`。
- 验证图片尺寸、后端类型、非空检测结果和 `bus` 类别。

`trtexec` 一秒短基准结果：

```text
GPU compute mean: 约 1.13 ms
Host latency mean: 约 1.74 ms
Throughput: 约 553.6 qps
```

该基准使用随机张量且不含 JPEG 解码、letterbox、NMS 和 HTTP，仅用于验证 Engine 本身。

HTTP 端到端请求：

```bash
curl -X POST http://127.0.0.1:10000/v1/infer \
  -F model=yolov8n \
  -F backend=tensorrt \
  -F threshold=0.25 \
  -F 'image=@root/picture/ultralytics_bus.jpg;type=image/jpeg'
```

返回 HTTP 200，检测到 4 个 `person` 和 1 个 `bus`，响应中的 `backend` 为 `tensorrt`。

## 本步骤边界

- 仅支持固定 Batch 1、640×640。
- 仅支持 YOLOv8 COCO detection 输出 `[1,84,8400]`。
- 当前使用 FP16 Engine，没有实现 INT8 校准。
- GPU 执行暂时由单 ExecutionContext 串行化。
- Engine 需要在目标 TensorRT/GPU 环境重新生成。
- 当前一次服务进程只选择一个实际后端；后续再实现多后端注册和请求级路由。
