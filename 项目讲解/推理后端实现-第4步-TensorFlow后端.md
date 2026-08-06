# 推理后端实现：第 4 步——TensorFlow CPU 后端

## 目标

本步骤将第 3 步的 MockBackend 替换为真实 TensorFlow CPU 推理：

```text
JPEG/PNG 字节
→ stb_image 解码为 RGB
→ YOLOv8 letterbox + 归一化
→ TensorFlow C API SessionRun
→ 解码 [1,84,8400]
→ 置信度过滤 + 按类别 NMS
→ 原图坐标检测框
→ HTTP JSON
```

TensorFlow 后端继续实现 `IInferenceBackend`，HTTP、调度器和 JSON 层不依赖 TensorFlow 类型。

## 模型来源和版本选择

模型采用 Ultralytics 官方 `yolov8n.pt`，通过 Ultralytics 8.3.0 的 `format=saved_model` 流程导出。

选择 YOLOv8n 的原因：

- 模型体积和 CPU 推理时间较小，适合先验证服务链路。
- 输出是标准 YOLOv8 detection tensor，容易与后续 TensorRT 后端统一后处理。
- 使用 COCO 80 类，测试图片和标签容易复现。

TensorFlow 官方预编译 Linux x86 C API 最后提供到 2.18，因此导出环境和 C++ 运行库统一使用 2.18。不能使用 2.19 导出后再假设 2.18 一定向前兼容。

保留的模型文件：

```text
models/yolov8n.pt
models/yolov8n_tf218_saved_model/
```

中间 ONNX、TFLite 和没有标准签名的原始 SavedModel 不进入最终项目。

模型和权重的 SHA-256、来源与许可提醒记录在 `models/README.md`。

## 标准 SavedModel Signature

Ultralytics 8.3.0 经 onnx2tf 生成的 SavedModel 可以在 Python 中直接调用，但没有 `serving_default` SignatureDef。TensorFlow C API 需要稳定的图输入输出名称，因此导出脚本增加一层不改变数值的 `tf.Module` 包装。

最终签名：

```text
signature: serving_default

input key:    images
tensor name:  serving_default_images:0
dtype:        float32
shape:        [1, 640, 640, 3]
layout:       NHWC

output key:   output0
tensor name:  PartitionedCall:0
dtype:        float32
shape:        [1, 84, 8400]
```

签名元数据同时保存在：

```text
models/yolov8n_tf218_saved_model/signature.json
```

导出脚本为 `tools/export_yolov8_savedmodel.py`。脚本也处理 Ultralytics 在可选 TFLite USB metadata 阶段失败、但 SavedModel 已经成功写出的情况。

## TensorFlow C API 依赖

运行库不提交到 Git，安装位置默认为：

```text
.deps/tensorflow-2.18.0/
```

安装：

```bash
./tools/fetch_tensorflow_c.sh
```

脚本从 TensorFlow 官方 GCS 下载 Linux CPU 包，并校验 SHA-256 后解压。`.deps/` 已加入 `.gitignore`。

TensorFlow 是可选构建项：

```bash
cmake -S . -B build \
  -DENABLE_TENSORFLOW=ON \
  -DTENSORFLOW_ROOT="$PWD/.deps/tensorflow-2.18.0"
cmake --build build -j
```

不开启 `ENABLE_TENSORFLOW` 时，项目不需要 TensorFlow Header/共享库，服务器继续使用 MockBackend。

## 图片解码和安全限制

系统没有 OpenCV C++ 开发库，因此使用单头文件 `stb_image` 解码 JPEG/PNG。它只承担内存字节到 RGB 像素的转换。

解码前使用 `stbi_info_from_memory()` 检查：

- 宽高必须为正数。
- 单边不能超过 8192 像素。
- 总像素数不能超过 4000 万。

HTTP Body 的 10 MiB 上限不能防止压缩率极高的图片解码后耗尽内存，所以还必须限制像素数。

## YOLOv8 预处理

模型要求 `[1,640,640,3]` NHWC RGB float32。

预处理步骤：

1. JPEG/PNG 解码为 RGB。
2. 计算等比例缩放系数：

   ```text
   scale = min(640 / width, 640 / height)
   ```

3. 双线性缩放。
4. 居中填充，填充值为 RGB `(114,114,114)`。
5. 每个通道除以 255，转换为 `[0,1]` float32。
6. 保存原图尺寸、scale、padX、padY，供后处理恢复坐标。

预处理发生在推理工作线程，不占用 EventLoop。

## TensorFlowBackend 生命周期

加载模型：

```text
TF_NewGraph
→ TF_NewSessionOptions
→ TF_LoadSessionFromSavedModel(tags={serve})
→ 查找 serving_default_images
→ 查找 PartitionedCall
```

预热使用全 114 灰色的 `[1,640,640,3]` tensor 执行一次 SessionRun，提前完成图初始化和 oneDNN 优化。

推理时：

```text
创建 TF_Tensor
→ TF_SessionRun
→ 复制 output tensor
→ TF_DeleteTensor
```

TensorFlow 对象通过析构函数关闭 Session 并释放 Graph/Status，避免模型重载或服务器退出时泄漏资源。

当前一个 Backend 由两个调度线程共享。图片预处理和后处理可以并行；`TF_SessionRun` 使用 Backend 内部互斥锁串行执行。这是第一版偏保守的线程安全策略，后续可以通过实测决定使用并发 SessionRun 或 Session 池。

## YOLOv8 输出解码

输出内存按 `[channel, candidate]` 排列：

```text
channel 0..3:  center_x, center_y, width, height
channel 4..83: 80 个 COCO 类别分数
candidate:      8400 个候选框
```

对每个候选框选择分数最高的类别，低于请求 `threshold` 的候选直接丢弃。

模型坐标通过下面的逆变换恢复到原图：

```text
original_x = (model_x - padX) / scale
original_y = (model_y - padY) / scale
```

最终坐标裁剪到原图范围内。

## NMS

候选框按置信度从高到低排序。只有“类别相同且 IoU 大于 0.45”的候选框才会互相抑制，因此重叠位置上的 person 和 bus 可以同时保留。

输出统一为：

```json
{
  "class_id": 5,
  "label": "bus",
  "score": 0.87,
  "bbox": [22.1, 230.4, 804.8, 756.2]
}
```

## 模型和后端路由

调度器现在检查请求：

- `model=default` 会解析成当前加载的 `yolov8n`。
- 请求其他未加载模型返回 `422 MODEL_UNAVAILABLE`。
- `backend=auto` 或 `backend=tensorflow` 可以进入 TensorFlow Backend。
- `backend=tensorrt` 在 TensorRT 尚未接入时返回 `422 BACKEND_UNAVAILABLE`。

图片媒体类型正确但内容无法解码时返回 `422 IMAGE_DECODE_FAILED`，而不是误报服务器内部错误。

## 测试

本步骤新增：

### YoloV8ProcessorTest

- 从内存解码真实 PNG。
- 验证 NHWC tensor 大小。
- 构造重叠候选框。
- 验证同类 NMS 和跨类别保留。

### TensorFlowBackendTest

- 使用 TensorFlow C API 加载真实 SavedModel。
- 执行 warmup。
- 解码并推理 Ultralytics 官方 `bus.jpg`。
- 验证实际后端、原图尺寸和非空检测结果。

### HTTP 端到端

真实请求：

```bash
curl -X POST http://127.0.0.1:10000/v1/infer \
  -F model=yolov8n \
  -F backend=tensorflow \
  -F threshold=0.25 \
  -F 'image=@root/picture/ultralytics_bus.jpg;type=image/jpeg'
```

本机测试结果：

```text
HTTP 200
backend: tensorflow
image: 810 × 1080
detections: 5
labels: person, person, person, bus, person
preprocess: 约 28.9 ms
inference: 约 38.4 ms
postprocess: 约 1.0 ms
total: 约 68.5 ms
```

延迟只代表当前机器的一次功能验证，不能替代正式 benchmark。

## 本步骤边界

- 只支持 Batch 1 和固定 640×640 输入。
- 只支持 YOLOv8 COCO detection 输出 `[1,84,8400]`。
- 只实现 TensorFlow CPU。
- 模型配置暂时在服务器启动代码中构造，尚未实现 YAML/JSON 模型注册中心。
- TensorFlow SessionRun 暂时串行。

下一步是 TensorRT 后端，需要先确定本机 TensorRT 版本是否可用，并使用同一个预处理、后处理和 `InferenceResult` 协议。
