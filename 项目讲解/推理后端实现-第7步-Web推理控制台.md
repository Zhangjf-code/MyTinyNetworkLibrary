# 推理后端实现：第 7 步——Web 推理控制台

## 目标

前六步已经完成 HTTP、异步调度、TensorFlow、TensorRT 和请求级路由，但使用方式仍以 curl 为主。本步骤把原来的图片回显演示页改造成可以直接操作 `/v1/infer` 的浏览器推理控制台。

前端不引入 Node.js 工程、打包器或第三方 JavaScript 框架，继续使用单文件：

```text
root/index4.html
```

浏览器打开服务首页即可使用。

## 页面功能

页面分为三个区域：

```text
检测画布 | 推理参数
-------------------
推理指标和检测明细
```

支持：

- 选择 JPEG/PNG 图片并立即本地预览。
- 输入模型名称，默认使用 `default`。
- 选择 `auto`、`tensorflow` 或 `tensorrt`。
- 使用滑块调整置信度阈值。
- 上传图片并执行异步推理。
- 在原图上绘制边界框、类别和置信度。
- 展示实际后端、模型版本、检测数量和耗时。
- 展示每个检测结果的类别、分数和坐标。
- 在窄屏设备上切换为单列布局。

原 `/upload` 图片回显接口仍然保留，但首页不再使用它。

## 请求构造

点击“开始推理”后，JavaScript 构造 `FormData`：

```javascript
form.append("model", model);
form.append("backend", backend);
form.append("threshold", threshold);
form.append("image", file);

fetch("/v1/infer", {
  method: "POST",
  body: form
});
```

不要手动设置 `Content-Type`。浏览器会自动生成包含 boundary 的 `multipart/form-data` Header，与第 1 步实现的 Multipart Parser 对接。

## 图片预览

图片选择后通过：

```javascript
URL.createObjectURL(file)
```

建立本地临时 URL，不需要先上传服务器即可预览。重新选择图片时释放旧 URL，避免页面长时间使用后积累 Blob 引用。

浏览器完成图片解码后，Canvas 的内部尺寸设置为图片自然尺寸：

```javascript
canvas.width = image.naturalWidth;
canvas.height = image.naturalHeight;
```

CSS 只负责缩放显示，因此后端返回的原图坐标可以直接用于 Canvas 绘制，不需要根据页面显示尺寸重新换算。

## 检测框绘制

每个检测结果包含：

```json
{
  "label": "person",
  "score": 0.89,
  "bbox": [x1, y1, x2, y2]
}
```

绘制顺序：

1. 重新绘制原始图片，清除上一次检测框。
2. 使用 `strokeRect` 绘制边框。
3. 使用半透明 `fillRect` 标记检测区域。
4. 在检测框顶部绘制类别和百分比置信度。

字体和线宽会根据原图宽度调整，避免高分辨率图片上的标签过小。

## 结果指标

页面展示响应中的：

- `backend`
- `model_version`
- `detections.length`
- `latency_ms.preprocess`
- `latency_ms.inference`
- `latency_ms.total`

响应中的实际 `backend` 比下拉框更重要。例如请求 `auto` 时，下拉框仍为 Auto，但结果会明确显示本次实际执行的是 `tensorrt` 或 `tensorflow`。

## 错误处理

前端同时检查 HTTP Status 和 JSON 的 `success`：

```text
response.ok && data.success
```

错误响应优先显示后端的 `error.message`，否则显示 HTTP 状态码。失败后会清除旧检测框和旧指标，避免用户把上一次成功结果误认为当前结果。

推理期间按钮进入 disabled 状态，阻止同一页面连续重复提交；请求完成或失败后恢复。

检测明细中的动态文本使用 `textContent` 和 DOM 节点构造，不把模型返回的标签直接拼接进 HTML，避免引入不必要的 HTML 注入风险。

## 首页路径修复

旧代码使用：

```text
../../root/index4.html
```

这要求服务器必须从特定工作目录启动，否则首页为空。

现在 CMake 注入绝对路径：

```text
DEFAULT_HTML_PATH=<project>/root/index4.html
```

因此可以从项目根目录、`src/http` 或其他工作目录启动同一个服务器二进制。

## 验证

本步骤完成以下检查：

- JavaScript 通过 Node.js `--check` 语法检查。
- 默认 MockBackend 配置编译成功。
- 从项目根目录启动服务器后，`GET /` 返回完整推理控制台。
- 首页包含 Canvas、后端选择、阈值和 `/v1/infer` 请求代码。
- 使用与浏览器一致的 multipart 字段提交图片，服务返回成功 JSON。

真实 TensorFlow/TensorRT 请求协议没有改变，继续使用第 6 步已经验证通过的同一个 `/v1/infer` 接口。

## 本步骤边界

- 前端模型输入框暂时由用户填写，没有调用模型列表 API。
- 只绘制目标检测框，不支持分割掩码或关键点。
- 页面刷新后不保留历史结果。
- 未实现多图片批量上传。
- 未实现推理进度流，只显示请求进行中状态。
- 页面仍嵌入 C++ 服务提供的单个 HTML 文件，没有静态资源缓存策略。
