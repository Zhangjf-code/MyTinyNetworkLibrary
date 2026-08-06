# 推理后端实现：第 1 步——HTTP 基础改造

## 目标

推理任务通常需要几十毫秒甚至数秒，HTTP 请求也可能被拆分成多个 TCP 包。进入推理层之前，网络层必须满足以下条件：

1. 一个连接的 HTTP 解析状态可以跨多次可读事件保存。
2. 请求体不完整时立即返回事件循环，等待下一次 EPOLLIN，而不是主动阻塞读取 socket。
3. 上传图片保存在内存中，避免网络线程执行文件 I/O。
4. 请求大小有明确上限，防止单个连接无限占用内存。
5. 其他线程发送响应时，待发送字符串的生命周期安全。

本步骤设定请求头上限为 16 KiB、请求体上限为 10 MiB。

## 连接级 HTTP 上下文

旧实现每次进入 `HttpServer::onMessage()` 都创建新的 `HttpContext`。如果一个请求被拆成两次到达，第二次解析时已经丢失第一次的状态和请求头。

现在 `TcpConnection` 提供通用的 `applicationContext_`：

```cpp
std::shared_ptr<void> applicationContext_;
```

HTTP 连接建立时创建一个 `HttpContext` 并放入其中，连接关闭时释放。一次连接的关系变成：

```text
TcpConnection
    └── HttpContext
          ├── 当前解析状态
          ├── HttpRequest
          ├── 已解析请求头字节数
          └── Content-Length
```

`TcpConnection` 仍然不依赖 HTTP 类型，因此网络核心层未来也可以承载其他上层协议。

## 增量解析

`HttpContext::parseRequest()` 不再只返回成功或失败，而是返回四种结果：

```cpp
enum ParseResult {
    kIncomplete,
    kComplete,
    kBadRequest,
    kRequestTooLarge,
};
```

状态机流程如下：

```text
kExpectRequestLine
        │ 收到完整 CRLF
        ▼
kExpectHeaders
        │ 收到空行
        ▼
kExpectBody
        │ Buffer 中达到 Content-Length
        ▼
kGotAll
```

如果请求行、某个请求头或者 Body 尚未收齐，解析器返回 `kIncomplete`，已收到的数据继续留在连接的输入 Buffer 中。后续收到数据后从原状态继续解析。

解析器不会在 HTTP 回调里再次调用 `read()`。socket 读取仍然只发生在 `TcpConnection::handleRead()`，从而保持 Reactor 的非阻塞职责边界。

完整请求处理后会重置同一个 `HttpContext`。如果 Buffer 中还有数据，服务器继续解析，因此同一连接可以处理连续或流水线到达的请求。

## multipart 图片上传

multipart 解析器现在是 `HttpContext` 的一部分，不再依赖仓库中缺失的 `FormDataParser.h`。

普通表单字段保存到：

```cpp
std::unordered_map<std::string, std::string> bodyform_;
```

文件字段保存为：

```cpp
struct UploadedFile {
    std::string fieldName;
    std::string fileName;
    std::string contentType;
    std::string content;
};
```

其中 `content` 是二进制安全的 `std::string`，允许包含 `\0`。解析过程不会根据客户端提供的文件名创建文件，也不会访问磁盘。后续推理层可以直接把 `content.data()` 和 `content.size()` 交给图片解码器。

示例业务中的 `/upload` 暂时将收到的第一个文件原样返回，用于验证上传内容没有被改变。

## 请求大小和错误响应

解析过程中持续统计请求头大小：

- 请求头超过 16 KiB：返回 `413 Payload Too Large` 并关闭连接。
- `Content-Length` 不是合法非负整数：返回 `400 Bad Request`。
- Body 超过 10 MiB：返回 `413 Payload Too Large` 并关闭连接。
- multipart 格式或 boundary 非法：返回 `400 Bad Request`。

采用有界请求是后续设置有界推理队列的前提，否则即使推理队列有限，等待解析的图片仍可能耗尽进程内存。

## 请求头名称

HTTP 请求头名称不区分大小写。写入 `HttpRequest` 时统一转换为小写，调用 `getHeader()` 时也会规范化查询名称。因此 `Content-Length`、`content-length` 和其他大小写组合会得到相同结果。

## 跨线程安全发送

旧版 `TcpConnection::send()` 在非 I/O 线程中把 `buf.c_str()` 指针绑定进延迟回调：

```text
调用 send()
→ 保存临时字符串内部指针
→ send() 返回，字符串可能析构
→ EventLoop 稍后使用悬空指针
```

新实现按值捕获字符串，并同时持有连接的 `shared_ptr`：

```cpp
TcpConnectionPtr self(shared_from_this());
loop_->runInLoop([self, buf]() {
    self->sendInLoop(buf);
});
```

直到 I/O 线程执行完发送任务，连接和待发送数据都保持有效。这是后续“推理线程完成任务，再把 JSON 响应投递回 EventLoop”的基础。

## 测试

`src/http/test/HttpContextTest.cc` 覆盖：

1. 请求行、请求头和 Body 分多次到达。
2. multipart 普通字段和含 `\0` 的二进制文件。
3. 10 MiB Body 上限。
4. 同一个 Buffer 内的连续 HTTP 请求。

执行方式：

```bash
cmake -S . -B build
cmake --build build --target HttpContextTest
ctest --test-dir build --output-on-failure -R HttpContextTest
```

## 本步骤边界

本步骤只建立可靠的 HTTP 数据通道，还没有加入 JSON API、推理线程池或真实模型。下一步将在此基础上定义统一推理请求、结果结构和 HTTP JSON 协议。
