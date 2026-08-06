#include "HttpServer.h"
#include "HttpRequest.h"
#include "HttpResponse.h"
#include "HttpContext.h"

#include <memory>
#include <iostream>

/**
 * 默认的http回调函数
 * 设置响应状态码，响应信息并关闭连接
 */
void defaultHttpCallback(const HttpRequest&, HttpResponse* resp)
{
    resp->setStatusCode(HttpResponse::k404NotFound);
    resp->setStatusMessage("Not Found");
    resp->setCloseConnection(true);
}

HttpServer::HttpServer(EventLoop *loop,
                      const InetAddress &listenAddr,
                      const std::string &name)
  : server_(loop, listenAddr, name),
    httpCallback_(defaultHttpCallback),
    loop_(loop)
{
    server_.setConnectionCallback(
        std::bind(&HttpServer::onConnection, this, std::placeholders::_1));
    server_.setMessageCallback(
        std::bind(&HttpServer::onMessage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    server_.setThreadNum(4);
}

void HttpServer::start()
{
    server_.start();
}

void HttpServer::onConnection(const TcpConnectionPtr& conn)
{
    if (conn->connected())
    {
        conn->setApplicationContext(std::make_shared<HttpContext>());
    }
    else 
    {
        conn->setApplicationContext(std::shared_ptr<void>());
    }
}

// 有消息到来时的业务处理
void HttpServer::onMessage(const TcpConnectionPtr& conn,
                           Buffer* buf,
                           Timestamp receiveTime)
{
    // LOG_INFO << "HttpServer::onMessage";
    std::shared_ptr<HttpContext> context =
        std::static_pointer_cast<HttpContext>(conn->getApplicationContext());
    if (!context)
    {
        context = std::make_shared<HttpContext>();
        conn->setApplicationContext(context);
    }

#if 0
    // 打印请求报文
    std::string request = buf->GetBufferAllAsString();
    std::cout << request << std::endl;
#endif

    // 进行状态机解析
    // 错误则发送 BAD REQUEST 半关闭

    while (conn->connected())
    {
        const HttpContext::ParseResult result = context->parseRequest(buf, receiveTime);
        if (result == HttpContext::kIncomplete) return;
        if (result == HttpContext::kBadRequest)
        {
            conn->send("HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
            conn->shutdown();
            return;
        }
        if (result == HttpContext::kRequestTooLarge)
        {
            conn->send("HTTP/1.1 413 Payload Too Large\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
            conn->shutdown();
            return;
        }

        const bool stopParsing = onRequest(conn, context->request());
        context->reset();
        if (stopParsing) return;
        if (buf->readableBytes() == 0) return;
    }
}

bool HttpServer::onRequest(const TcpConnectionPtr& conn, const HttpRequest& req)
{
    if (asyncHttpCallback_ && asyncHttpCallback_(conn, req)) return true;

    const std::string& connection = req.getHeader("Connection");

    // 判断长连接还是短连接
    bool close = connection == "close" ||
        (req.version() == HttpRequest::kHttp10 && connection != "Keep-Alive");
    // 响应信息
    HttpResponse response(close);

    // httpCallback_ 由用户传入，怎么写响应体由用户决定
    // 此处初始化了一些response的信息，比如响应码，回复OK
    httpCallback_(req, &response);
    Buffer buf;
    response.appendToBuffer(&buf);
    // TODO:需要重载 TcpConnection::send 使其可以接收一个缓冲区
    conn->send(buf.retrieveAllAsString());
    if (response.closeConnection())
    {
        conn->shutdown();
    }
    return response.closeConnection();
}
