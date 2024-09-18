#pragma once

#include "TcpServer.h"
#include "noncopyable.h"
#include "Logging.h"
#include <string>

class HttpRequest;
class HttpResponse;

class HttpServer 
{
public:
    using HttpCallback = std::function<void (const HttpRequest&, HttpResponse*)>;

    HttpServer(EventLoop *loop,
            const InetAddress& listenAddr,
            const std::string& name);

    void setHttpCallback(const HttpCallback& cb)
    {
        httpCallback_ = cb;
    }
    
    void start();

private:
    void onConnection(const TcpConnectionPtr& conn);
    void onMessage(const TcpConnectionPtr &conn,
                    Buffer *buf,
                    Timestamp receiveTime);
    void onRequest(const TcpConnectionPtr&, const HttpRequest&);

    EventLoop *loop_;
    HttpCallback httpCallback_;
    TcpServer server_;
};

