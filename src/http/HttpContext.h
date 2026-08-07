#pragma once

#include "HttpRequest.h"
#include "Buffer.h"

class HttpContext
{
public:
    enum ParseResult
    {
        kIncomplete,
        kComplete,
        kBadRequest,
        kRequestTooLarge,
    };

    static const size_t kMaxHeaderBytes = 16 * 1024;
    static const size_t kMaxBodyBytes = 10 * 1024 * 1024;

    // HTTP请求状态
    enum HttpRequestParseState
    {
        kExpectRequestLine, // 解析请求行状态
        kExpectHeaders,     // 解析请求头部状态
        kExpectBody,        // 解析请求体状态
        kGotAll,            // 解析完毕状态
    };

    HttpContext()
        : state_(kExpectRequestLine)
    {
    }

    ParseResult parseRequest(Buffer* buf, Timestamp receiveTime);

    bool gotAll() const { return state_ == kGotAll; }

    // 重置HttpContext状态，异常安全
    void reset()
    {
        state_ = kExpectRequestLine;
        headerBytes_ = 0;
        contentLength_ = 0;
        /**
         * 构造一个临时空HttpRequest对象，和当前的成员HttpRequest对象交换置空
         * 然后临时对象析构
         */
        HttpRequest dummy;
        request_.swap(dummy);
    }

    const HttpRequest& request() const { return request_; }

    HttpRequest& request() { return request_; }

private:
    bool processRequestLine(const char *begin, const char *end);
    bool prepareBody();
    bool parseMultipartBody();

    HttpRequestParseState state_;
    HttpRequest request_;
    size_t headerBytes_ = 0;
    size_t contentLength_ = 0;
};
