#ifndef HTTP_HTTPRESPONSE_H
#define HTTP_HTTPRESPONSE_H
#include <string.h>
#include <string>
#include <unordered_map>

class Buffer;
class HttpResponse
{
public:
    // 响应状态码
    enum HttpStatusCode
    {
        kUnknown,
        k200Ok = 200,
        k301MovedPermanently = 301,
        k400BadRequest = 400,
        k405MethodNotAllowed = 405,
        k413PayloadTooLarge = 413,
        k415UnsupportedMediaType = 415,
        k422UnprocessableEntity = 422,
        k500InternalServerError = 500,
        k501NotImplemented = 501,
        k503ServiceUnavailable = 503,
        k404NotFound = 404,
    };  

    explicit HttpResponse(bool close)
      : statusCode_(kUnknown),
        closeConnection_(close)
    {
    }   

    void setStatusCode(HttpStatusCode code)
    { statusCode_ = code; } 

    void setStatusMessage(const std::string& message)
    { statusMessage_ = message; }   

    void setCloseConnection(bool on)
    { closeConnection_ = on; }  

    bool closeConnection() const
    { return closeConnection_; }  

    HttpStatusCode statusCode() const { return statusCode_; }

    const std::string& body() const { return body_; }

    std::string getHeader(const std::string& key) const
    {
        const auto it = headers_.find(key);
        return it == headers_.end() ? std::string() : it->second;
    }

    void setContentType(const std::string& contentType)
    { addHeader("Content-Type", contentType); } 

    void addHeader(const std::string& key, const std::string& value)
    { headers_[key] = value; }  

    void addothermessage(const std::string& key, const std::string& value){
        othermessage[key] = value;
    }

    void setBody(const std::string& body)
    { body_ = std::move(body); }

    void setBody(const char* body, size_t len)
    { body_.assign(body, len); } 

    void appendToBuffer(Buffer* output) const;

private:
    std::unordered_map<std::string, std::string> headers_;
    std::unordered_map<std::string, std::string> othermessage;
    HttpStatusCode statusCode_;
    // FIXME: add http version
    std::string statusMessage_;
    bool closeConnection_;
    std::string body_;
};

#endif // HTTP_HTTPRESPONSE_H
