#pragma once

#include "noncopyable.h"
#include "Timestamp.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>


class HttpRequest
{
public:
    struct UploadedFile
    {
        std::string fieldName;
        std::string fileName;
        std::string contentType;
        std::string content;
    };

    enum Method { kInvalid, kGet, kPost, kHead, kPut, kDelete };
    enum Version { kUnknown, kHttp10, kHttp11 };

    HttpRequest()
        : method_(kInvalid),
          version_(kUnknown)
    {        
    }

    void setVersion(Version v)
    {
        version_ = v;
    }

    Version version() const  { return version_; }

    bool setMethod(const char *start, const char *end)
    {
        std::string m(start, end);
        if (m == "GET")
        {
            method_ = kGet;
        }
        else if (m == "POST")
        {
            method_ = kPost;
        }
        else if (m == "HEAD")
        {
            method_ = kHead;
        }
        else if (m == "PUT")
        {
          method_ = kPut;
        }
        else if (m == "DELETE")
        {
          method_ = kDelete;
        }
        else
        {
          method_ = kInvalid;
        }
        // 判断method_是否合法
        return method_ != kInvalid;
    }

    Method method() const { return method_; }

    const char* methodString() const
    {
        const char* result = "UNKNOWN";
        switch(method_)
        {
          case kGet:
            result = "GET";
            break;
          case kPost:
            result = "POST";
            break;
          case kHead:
            result = "HEAD";
            break;
          case kPut:
            result = "PUT";
            break;
          case kDelete:
            result = "DELETE";
            break;
          default:
            break;
        }
        return result;
    } 

    void setPath(const char *start, const char *end)
    {
        path_.assign(start, end);
    }

    const std::string& path() const { return path_; }

    void setQuery(const char *start, const char *end) 
    {
        query_.assign(start, end);
    }

    const std::string& query() const { return query_; }

    void setReceiveTime(Timestamp t) 
    { 
        receiveTime_ = t; 
    }

    Timestamp receiveTime() const { return receiveTime_; }

    void addHeader(const char *start, const char *colon, const char *end)
    {
        std::string field(start, colon);
        std::transform(field.begin(), field.end(), field.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        ++colon;
        // 跳过空格
        while (colon < end && isspace(*colon))
        {
            ++colon;
        }
        std::string value(colon, end);
        // value丢掉后面的空格，通过重新截断大小设置
        while (!value.empty() && isspace(value[value.size()-1]))
        {
          value.resize(value.size()-1);
        }
        headers_[field] = value;
    }

    void addBodyForm(const std::string &field, const std::string &value)
    {
        bodyform_[field] = value;
    }


    // 获取请求头部的对应值
    std::string getHeader(const std::string &field) const
    {
        std::string result;
        std::string normalized(field);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        auto it = headers_.find(normalized);
        if (it != headers_.end())
        {
            result = it->second;
        }
        return result;
    }


    std::string getBodyForm(const std::string &field) const
    {
        std::string result;
        auto it = bodyform_.find(field);
        if (it != bodyform_.end())
        {
            result = it->second;
        }
        return result;
    }

    const std::unordered_map<std::string, std::string>& headers() const
    {
        return headers_;
    }


    const std::unordered_map<std::string, std::string>& bodyfrom() const
    {
        return bodyform_;
    }

    void addFile(UploadedFile file) { files_.emplace_back(std::move(file)); }

    const std::vector<UploadedFile>& files() const { return files_; }


    void swap(HttpRequest &rhs)
    {
        std::swap(method_, rhs.method_);
        std::swap(version_, rhs.version_);
        path_.swap(rhs.path_);
        query_.swap(rhs.query_);
        std::swap(receiveTime_, rhs.receiveTime_);
        headers_.swap(rhs.headers_);
        bodyform_.swap(rhs.bodyform_);
        files_.swap(rhs.files_);
        body_.swap(rhs.body_);
    }

    void setBody(const std::string &body)
    {
        body_ = body;
    }

    const std::string& getbody() const
    {
        return body_;
    }
    
private:
    Method method_;         // 请求方法
    Version version_;       // 协议版本号
    std::string path_;      // 请求路径
    std::string query_;     // 询问参数
    Timestamp receiveTime_; // 请求时间
    std::unordered_map<std::string, std::string> headers_; // 请求头部列表
    std::unordered_map<std::string, std::string> bodyform_; // body 表单
    std::vector<UploadedFile> files_; // multipart上传文件（保存在内存中）
    std::string body_;
};
