#include "HttpContext.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace
{
std::string trim(const std::string& value)
{
    const std::string whitespace(" \t\r\n");
    const size_t first = value.find_first_not_of(whitespace);
    if (first == std::string::npos) return std::string();
    const size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

std::string parameterValue(const std::string& value, const std::string& name)
{
    const std::string key = name + "=";
    size_t pos = value.find(key);
    if (pos == std::string::npos) return std::string();
    pos += key.size();
    if (pos < value.size() && value[pos] == '"')
    {
        const size_t end = value.find('"', pos + 1);
        return end == std::string::npos ? std::string() : value.substr(pos + 1, end - pos - 1);
    }
    const size_t end = value.find(';', pos);
    return trim(value.substr(pos, end == std::string::npos ? end : end - pos));
}
}

bool HttpContext::processRequestLine(const char *begin, const char *end)
{
    const char *start = begin;
    const char *space = std::find(start, end, ' ');
    if (space == end || !request_.setMethod(start, space)) return false;

    start = space + 1;
    space = std::find(start, end, ' ');
    if (space == end) return false;

    const char* question = std::find(start, space, '?');
    request_.setPath(start, question);
    if (question != space) request_.setQuery(question + 1, space);

    start = space + 1;
    if (end - start != 8 || !std::equal(start, end - 1, "HTTP/1.")) return false;
    if (*(end - 1) == '1') request_.setVersion(HttpRequest::kHttp11);
    else if (*(end - 1) == '0') request_.setVersion(HttpRequest::kHttp10);
    else return false;
    return true;
}

bool HttpContext::prepareBody()
{
    const std::string value = request_.getHeader("Content-Length");
    if (value.empty())
    {
        contentLength_ = 0;
        return true;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) return false;
    contentLength_ = static_cast<size_t>(parsed);
    return true;
}

bool HttpContext::parseMultipartBody()
{
    const std::string contentType = request_.getHeader("Content-Type");
    if (contentType.find("multipart/form-data") == std::string::npos) return true;

    const std::string boundaryValue = parameterValue(contentType, "boundary");
    if (boundaryValue.empty()) return false;
    const std::string boundary = "--" + boundaryValue;
    const std::string& body = request_.getbody();
    size_t cursor = 0;

    while (true)
    {
        const size_t marker = body.find(boundary, cursor);
        if (marker == std::string::npos) return false;
        size_t partStart = marker + boundary.size();
        if (body.compare(partStart, 2, "--") == 0) return true;
        if (body.compare(partStart, 2, "\r\n") != 0) return false;
        partStart += 2;

        const size_t headerEnd = body.find("\r\n\r\n", partStart);
        if (headerEnd == std::string::npos) return false;
        const size_t nextMarker = body.find("\r\n" + boundary, headerEnd + 4);
        if (nextMarker == std::string::npos) return false;

        std::string disposition;
        std::string partContentType;
        size_t lineStart = partStart;
        while (lineStart < headerEnd)
        {
            size_t lineEnd = body.find("\r\n", lineStart);
            if (lineEnd == std::string::npos || lineEnd > headerEnd) lineEnd = headerEnd;
            const std::string line = body.substr(lineStart, lineEnd - lineStart);
            const size_t colon = line.find(':');
            if (colon == std::string::npos) return false;
            std::string key = line.substr(0, colon);
            std::transform(key.begin(), key.end(), key.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            const std::string value = trim(line.substr(colon + 1));
            if (key == "content-disposition") disposition = value;
            else if (key == "content-type") partContentType = value;
            lineStart = lineEnd + 2;
        }

        if (disposition.find("form-data") == std::string::npos) return false;
        const std::string fieldName = parameterValue(disposition, "name");
        const std::string fileName = parameterValue(disposition, "filename");
        if (fieldName.empty()) return false;
        const std::string content = body.substr(headerEnd + 4, nextMarker - (headerEnd + 4));
        if (fileName.empty())
        {
            request_.addBodyForm(fieldName, content);
        }
        else
        {
            HttpRequest::UploadedFile file;
            file.fieldName = fieldName;
            file.fileName = fileName;
            file.contentType = partContentType;
            file.content = content;
            request_.addFile(std::move(file));
        }
        cursor = nextMarker + 2;
    }
}

HttpContext::ParseResult HttpContext::parseRequest(Buffer* buf, Timestamp receiveTime)
{
    while (true)
    {
        if (state_ == kExpectRequestLine || state_ == kExpectHeaders)
        {
            const char* crlf = buf->findCRLF();
            if (!crlf)
            {
                return headerBytes_ + buf->readableBytes() > kMaxHeaderBytes
                    ? kRequestTooLarge : kIncomplete;
            }

            const size_t lineBytes = static_cast<size_t>(crlf + 2 - buf->peek());
            headerBytes_ += lineBytes;
            if (headerBytes_ > kMaxHeaderBytes) return kRequestTooLarge;

            if (state_ == kExpectRequestLine)
            {
                if (!processRequestLine(buf->peek(), crlf)) return kBadRequest;
                request_.setReceiveTime(receiveTime);
                state_ = kExpectHeaders;
            }
            else
            {
                const char* colon = std::find(buf->peek(), crlf, ':');
                if (colon == crlf)
                {
                    if (crlf != buf->peek()) return kBadRequest;
                    if (!prepareBody()) return kBadRequest;
                    if (contentLength_ > kMaxBodyBytes) return kRequestTooLarge;
                    state_ = kExpectBody;
                }
                else
                {
                    request_.addHeader(buf->peek(), colon, crlf);
                }
            }
            buf->retrieveUntil(crlf + 2);
        }
        else if (state_ == kExpectBody)
        {
            if (buf->readableBytes() < contentLength_) return kIncomplete;
            request_.setBody(buf->retrieveAsString(contentLength_));
            if (!parseMultipartBody()) return kBadRequest;
            state_ = kGotAll;
            return kComplete;
        }
        else
        {
            return kComplete;
        }
    }
}
