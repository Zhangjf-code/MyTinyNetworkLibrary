#include "HttpContext.h"

#include <cassert>
#include <iostream>
#include <string>

namespace
{
void append(Buffer& buffer, const std::string& text)
{
    buffer.append(text.data(), text.size());
}

void testFragmentedRequest()
{
    HttpContext context;
    Buffer buffer(32);
    append(buffer, "POST /v1/infer?backend=tensorrt HTTP/1.1\r\nContent-Len");
    assert(context.parseRequest(&buffer, Timestamp(1)) == HttpContext::kIncomplete);

    append(buffer, "gth: 5\r\nContent-Type: application/octet-stream\r\n\r\n12");
    assert(context.parseRequest(&buffer, Timestamp(1)) == HttpContext::kIncomplete);

    append(buffer, "345");
    assert(context.parseRequest(&buffer, Timestamp(1)) == HttpContext::kComplete);
    assert(context.request().path() == "/v1/infer");
    assert(context.request().query() == "backend=tensorrt");
    assert(context.request().getbody() == "12345");
}

void testMultipartFileStaysInMemory()
{
    std::string body =
        "--demo\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "yolov8\r\n"
        "--demo\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n";
    body.append("binary\0image", 12);
    body.append("\r\n--demo--\r\n");

    const std::string request =
        "POST /upload HTTP/1.1\r\n"
        "Content-Type: multipart/form-data; boundary=demo\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;

    HttpContext context;
    Buffer buffer(32);
    append(buffer, request);
    assert(context.parseRequest(&buffer, Timestamp(1)) == HttpContext::kComplete);
    assert(context.request().getBodyForm("model") == "yolov8");
    assert(context.request().files().size() == 1);
    const HttpRequest::UploadedFile& file = context.request().files().front();
    assert(file.fieldName == "image");
    assert(file.fileName == "a.jpg");
    assert(file.contentType == "image/jpeg");
    assert(file.content.size() == 12);
}

void testBodyLimit()
{
    HttpContext context;
    Buffer buffer(64);
    append(buffer,
        "POST /upload HTTP/1.1\r\nContent-Length: 10485761\r\n\r\n");
    assert(context.parseRequest(&buffer, Timestamp(1)) == HttpContext::kRequestTooLarge);
}

void testPipelinedRequestsRemainBuffered()
{
    HttpContext context;
    Buffer buffer(64);
    append(buffer,
        "GET /first HTTP/1.1\r\n\r\n"
        "GET /second HTTP/1.1\r\n\r\n");
    assert(context.parseRequest(&buffer, Timestamp(1)) == HttpContext::kComplete);
    assert(context.request().path() == "/first");
    context.reset();
    assert(context.parseRequest(&buffer, Timestamp(2)) == HttpContext::kComplete);
    assert(context.request().path() == "/second");
}
}

int main()
{
    testFragmentedRequest();
    testMultipartFileStaysInMemory();
    testBodyLimit();
    testPipelinedRequestsRemainBuffered();
    std::cout << "HttpContext tests passed" << std::endl;
    return 0;
}
