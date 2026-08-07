#include "HttpServer.h"
#include "HttpRequest.h"
#include "HttpResponse.h"
#include "HttpContext.h"
#include "Timestamp.h"
#include "InferenceController.h"
#include "InferenceScheduler.h"
#include "ModelRegistry.h"
#include "ManagementController.h"
#include "MockBackend.h"
#ifdef MYTINYMUDUO_ENABLE_TENSORFLOW
#include "TensorFlowBackend.h"
#endif
#ifdef MYTINYMUDUO_ENABLE_TENSORRT
#include "TensorRTBackend.h"
#endif
#include <fcntl.h>    // open
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // fstat
#include <unistd.h>   // close
#include <signal.h>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <thread>

extern char favicon[555];
bool benchmark = false;
InferenceController* inferenceController = nullptr;
ManagementController* managementController = nullptr;
const char* htmlFilePath = DEFAULT_HTML_PATH;

namespace
{
std::string environmentOrDefault(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

bool positiveEnvironment(const char* name, size_t fallback, size_t maximum,
                         size_t* result, std::string* error)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
    {
        *result = fallback;
        return true;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno || end == value || *end != '\0' || parsed == 0 || parsed > maximum)
    {
        if (error) *error = std::string(name) + " must be an integer between 1 and " +
                            std::to_string(maximum);
        return false;
    }
    *result = static_cast<size_t>(parsed);
    return true;
}
}

//打开.html文件并将其作为body
std::string read_file_to_string(const char* filename) {
    // 打开文件
    std::string err = "";
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("打开文件失败");
        return "";
    }

    // 获取文件大小
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("获取文件信息失败");
        close(fd);
        return "";
    }

    // 映射文件到内存
    char* mapped = static_cast<char*>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (mapped == MAP_FAILED) {
        perror("内存映射失败");
        close(fd);
        return "";
    }

    // 将映射的内容复制到 std::string
    std::string content(mapped, sb.st_size);

    // 解除映射
    if (munmap(mapped, sb.st_size) == -1) {
        perror("解除映射失败");
    }

    // 关闭文件
    close(fd);

    return content;
}



void onRequest(const HttpRequest& req, HttpResponse* resp)
{
    std::cout << "Headers " << req.method()<<req.methodString() << " " << req.path() << std::endl;
    // 打印头部
    if (!benchmark)
    {
        const std::unordered_map<std::string, std::string>& headers = req.headers();
        for (const auto& header : headers)
        {
            std::cout << header.first << ": " << header.second << std::endl;
        }
        for(const auto& file : req.files()){
            std::cout << "filename:" << file.fileName
                      << " bytes:" << file.content.size() << std::endl;
        }
    }
    std::cout<< "Path:" << req.path() << std::endl;
    if (req.path() == "/")
    {
        resp->setStatusCode(HttpResponse::k200Ok);
        resp->setStatusMessage("OK");
        resp->setContentType("text/html");
        resp->addHeader("Server", "mytinymuduo");
        std::string now = Timestamp::now().toFormattedString();
        resp->setBody(read_file_to_string(htmlFilePath));
    }

    else if (req.path() == "/favicon.ico")
    {
        resp->setStatusCode(HttpResponse::k200Ok);
        resp->setStatusMessage("OK");
        resp->setContentType("image/png");
        resp->setBody(std::string(favicon, sizeof favicon));
    }
    else if (req.path() == "/v1/infer")
    {
        inferenceController->handle(req, resp);
    }
    else if (req.path() == "/healthz")
    {
        managementController->handleHealth(req, resp);
    }
    else if (req.path() == "/v1/models")
    {
        managementController->handleModels(req, resp);
    }
    else if (req.path() == "/metrics")
    {
        managementController->handleMetrics(req, resp);
    }
    else if (req.path() == "/upload" )
    {
        if (req.files().empty())
        {
            resp->setStatusCode(HttpResponse::k400BadRequest);
            resp->setStatusMessage("Bad Request");
            resp->setCloseConnection(true);
            return;
        }
        resp->setStatusCode(HttpResponse::k200Ok);
        resp->setStatusMessage("OK");
        const HttpRequest::UploadedFile& file = req.files().front();
        resp->setContentType(file.contentType.empty() ? "application/octet-stream" : file.contentType);
        resp->addHeader("Server", "mytinymuduo");
        resp->setBody(file.content);
    }
    else
    {
        resp->setStatusCode(HttpResponse::k404NotFound);
        resp->setStatusMessage("Not Found");
        resp->setCloseConnection(true);
    }
}

int main(int argc, char* argv[])
{
    sigset_t shutdownSignals;
    sigemptyset(&shutdownSignals);
    sigaddset(&shutdownSignals, SIGINT);
    sigaddset(&shutdownSignals, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &shutdownSignals, nullptr) != 0)
    {
        std::cerr << "Failed to block shutdown signals" << std::endl;
        return 1;
    }

    std::string configError;
    size_t port = 0;
    size_t workerCount = 0;
    size_t queueCapacity = 0;
    if (!positiveEnvironment("MYWEBSERVER_PORT", 10000, 65535, &port, &configError) ||
        !positiveEnvironment("MYWEBSERVER_INFERENCE_WORKERS", 2, 1024,
                             &workerCount, &configError) ||
        !positiveEnvironment("MYWEBSERVER_QUEUE_CAPACITY", 128, 1000000,
                             &queueCapacity, &configError))
    {
        std::cerr << "Invalid configuration: " << configError << std::endl;
        return 1;
    }
    const std::string registryPath = environmentOrDefault(
        "MYWEBSERVER_MODEL_REGISTRY", DEFAULT_MODEL_REGISTRY_PATH);
    const char* configuredHtmlPath = std::getenv("MYWEBSERVER_HTML_PATH");
    htmlFilePath = configuredHtmlPath && *configuredHtmlPath
        ? configuredHtmlPath : DEFAULT_HTML_PATH;

    EventLoop loop;
    std::shared_ptr<ModelRegistry> registry(new ModelRegistry);
#if defined(MYTINYMUDUO_ENABLE_TENSORFLOW) || defined(MYTINYMUDUO_ENABLE_TENSORRT)
    std::string registryError;
    if (!registry->loadFromFile(registryPath, &registryError))
    {
        std::cerr << "Failed to load model registry: " << registryError << std::endl;
        return 1;
    }
#else
    std::shared_ptr<IInferenceBackend> backend(new MockBackend(std::chrono::milliseconds(20)));
    ModelConfig modelConfig;
    modelConfig.name = "mock";
    modelConfig.version = "mock-1";
    std::string registryError;
    if (!registry->add(modelConfig, backend, 0, &registryError))
    {
        std::cerr << "Failed to register Mock backend: " << registryError << std::endl;
        return 1;
    }
#endif
    std::shared_ptr<InferenceMetrics> metrics(new InferenceMetrics);
    InferenceScheduler scheduler(registry, workerCount, queueCapacity, metrics);
    std::string schedulerError;
    if (!scheduler.start(&schedulerError))
    {
        std::cerr << "Failed to start inference scheduler: " << schedulerError << std::endl;
        return 1;
    }
    for (const ModelRegistry::Registration& registration : registry->registrations())
    {
        std::cout << "Model " << registration.config.name
                  << " version=" << registration.config.version
                  << " backend=" << backendTypeName(registration.config.backend)
                  << " status=" << (registration.available ? "ready" : "unavailable");
        if (!registration.loadError.empty()) std::cout << " reason=" << registration.loadError;
        std::cout << std::endl;
    }
    InferenceController controller(&scheduler, metrics);
    ManagementController management(registry, metrics);
    inferenceController = &controller;
    managementController = &management;
    std::cout << "HTTP configuration port=" << port
              << " workers=" << workerCount
              << " queue_capacity=" << queueCapacity
              << " registry=" << registryPath << std::endl;
    HttpServer server(&loop, InetAddress(static_cast<uint16_t>(port)), "http-server");
    server.setHttpCallback(onRequest);
    server.setAsyncHttpCallback(
        [&controller](const TcpConnectionPtr& connection, const HttpRequest& request) {
            return controller.handleAsync(connection, request);
        });
    std::thread signalThread([&loop, &shutdownSignals]() {
        int signalNumber = 0;
        if (sigwait(&shutdownSignals, &signalNumber) == 0)
        {
            std::cout << "Received signal " << signalNumber
                      << ", shutting down" << std::endl;
            loop.quit();
        }
    });
    server.start();
    loop.loop();
    scheduler.stop();
    signalThread.join();
}

char favicon[555] = {
  '\x89', 'P', 'N', 'G', '\xD', '\xA', '\x1A', '\xA',
  '\x0', '\x0', '\x0', '\xD', 'I', 'H', 'D', 'R',
  '\x0', '\x0', '\x0', '\x10', '\x0', '\x0', '\x0', '\x10',
  '\x8', '\x6', '\x0', '\x0', '\x0', '\x1F', '\xF3', '\xFF',
  'a', '\x0', '\x0', '\x0', '\x19', 't', 'E', 'X',
  't', 'S', 'o', 'f', 't', 'w', 'a', 'r',
  'e', '\x0', 'A', 'd', 'o', 'b', 'e', '\x20',
  'I', 'm', 'a', 'g', 'e', 'R', 'e', 'a',
  'd', 'y', 'q', '\xC9', 'e', '\x3C', '\x0', '\x0',
  '\x1', '\xCD', 'I', 'D', 'A', 'T', 'x', '\xDA',
  '\x94', '\x93', '9', 'H', '\x3', 'A', '\x14', '\x86',
  '\xFF', '\x5D', 'b', '\xA7', '\x4', 'R', '\xC4', 'm',
  '\x22', '\x1E', '\xA0', 'F', '\x24', '\x8', '\x16', '\x16',
  'v', '\xA', '6', '\xBA', 'J', '\x9A', '\x80', '\x8',
  'A', '\xB4', 'q', '\x85', 'X', '\x89', 'G', '\xB0',
  'I', '\xA9', 'Q', '\x24', '\xCD', '\xA6', '\x8', '\xA4',
  'H', 'c', '\x91', 'B', '\xB', '\xAF', 'V', '\xC1',
  'F', '\xB4', '\x15', '\xCF', '\x22', 'X', '\x98', '\xB',
  'T', 'H', '\x8A', 'd', '\x93', '\x8D', '\xFB', 'F',
  'g', '\xC9', '\x1A', '\x14', '\x7D', '\xF0', 'f', 'v',
  'f', '\xDF', '\x7C', '\xEF', '\xE7', 'g', 'F', '\xA8',
  '\xD5', 'j', 'H', '\x24', '\x12', '\x2A', '\x0', '\x5',
  '\xBF', 'G', '\xD4', '\xEF', '\xF7', '\x2F', '6', '\xEC',
  '\x12', '\x20', '\x1E', '\x8F', '\xD7', '\xAA', '\xD5', '\xEA',
  '\xAF', 'I', '5', 'F', '\xAA', 'T', '\x5F', '\x9F',
  '\x22', 'A', '\x2A', '\x95', '\xA', '\x83', '\xE5', 'r',
  '9', 'd', '\xB3', 'Y', '\x96', '\x99', 'L', '\x6',
  '\xE9', 't', '\x9A', '\x25', '\x85', '\x2C', '\xCB', 'T',
  '\xA7', '\xC4', 'b', '1', '\xB5', '\x5E', '\x0', '\x3',
  'h', '\x9A', '\xC6', '\x16', '\x82', '\x20', 'X', 'R',
  '\x14', 'E', '6', 'S', '\x94', '\xCB', 'e', 'x',
  '\xBD', '\x5E', '\xAA', 'U', 'T', '\x23', 'L', '\xC0',
  '\xE0', '\xE2', '\xC1', '\x8F', '\x0', '\x9E', '\xBC', '\x9',
  'A', '\x7C', '\x3E', '\x1F', '\x83', 'D', '\x22', '\x11',
  '\xD5', 'T', '\x40', '\x3F', '8', '\x80', 'w', '\xE5',
  '3', '\x7', '\xB8', '\x5C', '\x2E', 'H', '\x92', '\x4',
  '\x87', '\xC3', '\x81', '\x40', '\x20', '\x40', 'g', '\x98',
  '\xE9', '6', '\x1A', '\xA6', 'g', '\x15', '\x4', '\xE3',
  '\xD7', '\xC8', '\xBD', '\x15', '\xE1', 'i', '\xB7', 'C',
  '\xAB', '\xEA', 'x', '\x2F', 'j', 'X', '\x92', '\xBB',
  '\x18', '\x20', '\x9F', '\xCF', '3', '\xC3', '\xB8', '\xE9',
  'N', '\xA7', '\xD3', 'l', 'J', '\x0', 'i', '6',
  '\x7C', '\x8E', '\xE1', '\xFE', 'V', '\x84', '\xE7', '\x3C',
  '\x9F', 'r', '\x2B', '\x3A', 'B', '\x7B', '7', 'f',
  'w', '\xAE', '\x8E', '\xE', '\xF3', '\xBD', 'R', '\xA9',
  'd', '\x2', 'B', '\xAF', '\x85', '2', 'f', 'F',
  '\xBA', '\xC', '\xD9', '\x9F', '\x1D', '\x9A', 'l', '\x22',
  '\xE6', '\xC7', '\x3A', '\x2C', '\x80', '\xEF', '\xC1', '\x15',
  '\x90', '\x7', '\x93', '\xA2', '\x28', '\xA0', 'S', 'j',
  '\xB1', '\xB8', '\xDF', '\x29', '5', 'C', '\xE', '\x3F',
  'X', '\xFC', '\x98', '\xDA', 'y', 'j', 'P', '\x40',
  '\x0', '\x87', '\xAE', '\x1B', '\x17', 'B', '\xB4', '\x3A',
  '\x3F', '\xBE', 'y', '\xC7', '\xA', '\x26', '\xB6', '\xEE',
  '\xD9', '\x9A', '\x60', '\x14', '\x93', '\xDB', '\x8F', '\xD',
  '\xA', '\x2E', '\xE9', '\x23', '\x95', '\x29', 'X', '\x0',
  '\x27', '\xEB', 'n', 'V', 'p', '\xBC', '\xD6', '\xCB',
  '\xD6', 'G', '\xAB', '\x3D', 'l', '\x7D', '\xB8', '\xD2',
  '\xDD', '\xA0', '\x60', '\x83', '\xBA', '\xEF', '\x5F', '\xA4',
  '\xEA', '\xCC', '\x2', 'N', '\xAE', '\x5E', 'p', '\x1A',
  '\xEC', '\xB3', '\x40', '9', '\xAC', '\xFE', '\xF2', '\x91',
  '\x89', 'g', '\x91', '\x85', '\x21', '\xA8', '\x87', '\xB7',
  'X', '\x7E', '\x7E', '\x85', '\xBB', '\xCD', 'N', 'N',
  'b', 't', '\x40', '\xFA', '\x93', '\x89', '\xEC', '\x1E',
  '\xEC', '\x86', '\x2', 'H', '\x26', '\x93', '\xD0', 'u',
  '\x1D', '\x7F', '\x9', '2', '\x95', '\xBF', '\x1F', '\xDB',
  '\xD7', 'c', '\x8A', '\x1A', '\xF7', '\x5C', '\xC1', '\xFF',
  '\x22', 'J', '\xC3', '\x87', '\x0', '\x3', '\x0', 'K',
  '\xBB', '\xF8', '\xD6', '\x2A', 'v', '\x98', 'I', '\x0',
  '\x0', '\x0', '\x0', 'I', 'E', 'N', 'D', '\xAE',
  'B', '\x60', '\x82',
};
