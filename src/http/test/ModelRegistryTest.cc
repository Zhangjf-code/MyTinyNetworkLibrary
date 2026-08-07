#include "ModelRegistry.h"

#include <cassert>
#include <memory>

class RegistryBackend final : public IInferenceBackend
{
public:
    RegistryBackend(BackendType type, bool loads) : type_(type), loads_(loads), loaded_(false) {}
    bool load(const ModelConfig&, std::string* error) override
    {
        loaded_ = loads_;
        if (!loaded_ && error) *error = "intentional load failure";
        return loaded_;
    }
    bool warmup(std::string*) override { return loaded_; }
    InferenceResult infer(const InferenceRequest&) override { return InferenceResult(); }
    bool healthy() const override { return loaded_; }
    BackendType type() const override { return type_; }

private:
    BackendType type_;
    bool loads_;
    bool loaded_;
};

ModelConfig config(BackendType backend)
{
    ModelConfig result;
    result.name = "yolov8n";
    result.version = backend == BackendType::kTensorRT ? "trt" : "tf";
    result.path = "unused";
    result.backend = backend;
    return result;
}

int main()
{
    {
        ModelRegistry registry;
        std::string error;
        assert(registry.loadFromFile(TEST_REGISTRY_PATH, &error));
        assert(registry.defaultModel() == "yolov8n");
        assert(registry.registrations().size() == 2);
        assert(registry.registrations()[0].config.path.find("models/yolov8n_fp16.engine") !=
               std::string::npos);
    }
    {
        ModelRegistry registry;
        std::string error;
        std::shared_ptr<IInferenceBackend> tensorflow(
            new RegistryBackend(BackendType::kTensorFlow, true));
        std::shared_ptr<IInferenceBackend> tensorrt(
            new RegistryBackend(BackendType::kTensorRT, true));
        assert(registry.add(config(BackendType::kTensorFlow), tensorflow, 10, &error));
        assert(registry.add(config(BackendType::kTensorRT), tensorrt, 100, &error));
        assert(registry.initialize(&error));

        std::string model = "default";
        assert(registry.resolve(&model, BackendType::kAuto) == tensorrt);
        assert(model == "yolov8n");
        model = "yolov8n";
        assert(registry.resolve(&model, BackendType::kTensorFlow) == tensorflow);
        model = "missing";
        assert(!registry.resolve(&model, BackendType::kAuto));
    }
    {
        ModelRegistry registry;
        std::string error;
        std::shared_ptr<IInferenceBackend> tensorflow(
            new RegistryBackend(BackendType::kTensorFlow, true));
        std::shared_ptr<IInferenceBackend> tensorrt(
            new RegistryBackend(BackendType::kTensorRT, false));
        assert(registry.add(config(BackendType::kTensorFlow), tensorflow, 10, &error));
        assert(registry.add(config(BackendType::kTensorRT), tensorrt, 100, &error));
        assert(registry.initialize(&error));
        std::string model = "yolov8n";
        assert(registry.resolve(&model, BackendType::kAuto) == tensorflow);
        model = "yolov8n";
        assert(!registry.resolve(&model, BackendType::kTensorRT));
    }
    return 0;
}
