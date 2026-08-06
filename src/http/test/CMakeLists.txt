add_executable(HttpContextTest
    HttpContextTest.cc
    ../HttpContext.cc
    ../../muduo_core/Buffer.cc
    ../../base/Timestamp.cc
)

target_include_directories(HttpContextTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/http
    ${PROJECT_SOURCE_DIR}/src/muduo_core
    ${PROJECT_SOURCE_DIR}/src/base
)

add_test(NAME HttpContextTest COMMAND HttpContextTest)

add_executable(InferenceApiTest
    InferenceApiTest.cc
)

target_include_directories(InferenceApiTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/api
    ${PROJECT_SOURCE_DIR}/src/inference
    ${PROJECT_SOURCE_DIR}/src/http
    ${PROJECT_SOURCE_DIR}/src/muduo_core
    ${PROJECT_SOURCE_DIR}/src/base
    ${PROJECT_SOURCE_DIR}/third_party
)

add_test(NAME InferenceApiTest COMMAND InferenceApiTest)

target_link_libraries(InferenceApiTest mytinymuduo)

add_executable(InferenceSchedulerTest
    InferenceSchedulerTest.cc
    ../../inference/InferenceScheduler.cc
    ../../inference/InferenceMetrics.cc
    ../../inference/ModelRegistry.cc
)

target_include_directories(InferenceSchedulerTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/inference
    ${PROJECT_SOURCE_DIR}/third_party
)

target_link_libraries(InferenceSchedulerTest pthread)

add_test(NAME InferenceSchedulerTest COMMAND InferenceSchedulerTest)

add_executable(ModelRegistryTest ModelRegistryTest.cc)
target_include_directories(ModelRegistryTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/inference
    ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(ModelRegistryTest PRIVATE
    TEST_REGISTRY_PATH="${PROJECT_SOURCE_DIR}/models/registry.json")
target_link_libraries(ModelRegistryTest mytinymuduo)
add_test(NAME ModelRegistryTest COMMAND ModelRegistryTest)

add_executable(ManagementApiTest ManagementApiTest.cc)
target_include_directories(ManagementApiTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/api
    ${PROJECT_SOURCE_DIR}/src/inference
    ${PROJECT_SOURCE_DIR}/src/http
    ${PROJECT_SOURCE_DIR}/src/muduo_core
    ${PROJECT_SOURCE_DIR}/src/base
    ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ManagementApiTest mytinymuduo)
add_test(NAME ManagementApiTest COMMAND ManagementApiTest)

add_executable(YoloV8ProcessorTest
    YoloV8ProcessorTest.cc
    ../../inference/YoloV8Processor.cc
)
target_include_directories(YoloV8ProcessorTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/inference
    ${PROJECT_SOURCE_DIR}/third_party
)
target_compile_definitions(YoloV8ProcessorTest PRIVATE
    TEST_IMAGE_PATH="${PROJECT_SOURCE_DIR}/root/picture/ann.png")
add_test(NAME YoloV8ProcessorTest COMMAND YoloV8ProcessorTest)

if(ENABLE_TENSORFLOW)
    add_executable(TensorFlowBackendTest TensorFlowBackendTest.cc)
    target_include_directories(TensorFlowBackendTest PRIVATE
        ${PROJECT_SOURCE_DIR}/src/inference
        ${TENSORFLOW_ROOT}/include)
    target_compile_definitions(TensorFlowBackendTest PRIVATE
        TEST_IMAGE_PATH="${PROJECT_SOURCE_DIR}/root/picture/ultralytics_bus.jpg"
        TEST_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/yolov8n_tf218_saved_model")
    target_link_libraries(TensorFlowBackendTest mytinymuduo)
    add_test(NAME TensorFlowBackendTest COMMAND TensorFlowBackendTest)
endif()

if(ENABLE_TENSORRT)
    add_executable(TensorRTBackendTest TensorRTBackendTest.cc)
    target_include_directories(TensorRTBackendTest PRIVATE
        ${PROJECT_SOURCE_DIR}/src/inference
        ${TENSORRT_ROOT}/include
        ${CUDA_ROOT}/include)
    target_compile_definitions(TensorRTBackendTest PRIVATE
        TEST_IMAGE_PATH="${PROJECT_SOURCE_DIR}/root/picture/ultralytics_bus.jpg"
        TEST_ENGINE_PATH="${PROJECT_SOURCE_DIR}/models/yolov8n_fp16.engine")
    target_link_libraries(TensorRTBackendTest mytinymuduo)
    add_test(NAME TensorRTBackendTest COMMAND TensorRTBackendTest)
endif()
