# 指定使用 CMake 的最低版本号
cmake_minimum_required(VERSION 3.0)

# 设置项目名称
project(mytinymuduo C CXX)

enable_testing()
option(ENABLE_TENSORFLOW "Build the TensorFlow C API inference backend" OFF)
set(TENSORFLOW_ROOT "${PROJECT_SOURCE_DIR}/.deps/tensorflow-2.18.0" CACHE PATH "TensorFlow C API root")
option(ENABLE_TENSORRT "Build the TensorRT inference backend" OFF)
set(TENSORRT_ROOT "${PROJECT_SOURCE_DIR}/.deps/tensorrt-10.0.1.6" CACHE PATH "TensorRT root")
set(CUDA_ROOT "/usr/local/cuda-12.4" CACHE PATH "CUDA Toolkit root")

# cmake => makefile   make
# mymuduo最终编译成so动态库，设置动态库的路径，放在根目录的lib文件夹下面
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 设置调试信息 以及 启动C++11语言标准
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -fPIC")

# 设置项目编译头文件搜索路径 -I
include_directories(${PROJECT_SOURCE_DIR}/src/base)
include_directories(${PROJECT_SOURCE_DIR}/src/http)
include_directories(${PROJECT_SOURCE_DIR}/src/muduo_core)
include_directories(${PROJECT_SOURCE_DIR}/src/timer)
include_directories(${PROJECT_SOURCE_DIR}/src/logger)
include_directories(${PROJECT_SOURCE_DIR}/src/keepalive)
include_directories(${PROJECT_SOURCE_DIR}/src/inference)
include_directories(${PROJECT_SOURCE_DIR}/src/api)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
add_definitions(-DDEFAULT_MODEL_REGISTRY_PATH="${PROJECT_SOURCE_DIR}/models/registry.json")
add_definitions(-DDEFAULT_HTML_PATH="${PROJECT_SOURCE_DIR}/root/index4.html")
# include_directories(${PROJECT_SOURCE_DIR}/src/mysql)

# 设置项目库文件搜索路径 -L
link_directories(${PROJECT_SOURCE_DIR}/lib)

# 定义参与编译的源代码文件 
aux_source_directory(${PROJECT_SOURCE_DIR}/src/base/ SRC_BASE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/http/ SRC_HTTP)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/timer/ SRC_TIMER)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/logger/ SRC_LOG)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/muduo_core/ SRC_MUDUO_CORE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/keepalive/ SRC_KEEPALIVE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/inference/ SRC_INFERENCE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/api/ SRC_API)
# aux_source_directory(${PROJECT_SOURCE_DIR}/src/mysql SRC_MYSQL)

# main.cc belongs to the HttpServer executable, not the reusable shared library.
set(SRC_HTTP_WITHOUT_MAIN)
foreach(HTTP_SOURCE ${SRC_HTTP})
    get_filename_component(HTTP_SOURCE_NAME "${HTTP_SOURCE}" NAME)
    if(NOT HTTP_SOURCE_NAME STREQUAL "main.cc")
        list(APPEND SRC_HTTP_WITHOUT_MAIN "${HTTP_SOURCE}")
    endif()
endforeach()
set(SRC_HTTP ${SRC_HTTP_WITHOUT_MAIN})

# aux_source_directory() may retain a double slash from the directory argument,
# so filter by filename instead of relying on an exact absolute-path match.
set(SRC_INFERENCE_WITHOUT_TENSORFLOW)
foreach(INFERENCE_SOURCE ${SRC_INFERENCE})
    get_filename_component(INFERENCE_SOURCE_NAME "${INFERENCE_SOURCE}" NAME)
    if(NOT INFERENCE_SOURCE_NAME STREQUAL "TensorFlowBackend.cc" AND
       NOT INFERENCE_SOURCE_NAME STREQUAL "TensorRTBackend.cc")
        list(APPEND SRC_INFERENCE_WITHOUT_TENSORFLOW "${INFERENCE_SOURCE}")
    endif()
endforeach()
set(SRC_INFERENCE ${SRC_INFERENCE_WITHOUT_TENSORFLOW})
if(ENABLE_TENSORFLOW)
    if(NOT EXISTS "${TENSORFLOW_ROOT}/include/tensorflow/c/c_api.h")
        message(FATAL_ERROR "TensorFlow C API not found. Run tools/fetch_tensorflow_c.sh or set TENSORFLOW_ROOT")
    endif()
    list(APPEND SRC_INFERENCE "${PROJECT_SOURCE_DIR}/src/inference/TensorFlowBackend.cc")
    include_directories(${TENSORFLOW_ROOT}/include)
endif()
if(ENABLE_TENSORRT)
    if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h" OR
       NOT EXISTS "${TENSORRT_ROOT}/lib/libnvinfer.so" OR
       NOT EXISTS "${CUDA_ROOT}/include/cuda_runtime_api.h")
        message(FATAL_ERROR "TensorRT/CUDA C++ dependencies not found. Set TENSORRT_ROOT and CUDA_ROOT")
    endif()
    list(APPEND SRC_INFERENCE "${PROJECT_SOURCE_DIR}/src/inference/TensorRTBackend.cc")
    include_directories(${TENSORRT_ROOT}/include ${CUDA_ROOT}/include)
endif()


# 编译生成动态库mytinymuduo
add_library(mytinymuduo SHARED 
            ${SRC_BASE}
            ${SRC_TIMER}
            ${SRC_LOG}
            ${SRC_HTTP}
            ${SRC_MUDUO_CORE}
            ${SRC_INFERENCE}
            ${SRC_API}
            ${SRC_MYSQL}
            )

# 目标动态库所需连接的库 libthread libmysqlclient
target_link_libraries(mytinymuduo pthread)
if(ENABLE_TENSORFLOW)
    target_link_libraries(mytinymuduo ${TENSORFLOW_ROOT}/lib/libtensorflow.so)
    target_compile_definitions(mytinymuduo PUBLIC MYTINYMUDUO_ENABLE_TENSORFLOW=1)
    target_compile_definitions(mytinymuduo PUBLIC
        DEFAULT_TENSORFLOW_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/yolov8n_tf218_saved_model")
    set_target_properties(mytinymuduo PROPERTIES
        BUILD_RPATH "${TENSORFLOW_ROOT}/lib"
        INSTALL_RPATH "${TENSORFLOW_ROOT}/lib")
endif()
if(ENABLE_TENSORRT)
    target_link_libraries(mytinymuduo
        ${TENSORRT_ROOT}/lib/libnvinfer.so
        ${CUDA_ROOT}/lib64/libcudart.so)
    target_compile_definitions(mytinymuduo PUBLIC
        MYTINYMUDUO_ENABLE_TENSORRT=1
        DEFAULT_TENSORRT_ENGINE_PATH="${PROJECT_SOURCE_DIR}/models/yolov8n_fp16.engine")
    set_property(TARGET mytinymuduo APPEND PROPERTY BUILD_RPATH
        "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64")
endif()

# 设置安装的默认路径
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

#  将src的头文件中的.h文件复制到lib/include/mytinymuduo 下
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION lib/include/mytinymuduo
        FILES_MATCHING PATTERN "*.h")


add_subdirectory(src/http)

add_subdirectory(src/http/test)

add_subdirectory(example/server)

add_subdirectory(example/server-timequeue)

add_subdirectory(example/client)
