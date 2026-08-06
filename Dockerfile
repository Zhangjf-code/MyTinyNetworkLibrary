# syntax=docker/dockerfile:1.7
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG TENSORFLOW_VERSION=2.18.0
ARG TENSORFLOW_SHA256=605bfcb370c7e7ec981eabada880f60784e3de018395be95c95c3e5592c3d9a2
ARG TENSORRT_VERSION=10.0.1.6
ARG TENSORRT_SHA256=a5cd2863793d69187ce4c73b2fffc1f470ff28cfd91e3640017e53b8916453d5

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake curl g++ make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/deps
RUN curl -fL --retry 3 \
        "https://storage.googleapis.com/tensorflow/versions/${TENSORFLOW_VERSION}/libtensorflow-cpu-linux-x86_64.tar.gz" \
        -o tensorflow.tar.gz && \
    echo "${TENSORFLOW_SHA256}  tensorflow.tar.gz" | sha256sum --check --status && \
    mkdir -p "tensorflow-${TENSORFLOW_VERSION}" && \
    tar -xzf tensorflow.tar.gz -C "tensorflow-${TENSORFLOW_VERSION}" && \
    rm tensorflow.tar.gz

RUN tensor_rt_archive="TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-12.4.tar.gz" && \
    curl -fL --retry 3 \
        "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/${tensor_rt_archive}" \
        -o tensorrt.tar.gz && \
    echo "${TENSORRT_SHA256}  tensorrt.tar.gz" | sha256sum --check --status && \
    mkdir -p "tensorrt-${TENSORRT_VERSION}" && \
    tar -xzf tensorrt.tar.gz --strip-components=1 -C "tensorrt-${TENSORRT_VERSION}" && \
    rm tensorrt.tar.gz

WORKDIR /workspace
COPY . .
RUN cmake -S . -B /tmp/mywebserver-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TENSORFLOW=ON \
        -DTENSORFLOW_ROOT="/opt/deps/tensorflow-${TENSORFLOW_VERSION}" \
        -DENABLE_TENSORRT=ON \
        -DTENSORRT_ROOT="/opt/deps/tensorrt-${TENSORRT_VERSION}" \
        -DCUDA_ROOT=/usr/local/cuda && \
    cmake --build /tmp/mywebserver-build --target HttpServer -j"$(nproc)"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TENSORFLOW_VERSION=2.18.0
ARG TENSORRT_VERSION=10.0.1.6
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/deps/tensorflow-${TENSORFLOW_VERSION} \
    /opt/deps/tensorflow-${TENSORFLOW_VERSION}
COPY --from=build /opt/deps/tensorrt-${TENSORRT_VERSION}/lib \
    /opt/deps/tensorrt-${TENSORRT_VERSION}/lib
COPY --from=build /workspace/lib/libmytinymuduo.so /opt/mywebserver/lib/libmytinymuduo.so
COPY --from=build /workspace/src/http/HttpServer /opt/mywebserver/bin/HttpServer
COPY models /opt/mywebserver/models
COPY root /opt/mywebserver/root

ENV LD_LIBRARY_PATH=/opt/mywebserver/lib:/opt/deps/tensorflow-${TENSORFLOW_VERSION}/lib:/opt/deps/tensorrt-${TENSORRT_VERSION}/lib:/usr/local/cuda/lib64 \
    MYWEBSERVER_PORT=10000 \
    MYWEBSERVER_INFERENCE_WORKERS=2 \
    MYWEBSERVER_QUEUE_CAPACITY=128 \
    MYWEBSERVER_MODEL_REGISTRY=/opt/mywebserver/models/registry.json \
    MYWEBSERVER_HTML_PATH=/opt/mywebserver/root/index4.html

WORKDIR /opt/mywebserver
EXPOSE 10000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${MYWEBSERVER_PORT}/healthz" || exit 1
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/mywebserver/bin/HttpServer"]
