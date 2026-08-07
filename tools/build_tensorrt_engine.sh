#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
tensorrt_root="${TENSORRT_ROOT:-${project_dir}/.deps/tensorrt-10.0.1.6}"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-12.4}"
onnx="${1:-${project_dir}/models/yolov8n.onnx}"
engine="${2:-${project_dir}/models/yolov8n_fp16.engine}"

export LD_LIBRARY_PATH="${tensorrt_root}/lib:${cuda_root}/lib64:${LD_LIBRARY_PATH:-}"
"${tensorrt_root}/bin/trtexec" \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    --fp16 \
    --memPoolSize=workspace:1024 \
    --skipInference
