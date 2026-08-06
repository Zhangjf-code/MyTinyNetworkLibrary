#!/usr/bin/env bash
set -euo pipefail

version="10.0.1.6"
archive="TensorRT-${version}.Linux.x86_64-gnu.cuda-12.4.tar.gz"
url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/${archive}"
expected_sha256="a5cd2863793d69187ce4c73b2fffc1f470ff28cfd91e3640017e53b8916453d5"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
download="${TMPDIR:-/tmp}/${archive}"
destination="${project_dir}/.deps/tensorrt-${version}"

curl -L --fail --retry 3 --continue-at - --output "$download" "$url"
printf '%s  %s\n' "$expected_sha256" "$download" | sha256sum --check --status
mkdir -p "$destination"
tar -xzf "$download" --strip-components=1 -C "$destination" \
    "TensorRT-${version}/include" \
    "TensorRT-${version}/lib" \
    "TensorRT-${version}/bin" \
    "TensorRT-${version}/targets/x86_64-linux-gnu"
printf 'TensorRT installed at %s\n' "$destination"
