#!/usr/bin/env bash
set -euo pipefail

version="2.18.0"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${project_root}/.deps/tensorflow-${version}"
archive="${TMPDIR:-/tmp}/libtensorflow-cpu-linux-x86_64-${version}.tar.gz"
url="https://storage.googleapis.com/tensorflow/versions/${version}/libtensorflow-cpu-linux-x86_64.tar.gz"
expected_sha256="605bfcb370c7e7ec981eabada880f60784e3de018395be95c95c3e5592c3d9a2"

mkdir -p "${destination}"
curl -fL "${url}" -o "${archive}"
echo "${expected_sha256}  ${archive}" | sha256sum --check --status
tar -xzf "${archive}" -C "${destination}"
echo "TensorFlow C ${version} installed at ${destination}"
