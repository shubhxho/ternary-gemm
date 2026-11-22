#!/usr/bin/env bash
# Compile kernels/tensor_core.cu to PTX for sm_80+ targets.
# Prefers nvcc when it is on PATH; otherwise prints how to use NVRTC (compile_check).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p target/ptx
if command -v nvcc >/dev/null 2>&1; then
  for arch in compute_80 compute_89 compute_120; do
    out="target/ptx/tensor_core_${arch}.nvcc.ptx"
    echo "nvcc -ptx -arch=$arch -> $out"
    nvcc -std=c++17 -ptx -arch="$arch" -o "$out" kernels/tensor_core.cu
  done
else
  echo "nvcc not on PATH. The project's usual path is NVRTC (no CUDA toolkit needed):"
  echo "  cargo run --release --bin compile_check -- compute_80 compute_89 compute_120"
  echo "If ptxas is available, set PTXAS=/path/to/ptxas to also assemble SASS."
  exit 0
fi
