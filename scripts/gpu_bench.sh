#!/usr/bin/env bash
# GPU kernel correctness and timing for someone with a CUDA device.
#
# What this script does *not* do: invent GPU numbers. On a CPU-only box it runs the
# CPU-side emulations (bit-exact vs the reference) and exits 0. Device tests and
# timings are skipped unless a GPU is actually present; any microsecond / GB/s figure
# printed by the examples is a measurement from *this* run only.
#
#   scripts/gpu_bench.sh              # CPU emulations always; device work if a GPU is visible
#   scripts/gpu_bench.sh --cpu-only   # never touch the driver
#   TERNARY_REQUIRE_GPU=1 scripts/gpu_bench.sh   # fail if there is no device
set -euo pipefail
cd "$(dirname "$0")/.."

cpu_only=0
for a in "$@"; do
  case "$a" in
    --cpu-only) cpu_only=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  esac
done

echo "== CPU-side kernel emulations (no GPU required)"
cargo test --lib --release -- cuda_ext pick_ext_kernel
cargo test --release --test kernel_emulation

have_gpu=0
if [ "$cpu_only" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  have_gpu=1
fi

if [ "$have_gpu" -eq 0 ]; then
  if [ "${TERNARY_REQUIRE_GPU:-0}" = 1 ]; then
    echo "TERNARY_REQUIRE_GPU=1 but nvidia-smi sees no device" >&2
    exit 1
  fi
  echo "No GPU visible (or --cpu-only). Skipping device tests and timings. No GPU numbers claimed."
  if command -v nvcc >/dev/null 2>&1; then
    echo "== nvcc is on PATH; compile-checking kernels for sm_80 and sm_120 (no run)"
    nvcc -ptx -std=c++17 -arch=sm_80 kernels/ternary_ext.cu -o /tmp/ternary_ext_sm80.ptx
    nvcc -ptx -std=c++17 -arch=sm_120 kernels/ternary_ext.cu -o /tmp/ternary_ext_sm120.ptx || \
      nvcc -ptx -std=c++17 -arch=sm_90 kernels/ternary_ext.cu -o /tmp/ternary_ext_sm90.ptx
    echo "nvcc PTX compile ok"
  else
    echo "nvcc not installed; kernel compile-check skipped (CPU emulations still ran)."
  fi
  exit 0
fi

echo "== device: $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1)"
echo "== bit-exact on device (ternary_ext + batched + bulk)"
cargo test --release --test gpu_ext
echo "== timings (measured on this GPU; write your own results/ file, do not copy numbers from a CPU box)"
cargo run --release --example bench_ext
echo "For the ITCH replay on GPU (batch 1 + throughput sweep): scripts/itch_gpu.sh"
