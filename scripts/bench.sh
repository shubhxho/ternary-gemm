#!/usr/bin/env bash
# One-shot: build, verify correctness on the GPU, benchmark, write a Markdown report.
#
#   ./scripts/bench.sh            # full run (batches 1,2,4,8,16)
#   ./scripts/bench.sh --quick    # batches 1 and 16 only
#
# Needs: an NVIDIA GPU + driver (>= 570 for RTX 50-series), Rust (rustup), and the CUDA 12.x
# runtime libraries libnvrtc + libcublas (a CUDA Toolkit install, or this script can pip-install them).
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No NVIDIA GPU/driver visible (nvidia-smi failed). GPU numbers cannot be measured here." >&2
  exit 1
fi
command -v cargo >/dev/null 2>&1 || { echo "cargo not found: install Rust from https://rustup.rs" >&2; exit 1; }

say "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.memory,compute_cap --format=csv

# Locate libnvrtc / libcublas (dlopen'ed at runtime by cudarc).
for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib /opt/cuda/lib64 /usr/lib/wsl/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
have_lib() {
  ldconfig -p 2>/dev/null | grep -q "$1" && return 0
  local d
  IFS=: read -ra dirs <<< "${LD_LIBRARY_PATH:-}"
  for d in "${dirs[@]}"; do compgen -G "$d/$1*" >/dev/null && return 0; done
  return 1
}
if ! have_lib libnvrtc.so || ! have_lib libcublas.so; then
  say "libnvrtc/libcublas not found; installing CUDA 12.8 runtime wheels into .venv-cuda (one-time, ~700 MB)"
  python3 -m venv .venv-cuda
  .venv-cuda/bin/pip install -q "nvidia-cuda-nvrtc-cu12==12.8.*" "nvidia-cublas-cu12==12.8.*"
  NV=$(echo "$PWD"/.venv-cuda/lib/python3*/site-packages/nvidia)
  # cudarc tries unversioned names first; provide them.
  ln -sf libnvrtc.so.12 "$NV/cuda_nvrtc/lib/libnvrtc.so"
  ln -sf libcublas.so.12 "$NV/cublas/lib/libcublas.so"
  export LD_LIBRARY_PATH="$NV/cuda_nvrtc/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
fi

say "Build"
cargo build --release --all-targets

say "Correctness: CPU reference + kernel-index emulation + GPU vs CPU (GPU required)"
TERNARY_REQUIRE_GPU=1 cargo test --release -- --test-threads=1

mkdir -p results
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '__')
STEM="results/$(date +%Y%m%d-%H%M%S)-${GPU_NAME}"
OUT="$STEM.md"
say "Benchmark -> $OUT"
cargo run --release --bin bench -- --csv "$STEM.csv" "$@" | tee "$OUT"
{
  echo
  echo "<details><summary>Environment</summary>"
  echo
  echo '```'
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.memory,clocks.max.sm,power.limit --format=csv
  rustc --version
  git rev-parse HEAD 2>/dev/null || true
  echo '```'
  echo "</details>"
} >> "$OUT"
say "Charts from the measured CSV"
PY=python3
if ! $PY -c "import matplotlib" 2>/dev/null; then
  [ -x .venv-plot/bin/python ] || { python3 -m venv .venv-plot && .venv-plot/bin/pip install -q matplotlib; }
  PY=.venv-plot/bin/python
fi
if CHARTS=$($PY scripts/plot_results.py "$STEM.csv"); then
  { echo; echo "## Charts (measured)"; echo; for c in $CHARTS; do echo "![]($(basename "$c"))"; done; } >> "$OUT"
  echo "$CHARTS"
else
  echo "(chart generation failed; the Markdown and CSV results are still in results/)"
fi
say "Signal-model latency (batch 1, host -> signal -> host) -> $STEM-latency.md"
PIN=(); command -v taskset >/dev/null && PIN=(taskset -c "${PIN_CORE:-2}")
"${PIN[@]}" ./target/release/latency --csv "$STEM-latency.csv" ${LATENCY_ARGS:-} | tee "$STEM-latency.md"
if LCHARTS=$($PY scripts/plot_latency.py "$STEM-latency.csv"); then
  { echo; echo "## Charts (measured)"; echo; for c in $LCHARTS; do echo "![]($(basename "$c"))"; done; } >> "$STEM-latency.md"
  echo "$LCHARTS"
fi
say "Done. Reports: $OUT and $STEM-latency.md (+ CSVs, charts results/*.png)"
