#!/usr/bin/env bash
# One-shot: build, verify correctness on the GPU, benchmark, write a Markdown report.
#
#   ./scripts/bench.sh                    # full run (batches 1,2,4,8,16)
#   ./scripts/bench.sh --quick            # batches 1 and 16 only
#   ./scripts/bench.sh --tensor-cores     # also verify and time the sm_80+ mma.sync path
#   ./scripts/bench.sh --quick --tensor-cores
#   ./scripts/bench.sh --ext              # also time cp.async / split-K / LUT / warp-spec / hand-PTX
#   ./scripts/bench.sh --cpu-threads 4    # latency: add a CpuEngine::with_threads(4) row (cores 2-5)
#   ./scripts/bench.sh --byte-codes       # latency: add the 1-byte-per-weight CPU path next to packed
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
cargo build --release --all-targets --features nvml

say "Correctness: CPU reference + kernel-index emulation + GPU vs CPU (GPU required)"
TERNARY_REQUIRE_GPU=1 cargo test --release --features nvml -- --test-threads=1

mkdir -p results
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '__')
STEM="results/$(date +%Y%m%d-%H%M%S)-${GPU_NAME}"
OUT="$STEM.md"
say "Benchmark -> $OUT"
TENSOR_CORES=0
EXT=0
CPU_THREADS=1
BYTE_CODES=0
BENCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tensor-cores) TENSOR_CORES=1; BENCH_ARGS+=("$1") ;;
    --ext) EXT=1 ;;
    --cpu-threads) CPU_THREADS=$2; shift ;;
    --byte-codes) BYTE_CODES=1 ;;
    *) BENCH_ARGS+=("$1") ;;
  esac
  shift
done
set -- ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
if [ "$TENSOR_CORES" = 1 ]; then
  echo "Including the tensor-core backend (mma.sync, sm_80+). GPU speedups are measured here; they are not estimated."
fi
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
say "Autotune launch configs for this GPU -> $STEM-autotune.{md,csv}"
# Tuning cache per machine (gitignored); the per-run CSV/MD is the recorded result.
./target/release/autotune --out "$STEM-autotune" --cache results/tune-cache.csv
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "$CC" = "12.0" ]; then
  echo "Blackwell sm_120 (RTX 50-series) detected: autotune results recorded in $STEM-autotune.md"
fi

if [ "$EXT" = 1 ]; then
  say "Extra batch-1 kernels (cp.async, split-K, LUT, warp-spec, hand PTX) -> $STEM-ext.md"
  cargo run --release --example bench_ext | tee "$STEM-ext.md"
fi

say "Signal-model latency (batch 1, host -> signal -> host) -> $STEM-latency.md"
LAT_EXTRA=()
[ "$BYTE_CODES" = 1 ] && LAT_EXTRA+=(--byte-codes)
PIN=(); command -v taskset >/dev/null && PIN=(taskset -c "${PIN_CORE:-2}")
if [ "$CPU_THREADS" -gt 1 ]; then
  # The threaded row needs one core per thread: pin to a range instead of one core.
  LAT_EXTRA+=(--threads "$CPU_THREADS")
  command -v taskset >/dev/null && PIN=(taskset -c "${PIN_CORE:-2}-$(( ${PIN_CORE:-2} + CPU_THREADS - 1 ))")
fi
"${PIN[@]}" ./target/release/latency --csv "$STEM-latency.csv" --energy ${LAT_EXTRA[@]+"${LAT_EXTRA[@]}"} ${LATENCY_ARGS:-} | tee "$STEM-latency.md"
if LCHARTS=$($PY scripts/plot_latency.py "$STEM-latency.csv"); then
  { echo; echo "## Charts (measured)"; echo; for c in $LCHARTS; do echo "![]($(basename "$c"))"; done; } >> "$STEM-latency.md"
  echo "$LCHARTS"
fi
say "Done. Reports: $OUT and $STEM-latency.md (+ CSVs, charts results/*.png)"
