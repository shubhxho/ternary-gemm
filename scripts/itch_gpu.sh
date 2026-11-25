#!/usr/bin/env bash
# ITCH replay on a CUDA machine: the same replay with inference on the GPU (ternary SignalEngine:
# CUDA graph + pinned buffers + spin wait, batch 1), then inference-only throughput at several
# batch sizes, plus the CPU ternary engine on the same host for a same-box comparison.
# Writes results/itch-gpu-<time>/. Not run in the PR (the authoring machine has no GPU).
#
#   scripts/itch_download.sh && scripts/itch_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${ITCH_DATA:-data/itch}
SLICE=${SLICE:-$DATA/slice.itch}
MODELS=${MODELS:-results/itch-20261125/models}
[ -s "$SLICE" ] || { echo "no $SLICE: run scripts/itch_download.sh first"; exit 1; }
OUT=results/itch-gpu-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
PIN=(); command -v taskset >/dev/null && PIN=(taskset -c "${PIN_CORE:-2}")
FEAT=cuda,avx2; grep -qE "avx512_?vnni" /proc/cpuinfo 2>/dev/null && FEAT=cuda,avx512
cargo build --release --bin itch --features "$FEAT"
BIN=./target/release/itch
{ echo "date: $(date -Iseconds)"; echo "git: $(git rev-parse HEAD)"; lscpu | grep -E "Model name";
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader || true; } > "$OUT/env.txt"
echo "== GPU replay (batch 1)"
"${PIN[@]}" $BIN replay --data "$SLICE" --engine gpu-ternary --model "$MODELS/qat.tgmlp" ${MAX_DECISIONS:+--max-decisions $MAX_DECISIONS} | tee "$OUT/replay_gpu_ternary.md"
echo "== CPU replay on this host (same model)"
"${PIN[@]}" $BIN replay --data "$SLICE" --engine ternary --model "$MODELS/qat.tgmlp" --json "$OUT/replay_cpu_ternary.json" | tee "$OUT/replay_cpu_ternary.md"
echo "== inference-only throughput"
"${PIN[@]}" $BIN throughput --data "$SLICE" --engine ternary --model "$MODELS/qat.tgmlp" | tee "$OUT/throughput.md"
for b in 1 8 64 512 4096; do
  "${PIN[@]}" $BIN throughput --data "$SLICE" --engine gpu-ternary --model "$MODELS/qat.tgmlp" --batch $b | tee -a "$OUT/throughput.md"
done
echo "results: $OUT"
