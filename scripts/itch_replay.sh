#!/usr/bin/env bash
# ITCH replay benchmark on the CPU: extract features, (re)train, replay every engine pinned to one
# core, then score signal quality on the held-out period. Writes results/itch-<time>/.
#
#   scripts/itch_replay.sh                 # uses the committed models in results/itch-20261125/models
#   RETRAIN=1 scripts/itch_replay.sh       # retrain dense + QAT ternary from the extracted samples
#   PIN_CORE=3 ITCH_DATA=/big/disk scripts/itch_replay.sh
#
# Needs data/itch/slice.itch (scripts/itch_download.sh) and python3 + numpy.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${ITCH_DATA:-data/itch}
SLICE=${SLICE:-$DATA/slice.itch}
[ -s "$SLICE" ] || { echo "no $SLICE: run scripts/itch_download.sh first"; exit 1; }
OUT=results/itch-$(date +%Y%m%d-%H%M%S)
MODELS=${MODELS:-results/itch-20261125/models}
mkdir -p "$OUT/models"
PIN=(); command -v taskset >/dev/null && PIN=(taskset -c "${PIN_CORE:-2}")
FEAT=avx2; grep -qE "avx512_?vnni" /proc/cpuinfo 2>/dev/null && FEAT=avx512
cargo build --release --bin itch --no-default-features --features "$FEAT"
BIN=./target/release/itch
{ echo "date: $(date -Iseconds)"; echo "git: $(git rev-parse HEAD)"; echo "rustc: $(rustc --version)";
  lscpu | grep -E "Model name|^CPU\(s\)|L1d|L2|L3" ; echo "slice: $SLICE $(stat -c %s "$SLICE") bytes";
  echo "pin: ${PIN[*]:-none}"; } > "$OUT/env.txt"

echo "== extract features"
"${PIN[@]}" $BIN replay --data "$SLICE" --engine none --samples-out "$DATA/samples.bin" --json "$OUT/replay_none.json" | tee "$OUT/replay_none.md"
if [ "${RETRAIN:-0}" = 1 ] || [ ! -s "$MODELS/dense.tgmlp" ]; then
  echo "== train (dense f32, then ternary QAT)"
  python3 scripts/itch_train.py "$DATA/samples.bin" --out "$OUT/models"
  MODELS=$OUT/models
else
  cp "$MODELS"/*.tgmlp "$OUT/models/"
  python3 - "$DATA/samples.bin" "$OUT/models" <<'PY'
import sys
sys.argv = [sys.argv[0], sys.argv[1], "--out", sys.argv[2]]
import importlib.util
spec = importlib.util.spec_from_file_location("t", "scripts/itch_train.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
import numpy as np
rec = np.memmap(sys.argv[1], dtype=t.REC, mode="r")
y, t_next, fwd1s = t.labels(rec)
ts = np.asarray(rec["ts"])
idx = {k: np.nonzero((ts >= lo) & (ts < hi) & (y >= 0) & (t_next < hi))[0] for k, (lo, hi) in t.SPLITS.items()}
np.savez(sys.argv[3] + "/split.npz", y=y, t_next=t_next, fwd1s=fwd1s, **idx)
PY
fi
SPLIT=$OUT/models/split.npz

run() { # name engine model
  echo "== replay $1"
  "${PIN[@]}" $BIN replay --data "$SLICE" --engine "$2" --model "$3" --preds-out "$DATA/preds_$1.f32" --json "$OUT/replay_$1.json" | tee "$OUT/replay_$1.md"
  "${PIN[@]}" $BIN throughput --data "$SLICE" --engine "$2" --model "$3" | tee -a "$OUT/throughput.md"
}
run dense dense "$MODELS/dense.tgmlp"
run dense_kmajor dense-kmajor "$MODELS/dense.tgmlp"
run ternary_qat ternary "$MODELS/qat.tgmlp"
run ternary_ptq ternary "$MODELS/dense.tgmlp"
run ternary_bytes ternary-bytes "$MODELS/qat.tgmlp"

echo "== signal quality (test period 13:35-16:00 ET)"
python3 scripts/itch_eval.py "$DATA/samples.bin" "$SPLIT" \
  "dense f32=$DATA/preds_dense.f32" "dense f32 k-major=$DATA/preds_dense_kmajor.f32" \
  "ternary QAT=$DATA/preds_ternary_qat.f32" "ternary PTQ=$DATA/preds_ternary_ptq.f32" \
  --json "$OUT/quality.json" | tee "$OUT/quality.md"
rm -f "$SPLIT"
echo "results: $OUT"
