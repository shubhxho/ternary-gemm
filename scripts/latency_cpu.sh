#!/usr/bin/env bash
# Signal-model latency on the CPU only (works without a GPU): results/<time>-cpu-latency.{md,csv,png}
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --bin latency
STEM="results/$(date +%Y%m%d-%H%M%S)-cpu-latency"
PIN=(); command -v taskset >/dev/null && PIN=(taskset -c "${PIN_CORE:-2}")
"${PIN[@]}" ./target/release/latency --cpu-only --csv "$STEM.csv" "$@" | tee "$STEM.md"
PY=python3; $PY -c "import matplotlib" 2>/dev/null || { [ -x .venv-plot/bin/python ] || { python3 -m venv .venv-plot && .venv-plot/bin/pip install -q matplotlib; }; PY=.venv-plot/bin/python; }
CHARTS=$($PY scripts/plot_latency.py "$STEM.csv")
{ echo; echo "## Charts (measured)"; echo; for c in $CHARTS; do echo "![]($(basename "$c"))"; done; } >> "$STEM.md"
echo "$CHARTS"
