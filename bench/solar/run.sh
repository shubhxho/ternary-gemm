#!/usr/bin/env bash
# Fetch corpora, build tools if needed, run the Solar vs solc benchmark, write results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/fetch_corpus.sh"
"$ROOT/setup.sh"
python3 "$ROOT/run_bench.py" "$@"
