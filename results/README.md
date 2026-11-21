Benchmark reports from the scripts go here, one set per run:
* `scripts/bench.sh` (needs a GPU): `<timestamp>-<gpu>.{md,csv}` for the GEMM/layer-stack benchmark, `<timestamp>-<gpu>-latency.{md,csv}` for signal-model latency, plus `-*.png` charts drawn from each CSV.
* `scripts/latency_cpu.sh` (no GPU needed): `<timestamp>-cpu-latency.{md,csv,png}`.

Committed so far: `20261122-025900-cpu-latency.*`, a CPU-only measurement on the build machine (Intel Xeon cloud VM, 8 vCPUs, pinned to one core). **No GPU report has been committed yet. Nothing in this repo has been measured on a GPU.**
