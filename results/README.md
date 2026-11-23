Benchmark reports from the scripts go here, one set per run:
* `scripts/bench.sh` (needs a GPU): `<timestamp>-<gpu>.{md,csv}` for the GEMM/layer-stack benchmark, `<timestamp>-<gpu>-latency.{md,csv}` for signal-model latency, plus `-*.png` charts drawn from each CSV.
* `scripts/latency_cpu.sh` (no GPU needed): `<timestamp>-cpu-latency.{md,csv,png}`.

Committed so far:
* `20261122-025900-cpu-latency.*`, a CPU-only measurement on the build machine (Intel Xeon cloud VM, 8 vCPUs, pinned to one core).
* `20261123-020403-NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition*`: the first GPU run, from `./scripts/bench.sh --tensor-cores --ext --cpu-threads 4` with `PEAK_GBPS=1597` on an RTX PRO 6000 Blackwell Server Edition (sm_120, GB202, 188 SMs, 1597 GB/s; a GCP g4-standard-48 VM with an AMD EPYC 9B45). The tree was commit 028884b plus the tensor-core fragment fix (the environment block shows the base commit). Two caveats for reading it: on this VM, stream-launched kernels complete on a 2.048 µs grid, so rows under about 20 µs are rounded up (an empty kernel costs 2.03 µs per launch on a stream and 0.47 µs through a CUDA graph); and `-ext.md` reuses the same weights for every launch, so its large shapes are served from L2 (128 MB), not DRAM.
