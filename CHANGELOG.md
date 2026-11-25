# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
* Improved ITCH training recipe (`scripts/itch_train.py --recipe improved`, the new default):
  all 3.61M labelled train samples, AdamW, cosine LR, and QAT logit distillation from the
  frozen dense teacher. Same official split, features and `[64, 256, 256, 1]` architecture as
  `results/itch-20261125/`. Rust-scored official test: QAT AUC 0.7124 vs the previous 0.7137
  (no improvement), PTQ 0.6763 → 0.6883, QAT 1 s Pearson IC 0.2045 → 0.2083, sign agreement
  with dense 0.8671 → 0.9231. `--recipe legacy` reproduces the original run. Walk-forward expanding-window folds (`--walk-forward`) on the same day: QAT mean AUC 0.7124 ± 0.0130 (four folds; not independent).
  Unit tests: `python3 scripts/test_itch_train.py`. Results: `results/itch-20261125-distill/`,
  `docs/ITCH_REPLAY.md`.
* Extra CUDA kernels for batched decode and Hopper/Blackwell prefetch (`kernels/ternary_ext.cu`):
  `tgx_gemv_batched` (W2A8 GEMV, batch 1..16, weight tiles reused across the batch) and
  `tgx_gemv_bulk_r2` (`cp.async.bulk` / TMA-style weight prefetch on sm_90+ including sm_120,
  `cp.async` fallback on sm_80). CPU-side `reference_gemm` / `emulate_batched` are bit-exact;
  `pick_ext_kernel` picks batched / split-K / bulk / cp.async from the device shape.
  **No GPU timings were measured** (authoring machine has no NVIDIA GPU). Run
  `scripts/gpu_bench.sh` on a CUDA box (`docs/CUDA_EXT.md`).
* `cpu_lanes::PackedLanes`: rows-in-lanes 2-bit GEMV for AVX-512 VNNI with GFNI (or VBMI)
  decode, a fused bit-exact dequant/bias/ReLU epilogue that also returns `max|out|`, and a
  threaded path on `SpinPool`. Default `CpuEngine` batch-1 path on AVX-512; `with_row_packed()`
  keeps Packed2. Measured p50 vs Packed2 (`main` 75af843), same VM: tiny 1.09 → 0.59 µs, small
  9.32 → 5.98 µs, medium 20.60 → 13.63 µs; 4 threads small 5.81 → 4.14, medium 10.11 → 8.84
  (`docs/CPU_PERF.md`).
* `mlp::quantize_with_max` and an AVX-512 quantizer (k = 256: 27 → 17 ns).
* ITCH hot path: single tree walk per side in `features`, integer `ln(1+x)` table, single-descent
  book update; `features_reference` oracle checked on every warm-up decision. Ternary replay end
  to end p50 1.455 → 0.895 µs, p99 2.11 → 1.47 µs, 0.70 → 1.14 M msgs/s (`docs/ITCH_REPLAY.md`,
  `results/itch-20261125-perf/`).

* ITCH 5.0 replay workload (`src/itch.rs`, `itch` binary, `scripts/itch_*.{sh,py}`): Nasdaq
  TotalView-ITCH parser, order-by-order book, top-of-book features, a dense vs ternary (QAT/PTQ)
  order-book model with chronological held-out evaluation, and per-stage replay latency. Measured
  on one public day (2019-01-30, 5 symbols); GPU replay script provided, not run
  (`docs/ITCH_REPLAY.md`, `results/itch-20261125/`).
* `mlp::DenseF32Mlp::from_f32` and `mlp::DenseF32KMajor` (AVX-512 k-major dense f32 baseline).
* `cpu_packed::Packed2`: 16-row × 256-K blocked 2-bit layout with an AVX-512 VBMI/VNNI GEMV
  (`vpmultishiftqb` decode + `vpdpbusd`, 16 accumulators), AVX2 and scalar fallbacks, and a
  bit-exact vectorized epilogue. Used by `CpuEngine` (AVX-512 only) and `CpuTernary`. Measured p50
  on the build VM: tiny 2.53 → 1.11 µs, small 20.2 → 9.32 µs, medium 87.7 → 22.3 µs
  (`docs/CPU_PERF.md`).
* `spin_pool::SpinPool` and `CpuEngine::with_threads`: medium 11.6 µs at 4 threads (measured).
* Explainer video for the Packed2 CPU speedup: `docs/media/cpu-speedup-explainer.mp4` (`scripts/make_cpu_speedup_media.py`).
* `latency --threads N --byte-codes --skip-reference`; `bench.sh --ext --cpu-threads N --byte-codes`.
* `kernels/ternary_ext.cu`: cp.async pipeline (r1/r2/r4), split-K, LUT and warp-specialized
  GEMV kernels with fused act + residual epilogue; `kernels/ternary_gemv.ptx` hand PTX;
  `Gpu::gemv_ext` / `gemv_hand_ptx`; `cuda_ext` CPU emulations; `examples/bench_ext.rs`.
  Compile-checked only (NVRTC 12.8, nvcc 13.5, sm_80..sm_120); no GPU numbers (`docs/CUDA_EXT.md`).
* Language ports with bit-exact tests against Rust vectors: Zig, Triton, Mojo, Julia, Futhark,
  Halide (`langs/`, `make langs`, `docs/LANGUAGES.md`).
* Tensor-core path for ternary GEMM/GEMV (sm_80+, including sm_89 and sm_120): unpack 2-bit
  natural-layout weights into int8/fp16 fragments and issue PTX `mma.sync.aligned.m16n8k16`,
  in the style of TriRun / BitBLAS W2A8. Selectable via `Launch::tensor_core` / `Backend::TensorCore`
  and included in the autotuner search space. CPU fragment-level reference is bit-exact on the
  int8 path (`tests/tensor_core.rs`); GPU tests are skipped when no device is present.
  `./scripts/bench.sh --tensor-cores` verifies and times the path. The 8× vs fp16 figure at
  batch 1 is documented as a theoretical bandwidth ceiling, not a measured GPU result.
  Docs: `docs/TENSOR_CORES.md`, report `docs/reports/tensor-core-ternary.{md,html}`,
  explainer video `docs/media/tensor-core-explainer.mp4`.
* GitHub Actions CI: fmt, clippy `-D warnings` over the feature matrix, tests on Linux and macOS,
  MSRV 1.88, rustdoc `-D warnings`, NVRTC/ptxas compile check for sm_80..sm_120 (+ sm_120a and a
  register-table test), cargo-deny, ruff, fuzz smoke, FPGA testbenches, Python wheel + pytest,
  `cargo package`.
* Repository tooling: rustfmt/clippy/ruff configs, `.editorconfig`, `.gitattributes`, Dependabot,
  issue/PR templates, CONTRIBUTING, SECURITY, Makefile, pre-commit hook, Dockerfile, devcontainer.
* Cargo features: `cuda` (default), `avx2` (default), `avx512` (VNNI), `convert`, `nvml`;
  `--no-default-features` is a portable CPU-only build.
* `cpu::CpuTernary`: bit-exact CPU W2A8 GEMM for any batch, SIMD, threaded, 4-row register
  blocking; `cpu::quantize_acts`: fast bit-exact activation quantizer.
* AVX2 activation quantizer and AVX-512 VNNI dot products for `CpuEngine`.
* `looped`: weight-shared recurrent-depth ternary transformer (input injection none/add/adapter,
  adaptive halting), `CpuBackend` and CUDA `GpuBackend`, example, criterion bench, docs.
* `tuning` (RTX 5090 / sm_120 first-class profile with cited specs, occupancy model, roofline
  projection), `autotune` runtime autotuner with CSV cache, `autotune` and `roofline` binaries;
  `scripts/bench.sh` autotunes and records results.
* `energy`: RAPL and NVML energy counters; `latency --energy` reports J/inference.
* `results`: environment capture + schema-versioned JSON (`latency --json`),
  `docs/results-schema.json`, `scripts/compare_results.py`.
* `.tgm` container (`format`), safetensors converter (`convert` feature + binary),
  `PackedTernary::{unpack, sparsity, from_natural}`, `pack::{pad_k, pad_k_acts}`.
* proptest properties, cargo-fuzz targets, criterion benchmarks.
* Python bindings (pyo3, abi3) with a PyTorch custom-op stub.
* FPGA skeleton: ternary dot/MAC RTL, HLS C++ kernel, self-checking testbenches.
* `third_party/linux` submodule (shallow, `update = none`) and `scripts/linux_tools.sh`.
* Docs: ARCHITECTURE, LOOPED, RTX5090, ENERGY, HARDWARE, LINUX.
* `bench/solar`: reproducible Paradigm Solar vs solc benchmark (pinned compiler and OpenZeppelin / Uniswap v3/v4 corpora), with measured median/p95 times, peak RSS, and a chart.

### Changed
* `SpinPool`: padded epoch and per-worker completion flags instead of a shared counter.
* GPU shape errors now say how to proceed (chunk batches > 16, zero-pad K, or use the CPU
  backend).

### Fixed
* MSRV corrected to 1.88 (a transitive dependency of `cudarc` requires it).
* Tensor-core path (`kernels/tensor_core.cu`) gave wrong results on real hardware (sm_120):
  accumulator fragment order (d1/d2 swapped) in both kernels, fp16 A-fragment register order
  (a1/a2 swapped), and warps past N returning before the cooperative activation load and
  `__syncthreads()`. The CPU model in `src/tensor_core.rs` had the same C-fragment layout and is
  corrected to the PTX ISA m16n8k16 table. All 85 tests pass on an RTX PRO 6000 Blackwell.

## [0.1.0] - 2026-11-22

### Added
* W2A8 ternary GEMV/GEMM kernels (batch 1-16) with fused RMSNorm + int8 quantization prologue and
  scale/residual epilogue; W2A16 path; naive baselines; NVRTC runtime compilation.
* Fused-kernel shared-memory opt-in, CUDA graph capture (`Gpu::capture`).
* Signal-model path: whole-MLP `ternary_mlp` kernel, `SignalEngine`, experimental
  `PersistentEngine`, cuBLAS FP16 baseline, AVX2 `CpuEngine`.
* Benchmarks (`bench`, `latency`), chart scripts, CPU reference and bit-exact tests.
