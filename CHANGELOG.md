# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
* GPU shape errors now say how to proceed (chunk batches > 16, zero-pad K, or use the CPU
  backend).

### Fixed
* MSRV corrected to 1.88 (a transitive dependency of `cudarc` requires it).

## [0.1.0] - 2026-11-22

### Added
* W2A8 ternary GEMV/GEMM kernels (batch 1-16) with fused RMSNorm + int8 quantization prologue and
  scale/residual epilogue; W2A16 path; naive baselines; NVRTC runtime compilation.
* Fused-kernel shared-memory opt-in, CUDA graph capture (`Gpu::capture`).
* Signal-model path: whole-MLP `ternary_mlp` kernel, `SignalEngine`, experimental
  `PersistentEngine`, cuBLAS FP16 baseline, AVX2 `CpuEngine`.
* Benchmarks (`bench`, `latency`), chart scripts, CPU reference and bit-exact tests.
