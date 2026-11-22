# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
* GitHub Actions CI: fmt, clippy `-D warnings`, tests on Linux and macOS, MSRV, rustdoc,
  NVRTC/ptxas kernel compile check for sm_80..sm_120, cargo-deny.
* Repository tooling: rustfmt/clippy configs, `.editorconfig`, `.gitattributes`, Dependabot.

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
