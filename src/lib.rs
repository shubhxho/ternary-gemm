//! # ternary-gemm
//!
//! Fused ternary (BitNet b1.58-style, weights in `{-1, 0, +1}`) matmul kernels for LLM decode on
//! NVIDIA GPUs (sm_61+, tuned with sm_80..sm_120 in mind), driven from Rust via `cudarc`.
//!
//! * [`pack`]: absmean weight quantization and the 2-bit packed layouts.
//! * [`reference`]: pure-Rust CPU reference (correctness oracle).
//! * [`gpu`]: NVRTC-compiled CUDA kernels with a safe launch API, plus a cuBLAS FP16 baseline.
//! * [`mlp`]: small ternary MLPs (signal models): CPU reference, allocation-free SIMD CPU engine.
//! * [`signal`]: latency-first GPU engines for those models (CUDA graph + pinned memory, and an
//!   experimental persistent kernel), plus a cuBLAS FP16 baseline.

#![allow(clippy::needless_range_loop)]

pub mod gpu;
pub mod mlp;
pub mod pack;
pub mod reference;
pub mod signal;

pub use pack::{quantize_absmean, PackedTernary};
