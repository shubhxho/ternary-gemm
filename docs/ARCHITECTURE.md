# Architecture

```text
                 safetensors (f32/f16/bf16)
                          | convert (feature convert): absmean ternarization
                          v
   .tgm container  <-- format.rs (magic, versioned, typed tensors, FNV-1a checksum)
                          |
                          v
   PackedTernary (pack.rs): 2-bit codes w+1, permuted layout (GPU) + natural layout (CPU/FPGA/.tgm),
                            per-row f32 scale; pad_k for K % 64 != 0
        |                    |                         |                        |
        v                    v                         v                        v
 reference.rs          cpu.rs CpuTernary          gpu.rs (feature cuda)     hw/ (RTL, HLS)
 scalar oracle         SIMD, any batch,           NVRTC kernels, safe       multiplier-free
 (f64-free int math)   threads, bit-exact         launch API, cuBLAS        dot/MAC, same
                                                  baseline                  natural layout
        \____________________ all bit-exact on the int8 path ______________________/
                                   |
              +--------------------+--------------------+
              v                                         v
   looped.rs: weight-shared transformer         mlp.rs / signal.rs: small signal models
   (Backend trait: CpuBackend / GpuBackend)     (CpuEngine SIMD, GPU graph/persistent engines)

   tuning.rs (arch profiles, occupancy, roofline)  ->  autotune.rs (times configs on the device)
   energy.rs (RAPL, NVML)  +  results.rs (env capture, JSON)  ->  latency/bench/autotune binaries
```

## Invariants

1. **One arithmetic contract.** `y = (sum_k w[k] q[k]) / s * scale (+ residual)`, integer
   accumulation, then a fixed sequence of IEEE round-to-nearest f32 ops, rounded once to fp16.
   `reference::gemm_i8` defines it; `cpu::CpuTernary`, the CUDA kernels (`__fdiv_rn`,
   `__fmul_rn`) and the FPGA integer core (epilogue on the host) implement it. Bit-exactness is
   tested pairwise (`tests/cpu_backend.rs`, `tests/properties.rs`, `tests/gpu.rs`,
   `tests/kernel_emulation.rs`, `hw/tb`).
2. **One quantizer contract.** Absmax int8 per token, `s = 127 / max(absmax, 1e-5)`,
   round-half-even, optional RMSNorm (`(x * g) * inv`, sum of squares in f64 on the CPU).
   `mlp::quantize_scalar` and `reference::quantize_acts` are the oracles; every SIMD variant
   (`quantize_avx2`, `cpu::quantize_acts`) is tested against them, including by fuzzing.
   The GPU RMSNorm uses `rsqrtf`, so only RMSNorm'ed GPU paths carry a tolerance.
3. **Safe API.** Every public GPU entry point validates shapes before launching. Shape limits
   (batch tile <= 16, K % 64) return actionable errors pointing to chunking, `pad_k`, or the CPU
   backend.
4. **No fabricated numbers.** Benchmarks only print what they timed; projections are produced by
   `tuning` / `roofline` and labeled as such; energy is reported unavailable without counters.

## Packed layouts

* **Natural**: weight `t` of a 16-weight word at bits `[2t+1:2t]`. Used by `.tgm`, the CPU
  decode, the FPGA RTL and `PackedTernary::from_natural`.
* **Permuted**: words reordered per 64-weight chunk so that one 16-byte `uint4` load per lane
  feeds the warp's dp4a pipeline without shuffles (see `pack::permuted_word_index` and the
  kernel comments in `kernels/ternary.cu`).

## `.tgm` container (format.rs)

`"TGEMM\0"`, `u16` version (1), `u32` tensor count, then per tensor: `u16` name length + UTF-8
name, `u8` kind (0 = ternary: `n`, `k`, row scales, natural words; 1 = f32 vector; 2 = UTF-8
text), and a trailing `u64` FNV-1a checksum over everything before it. The parser checks
lengths before allocating and never panics on malformed input (fuzzed: `fuzz/tgm_parse`).

## Feature gating

`cuda` gates everything that touches `cudarc` (`gpu`, `signal`, `autotune`, the GPU parts of
`looped`). `--no-default-features` builds a portable CPU-only crate. `avx2` / `avx512` gate the
x86 SIMD code (runtime-detected inside). `convert` and `nvml` gate their optional dependencies.

## Testing layers

| layer | where |
|---|---|
| unit / doc | `src/**` `#[cfg(test)]` |
| bit-exactness between backends | `tests/cpu_backend.rs`, `tests/cpu_reference.rs`, `tests/kernel_emulation.rs`, `tests/mlp.rs` |
| properties | `tests/properties.rs` (proptest) |
| fuzzing | `fuzz/` (cargo-fuzz, nightly; CI smoke) |
| GPU (skipped without a device) | `tests/gpu.rs`, `tests/looped_gpu.rs` |
| NVRTC / ptxas (no GPU) | `tests/nvrtc_sm120.rs`, `compile_check` |
| hardware | `hw/tb` (Icarus, Verilator), `hw/hls` (g++) |
| Python | `bindings/python/tests` (pytest) |
