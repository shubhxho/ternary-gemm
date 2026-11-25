# The ternary GEMV in other languages

The same W2A8 ternary GEMV is ported to six more languages, plus hand-written PTX. Each port
reads the **natural** layout (weight `k` at word `k/16`, bits `[2·(k%16), +2)`, code = w + 1),
the int8 activations with their `(s, Σq)`, and the per-row f32 scales. It must reproduce the
Rust reference **bit for bit**: the int32 accumulators, and the fp16 output bits of
`fp16_rn((acc − Σq) / s · scale)` with IEEE round-to-nearest division and multiplication.

The test vectors come from the Rust reference (`cargo run --example lang_vectors` writes
`langs/vectors/case0..4.txt`; the shapes are 1×64, 7×128, 64×256, 33×1088 and 256×4096,
including rows that are not multiples of 16 and K that is not a multiple of 256).
`make langs` (that is, `scripts/langs_check.sh`) regenerates them and runs every port whose
toolchain it finds. Toolchain paths come from the `ZIG`, `MOJO`, `JULIA`, `FUTHARK` and `PYTHON`
env vars (`PYTHON` needs `triton`, `halide`, `numpy`, `torch` and `pytest`). Missing toolchains
show as SKIP, not FAIL.

| language | files | technique | correctness vs Rust | 256×4096 time on this VM (measured, CPU) | GPU status |
|---|---|---|---|---|---|
| **Zig** 0.17.1 | `langs/zig/ternary.zig`, `check.zig` | `@Vector` SIMD: uniform shift over 16 words, `@select` add/sub, activations transposed once | PASS, 5/5 bit-exact | 54–65 µs (1 core, mean) | n/a |
| **Triton** 3.9.0 | `langs/triton/ternary_gemv.py`, `test_ternary_triton.py` | block of rows × K tiles, in-register decode, `div_rn` epilogue, `enable_fp_fusion=False` | PASS, 5/5 bit-exact in the Triton interpreter | not timed (interpreter only) | **compile-checked only**: lowers to PTX for sm_80 and sm_120 (`TRITON_INTERPRET=0 TRITON_PTX_CHECK=1`); never run on a GPU |
| **Mojo** 1.2.0 | `langs/mojo/ternary.mojo` | `SIMD` types, `comptime for` unroll, `select`, transposed activations | PASS, 5/5 bit-exact | 75–78 µs (mean) | n/a |
| **Julia** 1.13.4 | `langs/julia/ternary.jl` | `@simd ivdep` with lane accumulators | PASS, 5/5 bit-exact | 156–171 µs (mean) | n/a |
| **Futhark** 0.30.0 (nightly) | `langs/futhark/ternary.fut`, `check.py` | data-parallel `map`/`reduce`; built with the `c` and `multicore` backends | PASS, 5/5 bit-exact on both backends | c: 341–383 µs median; multicore (8 threads): 102–115 µs median | CUDA backend: host C compiles and the kernel source is generated without a GPU; device code is **not** compile-checked (it needs macros the runtime defines) |
| **Halide** 22.1.0 (Python) | `langs/halide/ternary_halide.py`, `test_ternary_halide.py` | `strict_float`; `rfactor` + vectorize, split with `GuardWithIf`, parallel rows | PASS, 6/6 tests bit-exact | 69 µs on 1 core, 38 µs at 2 threads, 19.2 µs at 8 threads | **compile-checked only**: the CUDA schedule compiles to an object for `cuda_capability_86` |
| **PTX** (hand-written) | `kernels/ternary_gemv.ptx` | permuted layout, `dp4a.u32.s32`, `shfl` butterfly, `div.rn`/`mul.rn`/`cvt.rn.f16` | logic mirrors the CUDA kernel; GPU test written (`tests/gpu_ext.rs`) but skipped | n/a | **compile-checked only**: ptxas assembles it for sm_61..sm_120 with 0 spills (30–34 registers) |

For reference, the Rust `Packed2` AVX-512 GEMV core does 256×4096 in **5.9 µs** p50 on one core
of the same VM. That figure covers only the integer GEMV on its own repacked layout: no
natural-layout decode and no epilogue, so it is not a like-for-like comparison. The ports
favor readability and bit-exactness over chasing that number.

Notes:
* Times are wall-clock on a shared cloud VM and are only meant to show orders of magnitude
  between ports.
* Halide: `HL_NUM_THREADS=1` segfaults in the Halide 22.1 runtime. To measure one core, pin
  with `taskset -c 3` instead.
* Triton: 2 of the 7 pytest cases (the PTX lowering checks) are skipped unless
  `TRITON_INTERPRET=0 TRITON_PTX_CHECK=1`.
* GitHub Actions is currently locked by billing, so these checks were run locally. The ruff
  step in CI covers `langs/`.
