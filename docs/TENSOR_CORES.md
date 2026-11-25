# Tensor-core path (sm_80+)

A selectable backend for ternary GEMV/GEMM that unpacks the 2-bit natural-layout weights into int8 or fp16 fragments and issues PTX `mma.sync.aligned.m16n8k16` on NVIDIA tensor cores. It is written in the style of [TriRun](https://github.com/NolanoOrg/TriRun) and [BitBLAS](https://github.com/microsoft/BitBLAS) W2A8 kernels.

> **No GPU number on this path has been measured.** The kernels compile with NVRTC and assemble with `ptxas` for sm_80, sm_89 and sm_120. The box this was written on has no GPU, so any “8× vs fp16” figure below is the **theoretical bandwidth ceiling** at batch 1 (2-bit vs 16-bit weight traffic), not a result. TriRun and BitBLAS are prior art that reached near that ceiling on GEMV. Run `./scripts/bench.sh --tensor-cores` on a real sm_80+ GPU to fill the numbers in.

## What it does

Weights stay in the existing *natural* 2-bit packed layout: 16 codes per `u32`, code `c = w + 1` in `{0,1,2}`, weight `k` at word `k/16`, bits `2*(k % 16)`. (The older `dp4a` kernels use the permuted layout; both layouts are already stored on the device.)

Each warp owns a 16-row output tile (the M dimension of `m16n8k16`). Batch is tiled by 8 (the N dimension). K is reduced in steps of 16. On every K-tile:

1. The block stages a 16-wide slice of activations in shared memory.
2. Each lane unpacks one natural word per owned row into an A fragment in registers (`w = c - 1`, so the MMA sees `{-1,0,+1}`).
3. The matching activation nibble is packed into a B fragment.
4. The warp issues `mma.sync`:
   * W2A8: `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`
   * W2A16: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
5. The epilogue matches the CPU reference: `y = acc * row_scale / act_scale` (+ residual) for W2A8, and `y = acc * row_scale` (+ residual) for W2A16.

`K` must still be a multiple of 64 and `batch` in 1..=16 per launch, same as the rest of the crate. `N` need not be a multiple of 16; out-of-range rows are padded with zero weights and not written.

## How to select it

```rust
use ternary_gemm::gpu::{Gpu, Launch};

let gpu = Gpu::new(0)?;
assert!(gpu.supports_tensor_cores()); // sm_80+, kernels compiled
gpu.quantize(&x, None, 1e-5, &mut acts)?;
gpu.gemm_i8(&w, &acts, None, &mut out, Launch::tensor_core(4))?;
gpu.gemm_f16(&w, &x, batch, None, &mut out, Launch::tensor_core(4))?;
```

`Launch::backend` is `Backend::Dp4a` by default, so existing callers are unchanged. The fused `gemm_i8_fused` path is still dp4a-only; use `quantize` + `gemm_i8` for tensor cores. `rows_per_warp` is ignored on this backend (the MMA M tile is 16).

The runtime autotuner (`cargo run --release --bin autotune`) adds tensor-core candidates (`1, 2, 4, 8` warps) whenever `Gpu::supports_tensor_cores()` is true, times them against the dp4a search space, and records the winner with a `backend` column (`dp4a` or `tc`) in the CSV cache. Older 10-column cache files still load as `dp4a`.

## Theoretical ceiling (not a measurement)

At batch 1 the working set is a GEMV: each output row reads its weights once, and those bytes dominate. fp16 weights move 2 bytes per element; 2-bit ternary weights move 0.25 bytes, which is **exactly 8× less weight memory traffic**. That 8× is a bandwidth *ceiling*, not a prediction:

* Activations, outputs and per-row scales shrink the net ratio slightly below 8.
* A GEMV of a few microseconds sits close to kernel-launch overhead.
* At large batch the problem becomes compute-bound on the tensor cores, and a well-tuned fp16 tensor-core GEMM can catch up. The 8× figure is specifically the batch-1, weight-bandwidth case.

[TriRun](https://github.com/NolanoOrg/TriRun) and [BitBLAS](https://github.com/microsoft/BitBLAS) are the prior art that reached near this ceiling on GEMV. This repo has not yet measured the same thing.

## What *has* been measured (CPU only)

On the build machine (Intel Xeon cloud VM, 8 vCPUs, one core pinned, AVX-512 VNNI for the ternary path), the small signal model `256-1024-1024-16` ran in **18.30 µs** (ternary SIMD) versus **147.78 µs** (dense f32 AVX2), **8.1×**. That is a CPU result, reported in the [README](../README.md) and [`results/20261122-162423-cpu-latency-avx512.json`](../results/20261122-162423-cpu-latency-avx512.json). It is not a GPU tensor-core number.

The tensor-core *algorithm* is checked on the CPU: `tests/tensor_core.rs` unpacks natural-layout words into the same m16n8k16 fragments the kernel uses and compares a tiled GEMM against the scalar reference. The int8 path is bit-exact. The GPU tests in `tests/gpu.rs` run the same check on device and are skipped when no GPU (or no sm_80+) is present.

## Build and compile

The kernels live in [`kernels/tensor_core.cu`](../kernels/tensor_core.cu). They are self-contained (no `#include`) so NVRTC needs no CUDA headers. `Gpu::new` compiles them only for compute capability 8.0 and newer; older devices keep the dp4a kernels only.

```bash
# NVRTC + ptxas, no GPU required (needs libnvrtc; ptxas optional via PTXAS=...)
cargo run --release --bin compile_check -- compute_80 compute_89 compute_120

# On a machine with nvcc:
nvcc -std=c++17 -ptx -arch=compute_80  -o tensor_core_80.ptx  kernels/tensor_core.cu
```

A full correctness + timing run on a GPU:

```bash
./scripts/bench.sh --tensor-cores
# or a shorter sweep:
./scripts/bench.sh --quick --tensor-cores
```

## Further reading

* Short explainer video: [`docs/media/tensor-core-explainer.mp4`](media/tensor-core-explainer.mp4)
* Technical report: [`docs/reports/tensor-core-ternary.md`](reports/tensor-core-ternary.md)
