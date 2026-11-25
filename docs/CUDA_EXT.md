# Extra CUDA kernels (`kernels/ternary_ext.cu`)

Batch-1 decode GEMV variants, on top of the default dp4a kernels in `kernels/ternary.cu`.
Every kernel accumulates `Σ code·q` exactly in int32 and runs the same fused epilogue
(dequant → identity / ReLU / ReLU² → optional fp16 residual → `cvt.rn.f16`). That makes
them all bit-identical to `cuda_ext::reference_gemv` on the CPU.

> **No GPU was available.** Every kernel below was **compile-checked only**, with NVRTC 12.8 +
> ptxas (sm_80, 86, 89, 90, 100, 120, 120a) and with nvcc 13.5 (sm_80/86/89/90/100/120), and
> they show 0 spills. Their logic is verified on the CPU through emulations in
> `src/cuda_ext.rs`. The GPU correctness and timing tests (`tests/gpu_ext.rs`,
> `examples/bench_ext.rs`) exist but were skipped here. Any speed figure on this page is an
> **estimate**, not a measurement.

| kernel | idea | when it should help (estimate) | sm_120 ptxas |
|---|---|---|---|
| `tgx_gemv_cpasync_r{1,2,4}` | 4-stage `cp.async` ring per warp; activations staged in shared memory; r rows per warp | large K, where load latency is not hidden by the plain kernel's ILP | 43 / 50 / 66 regs, 0 spills |
| `tgx_gemv_splitk_r2` + `tgx_splitk_finalize` | split K across blocks, int32 `atomicAdd`, then a finalize pass that runs the epilogue | short N / long K (e.g. 1024×16384: 64 row blocks cannot fill 170 SMs). `cuda_ext::splitk_factor` targets about 2 blocks per SM | 40 / 18 regs |
| `tgx_gemv_lut` | T-MAC-style lookup: 2 weights index a 16-entry int16 table built per activation pair (32 KiB smem) | mainly an experiment; dp4a is usually as fast on NVIDIA. Included for comparison | 48 regs, 32 KiB smem |
| `tgx_gemv_warpspec` | 1 producer warp (`cp.async`) + 4 consumer warps (dp4a) on a named-barrier ring (barriers 1–4) | the pre-Hopper warp-specialization pattern; helps when consumers stall on loads | 40 regs, 5 barriers |
| `tgx_gemv_batched` | tiled W2A8 GEMV for batch 1..16: one K-chunk of weights is staged and reused across the batch | decode with a small batch (speculative multi-request, prefill microbatch); amortizes weight traffic | compile-check only |
| `tgx_gemv_bulk_r2` | same math as `tgx_gemv_cpasync_r2`; weight chunks prefetched with `cp.async.bulk` (TMA-style) on sm_90+ / sm_120, `cp.async` 16B fallback on sm_80/86/89 | Hopper/Blackwell batch-1 GEMV, where a 512 B bulk copy per row per chunk should beat many 16 B copies | compile-check only |
| `ternary_gemv_ptx` (hand PTX, `kernels/ternary_gemv.ptx`) | hand-scheduled dp4a GEMV | readability and a baseline for the compiler | 30–34 regs (sm_61..120) |

CPU-side verification (`cargo test cuda_ext`, `cargo test pick_ext_kernel`):
* `emulate_splitk` checks that any split factor gives the same accumulators (int32 adds
  commute).
* `emulate_lut` checks that table lookups equal dp4a sums.
* `emulate_batched` checks that K-chunk tiling reused across a batch is bit-identical to
  the batched reference (`reference_gemm`) for batch 1..16, including residuals and ReLU/ReLU².
* `simulate_warpspec` models the producer/consumer barrier protocol and proves there is no
  deadlock and no slot reuse before it is consumed, for 1–39 chunks.
* `splitk_factor` is checked to fill the GPU without over-splitting, and the dynamic smem
  budgets (including the batched kernel at batch 16 and the bulk kernel at K=8192) are
  checked to fit the default 48 KiB.
* `pick_ext_kernel` is checked to prefer the batched path when batch > 1, split-K for
  short-N / long-K, and the bulk path on sm_90+ / sm_120.

## Bandwidth floor (estimate, not a measurement)

At batch 1 the GEMV has to read every weight once. With the RTX 5090's specified 1792 GB/s,
the lower bounds are:

| shape (N×K) | ternary bytes | ternary floor | fp16 floor | ceiling ratio |
|---|---|---|---|---|
| 4096×11008 | 11.3 MB | ≈ 6.3 µs | ≈ 50 µs | 8× |
| 2560×6912 | 4.4 MB | ≈ 2.5 µs | ≈ 20 µs | 8× |
| 1024×16384 | 4.2 MB | ≈ 2.3 µs | ≈ 19 µs | 8× |

These are DRAM floors; shapes that stay in L2 (96 MB on the 5090) can go below them, and launch
overhead (a few µs, or ~1 µs with CUDA graphs) comes on top. **8× vs fp16 is the theoretical
batch-1 ceiling**, which is why this repository does not claim "100×".

## Run on a GPU machine

```bash
scripts/gpu_bench.sh                               # CPU emulations always; device tests + timings if a GPU is visible
scripts/gpu_bench.sh --cpu-only                    # never touch the driver
cargo test --release --test gpu_ext                # bit-exact vs reference on the device
cargo run --release --example bench_ext            # µs and GB/s per kernel and shape (measured on that box)
./scripts/bench.sh --ext                           # same, written to results/<stamp>-ext.md
TERNARY_REQUIRE_NVRTC=1 cargo test --release --test nvrtc_sm120   # compile check only
```

No GPU number in this document is a measurement from the authoring machine (it has no NVIDIA GPU).
