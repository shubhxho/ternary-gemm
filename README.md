# ternary-gemm

Fused **ternary matmul kernels for LLM decode on NVIDIA GPUs**, written in Rust and CUDA. The weights follow BitNet b1.58: each one is in {-1, 0, +1} and stored in 2 bits.

* **W2A8**: 2-bit weights × int8 activations using BitNet's per-token absmax scheme. It runs as a GEMV for batch 1 and a small-batch GEMM for batch 1–16. Activation quantization and an optional RMSNorm can be **fused into the prologue**, and the scale and residual add are **fused into the epilogue**.
* **W2A16**: 2-bit weights × fp16 activations, accumulated in fp32.
* A safe Rust API on top of [`cudarc`](https://crates.io/crates/cudarc). The kernels are compiled at runtime with NVRTC, so **no CUDA toolkit or nvcc is needed to build**.
* A pure-Rust CPU reference. On the int8 path the GPU output is designed to be **bit-identical** to it.
* Benchmarks against **cuBLAS FP16** and a **naive ternary kernel**, timed with CUDA events.

> **Status: no GPU numbers have been measured yet.** The kernels compile with NVRTC 12.8, and `ptxas` assembles them for sm_80, sm_86, sm_89, sm_90 and sm_120. The CPU reference and a lane-by-lane emulation of the kernel index math pass on the CPU. The machine this was written on has **no GPU**, so every GPU latency or bandwidth figure below is either **NOT MEASURED** or explicitly a **theoretical estimate**. Run `./scripts/bench.sh` on a real GPU to fill them in.

## Quick start (on the GPU machine, e.g. an RTX 5060)

```bash
git clone https://github.com/shubhxho/ternary-gemm && cd ternary-gemm
./scripts/bench.sh            # build -> GPU correctness tests -> benchmarks -> results/<time>-<gpu>.md
./scripts/bench.sh --quick    # batches 1 and 16 only
```

You need an NVIDIA driver (RTX 50-series needs driver ≥ 570 and CUDA ≥ 12.8 for sm_120) and Rust from [rustup](https://rustup.rs). You also need `libnvrtc` and `libcublas` from CUDA 12.x. If they aren't installed, the script pip-installs the CUDA 12.8 runtime wheels into `.venv-cuda/`. The script fails instead of skipping if no GPU is found (it sets `TERNARY_REQUIRE_GPU=1` for the tests).

On Windows (native, with CUDA Toolkit 12.8+ installed):

```powershell
$env:TERNARY_REQUIRE_GPU=1; cargo test --release -- --test-threads=1
cargo run --release --bin bench > results\my-gpu.md
```

Options: `cargo run --release --bin bench -- --batches 1,4 --layers 30 --no-verify`. Set `PEAK_GBPS=448` to override the peak bandwidth used for the "% of peak" column. By default it comes from the driver's memory-clock and bus-width attributes, which can be wrong on GDDR7 boards. The RTX 5060 spec is 448 GB/s: 128-bit bus at 28 Gbps.

Checks that need no GPU:

```bash
cargo test --release                          # CPU reference + kernel-index emulation (GPU tests auto-skip)
PTXAS=/path/to/ptxas cargo run --release --bin compile_check   # NVRTC + ptxas for sm_80..sm_120: registers/spills
```

## Library use

```rust
use ternary_gemm::{gpu::{Gpu, Launch}, pack::{quantize_absmean, PackedTernary}};

let (w_t, scales) = quantize_absmean(&w_f32, n, k, /*per_row=*/ false); // BitNet absmean
let packed = PackedTernary::from_ternary(&w_t, n, k, scales)?;           // K % 64 == 0
let gpu = Gpu::new(0)?;                     // compiles kernels for this GPU via NVRTC
let w = gpu.upload_weights(&packed)?;
let x = gpu.upload_f16(&x_host)?;           // [batch][k] fp16
let mut y = gpu.alloc_f16(batch * n)?;
// one launch: RMSNorm(gamma) -> absmax int8 quant -> ternary GEMV -> * scale -> + residual -> fp16
gpu.gemm_i8_fused(&w, &x, batch, Some(&gamma), 1e-5, Some(&residual), &mut y, Launch::default())?;
```

Each call checks every buffer length and shape before launching, so bad input returns an error instead of reading or writing out of bounds.

## Design

### Weight format

* Each weight `w ∈ {-1,0,+1}` is stored as **code `c = w + 1 ∈ {0,1,2}`**, 2 bits each, 16 per `u32`, with one f32 scale per output row. BitNet's per-tensor scale is the case where every row has the same scale.
* **In-word interleave:** weight `16·kb + 4j + i` sits at bits `[8i + 2j, +2)`. So `(word >> 2j) & 0x03030303` gives four consecutive weights as four bytes, and the whole decode is **one shift plus one AND per 4 weights**, with no lookup table. Those bytes feed straight into `__dp4a` against four consecutive int8 activations.
* **Offset trick:** with codes in {0,1,2}, `Σ c·x = Σ w·x + Σ x`. The kernel subtracts the per-token `Σ x`, which is computed once during quantization. Codes therefore never need sign handling.
* **Chunk permutation:** a row is split into chunks of 128 words (2048 weights; the last chunk can be shorter). In a chunk of `Wc` words with `L = Wc/4` lanes, the word for k-block `q·L + lane` is stored at offset `4·lane + q`. Each lane makes **one 128-bit load per row per chunk**, so the warp reads 512 contiguous bytes. For each `q`, the active lanes read a **contiguous** run of activations, which means coalesced global loads and conflict-free shared-memory reads in the fused kernel.

Microsoft's BitNet GPU kernel uses the same basic decode-to-int8-then-dp4a idea with a different interleave; see the novelty notes below.

### Kernels (`kernels/ternary.cu`)

| kernel | what it does |
|---|---|
| `quantize_i8` | One block per token. Optional RMSNorm, then absmax, `s = 127/absmax`, `q = clamp(rint(x·s))` and `Σq`. |
| `ternary_i8_b{BT}_r{R}` | Main W2A8 loop. One warp computes `R ∈ {1,2,4}` rows for up to `BT ∈ {1,2,4,8,16}` tokens. Weights are decoded once into registers and reused across the batch. 16 `dp4a` per lane per row per token per chunk. Warp-shuffle reduction. Epilogue `(acc − Σq) / s · scale (+ residual)` with explicit `__fdiv_rn`, `__fmul_rn` and `__fadd_rn`, so there is no FMA contraction and the result is bit-exact against the CPU. |
| `ternary_i8_fused_b{BT}_r{R}` | Single launch. Each block re-quantizes the activations (fp16 → RMSNorm → int8) into shared memory, then runs the same loop. It trades redundant L2 reads of the small activation vector for one fewer launch and one fewer global round trip. Needs `batch·K ≤ 47 KiB` of shared memory. |
| `ternary_f16_b{BT}_r{R}` | W2A16. Each code byte is spliced under `0x4B000000`, giving the float `2²³ + c`, and `2²³ + 1` is subtracted, so decoding a weight is one `PRMT` plus one `FADD`. fp32 FMA accumulation. |
| `naive_i8`, `naive_f16` | Baselines: one thread per output, natural 2-bit layout, scalar decode and multiply, uncoalesced. |

The batch tile and rows-per-warp are template instantiations, 51 kernels in total. The benchmark sweeps `R` and reports the best one. The library default is a heuristic that picks the largest `R` that still gives at least 4 blocks per SM.

From `ptxas -v` (CUDA 12.8; full output in [`docs/compile_check_output.txt`](docs/compile_check_output.txt)): on sm_120, `ternary_i8_b1_r{1,2,4}` uses 38, 40 and 48 registers and `ternary_i8_b16_r{1,2,4}` uses 62, 108 and 204. Nothing spills on sm_90 or sm_120. On sm_80, sm_86 and sm_89, two variants spill up to 24 bytes (`ternary_f16_b4_r2`, plus `ternary_i8_fused_b2_r2` on sm_80).

### Why NVRTC at runtime rather than nvcc in `build.rs`

The kernels are compiled at first use, for the exact GPU (`compute_XY` of the device), then JIT-compiled to SASS by the driver. If the installed NVRTC doesn't know the GPU's architecture yet, the code falls back to older PTX targets. So `cargo build` works on any machine, even without CUDA, and the same binary targets sm_80, sm_120 and anything newer. The kernel file has **no `#include`**: fp16 is handled as raw bits with PTX `cvt`, so NVRTC needs no CUDA headers on disk.

### Correctness

* `tests/cpu_reference.rs`: the reference against direct dot products, BitNet quantization properties, and absmean weight quantization.
* `tests/kernel_emulation.rs`: a lane-by-lane Rust transliteration of both CUDA main loops (chunking, permutation, dp4a byte lanes, PRMT decode), checked against the reference at K = 64, 2048, 2560, 4096, 6912 and 11008. It **runs on the CPU**.
* `tests/gpu.rs`: GPU against CPU for shapes 2560×2560, 640×2560, 6912×2560, 2560×6912, 11008×4096 and 4096×11008 (N×K), at batches 1, 3 and 16, with and without residual, for every `R`:
  * int8 quantization and all W2A8 kernels, including the naive one: **bit-exact**.
  * Fused RMSNorm: within 1% of max |y|. The GPU uses `rsqrtf` and a different summation order, so the int8 rounding of a few elements can differ by 1.
  * W2A16 and cuBLAS: within 0.2% and 1% of max |y|.
* The benchmark re-runs a correctness check before timing and labels any mismatch.

## Results

### Measured on a GPU

**NOT MEASURED.** No GPU was available where this was built: no `nvidia-smi`, no `libcuda`. Run `./scripts/bench.sh` and commit the report from `results/`.

The report contains:
* A correctness table.
* A per-shape table for cuBLAS FP16, naive ternary, W2A8 GEMV only, W2A8 quantize + GEMV, W2A8 fused and W2A16, at batches 1, 2, 4, 8 and 16. It gives latency in µs, effective GB/s, % of peak DRAM bandwidth and speedup over cuBLAS.
* A full **BitNet-b1.58-2B linear-layer stack**: 30 layers × (q, k, v, o, gate, up, down), comparing the ternary path with cuBLAS FP16 as linear-only tokens/s.

For honest DRAM numbers, weights rotate through enough copies to exceed 3× the L2 size, the layer stack allocates all 30 layers' weights, and buffers are filled with random data, not zeros.

### Measured on the CPU (sanity only, not a GPU number)

When no GPU is found, `cargo run --release --bin bench` times the scalar CPU reference. On the 8-vCPU build box it took 16.4 ms for 2560×2560 and 42.5 ms for 2560×6912 at batch 1, single-threaded. This is an oracle and was deliberately left unoptimized.

### Theoretical estimate (NOT measured): DRAM roofline at batch 1, RTX 5060 at 448 GB/s

`t_min = bytes / 448 GB/s`. fp16 moves 2 bytes per weight and 2-bit ternary moves 0.25 bytes per weight, so **weight traffic is exactly 8× smaller**. Activations, outputs and per-row scales bring the total ratio slightly under 8.

| shape (N × K) | fp16 bytes | ternary bytes | t_min fp16 (µs) | t_min ternary (µs) | ideal ratio |
|---|---|---|---|---|---|
| 2560 × 2560 | 13.12 MB | 1.66 MB | 29.3 | 3.7 | 7.91× |
| 640 × 2560 | 3.28 MB | 0.42 MB | 7.3 | 0.9 | 7.84× |
| 6912 × 2560 | 35.41 MB | 4.47 MB | 79.0 | 10.0 | 7.92× |
| 2560 × 6912 | 35.41 MB | 4.45 MB | 79.0 | 9.9 | 7.95× |
| 11008 × 4096 | 90.21 MB | 11.35 MB | 201.4 | 25.3 | 7.95× |
| 4096 × 11008 | 90.21 MB | 11.32 MB | 201.4 | 25.3 | 7.97× |

For the whole BitNet-2B linear stack, 30 × 69.5 M weights are 4.17 GB in fp16 against 0.52 GB in ternary. That puts the bandwidth ceiling at ≈ 107 tokens/s for fp16 and ≈ 855 tokens/s for ternary at batch 1 (`1 / (bytes / 448 GB/s)`).

These are **upper bounds**, not predictions. Several things will keep the real speedup below 8×:
* Small ternary GEMVs, at 1–10 µs ideal, are close to **kernel-launch overhead** of a few µs each. The layer stack issues 11 launches per layer; CUDA graphs would help and aren't implemented yet.
* The tail chunk, where K is not a multiple of 2048, idles some lanes.
* cuBLAS is near roofline on large shapes.
* On small shapes where fp16 weights fit in L2, cuBLAS also benefits if the cache isn't flushed. The benchmark prevents this by rotating weight copies.

Microsoft reports 1.3×–3.6× over BF16 for its W2A8 GEMV on an A100. Expect something in that range, not 8×, until it's measured.

## Novelty check: what already exists (searched Nov 2026)

Honest summary: **ternary GPU GEMV with 2-bit packing and dp4a is not new.** What this repo adds is the packaging and the fusion, not the core trick.

| project | what it is | how this differs |
|---|---|---|
| [microsoft/BitNet `gpu/`](https://github.com/microsoft/BitNet/tree/main/gpu) | CUDA W2A8 **GEMV** for BitNet-b1.58-2B: interleaved 2-bit packing, decode to int8, `dp4a`, bf16 output, called from PyTorch via ctypes. **M = 1 only, hard-coded shapes**, and activation quantization done separately in PyTorch. A100 results: 13–31 µs, 1.3–3.6× over BF16. | Same core idea. Here it's Rust-native (cudarc + NVRTC, no PyTorch or nvcc), handles **any N and any K % 64**, **batch 1–16**, fuses **RMSNorm + absmax quantization into the prologue** and **residual into the epilogue**, uses offset codes so decode is shift+mask only, adds a W2A16 path, and checks bit-exactly against a CPU oracle. |
| [microsoft/BitBLAS](https://github.com/microsoft/BitBLAS) (Ladder, OSDI '24) | TVM/TIR-generated mixed-precision GEMV/GEMM, including INT2×INT8 for BitNet. Reports 8× (GEMV) and 2× (GEMM) over cuBLAS FP16 on an A100. Python/TVM toolchain. | No TVM or tuning step; a small, hand-written, auditable kernel set callable from Rust. BitBLAS is likely the stronger performance baseline and isn't benchmarked here yet. |
| llama.cpp `TQ1_0` / `TQ2_0` | Ternary GGUF formats, CPU kernels merged in 2024. CUDA `TQ2_0` PRs ([#11183](https://github.com/ggml-org/llama.cpp/pull/11183), draft [#30278](https://github.com/ggml-org/llama.cpp/pull/30278)) were still open, unmerged, when searched. | Standalone library, not an inference engine. No GGUF loader yet. |
| [microsoft/T-MAC](https://github.com/microsoft/T-MAC) | Lookup-table low-bit GEMM, **CPU only**. | GPU. |
| Rust: [`bitnet-quantize`](https://docs.rs/bitnet-quantize) (CubeCL kernels), [`tritium-candle`](https://docs.rs/tritium-candle) (CPU reference op for candle), [`unsloth-rs`](https://docs.rs/unsloth-rs) (ternary GPU path still a TODO, falls back to CPU), [`mistralrs-quant`](https://docs.rs/mistralrs-quant) (fp16/bf16 GEMV, not ternary) | There is Rust BitNet work, but as far as I found there is no hand-tuned CUDA W2A8 decode kernel with fused norm and quantization and published cuBLAS comparisons. | That gap is what this repo targets. I didn't run or benchmark those crates. |

## Limitations

* **No GPU measurements yet**, as described above.
* `K` must be a multiple of 64. Batch is limited to 1–16, which is decode-oriented; there is no tensor-core path for prefill.
* Scales are per row (or per tensor), with no per-group scales.
* The fused kernel needs `batch·K ≤ 47 KiB` of shared memory, for example batch ≤ 4 at K = 11008. Above that, use `quantize` plus `gemm_i8`.
* There is no split-K, so very small `N` with very large `K` underfills the GPU.
* There are no CUDA graphs; launch overhead isn't hidden in the layer-stack benchmark.
* The layer-stack benchmark is linear layers only. It leaves out attention, the activation function, embeddings and lm_head.
* Not yet compared against BitBLAS, the BitNet GPU kernel, or llama.cpp on the same GPU.

## Layout

```
kernels/ternary.cu        all CUDA kernels (NVRTC, no includes)
src/pack.rs               absmean quantization, 2-bit packing (permuted + natural layouts)
src/reference.rs          CPU reference (oracle)
src/gpu.rs                safe cudarc wrapper, cuBLAS baseline, CUDA-event timer
src/bin/bench.rs          benchmark -> Markdown report
src/bin/compile_check.rs  NVRTC + ptxas register/spill report (no GPU needed)
tests/                    cpu_reference, kernel_emulation (CPU), gpu (needs GPU)
scripts/bench.sh          one-shot build + verify + benchmark
```

License: MIT.
