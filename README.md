# ternary-gemm

Fused **ternary matmul kernels for LLM decode on NVIDIA GPUs**, written in Rust and CUDA. The weights follow BitNet b1.58: each one is in {-1, 0, +1} and stored in 2 bits.

* **W2A8**: 2-bit weights × int8 activations using BitNet's per-token absmax scheme. It runs as a GEMV for batch 1 and a small-batch GEMM for batch 1–16. Activation quantization and an optional RMSNorm can be **fused into the prologue**, and the scale and residual add are **fused into the epilogue**.
* **W2A16**: 2-bit weights × fp16 activations, accumulated in fp32.
* A safe Rust API on top of [`cudarc`](https://crates.io/crates/cudarc). The kernels are compiled at runtime with NVRTC, so **no CUDA toolkit or nvcc is needed to build**.
* A pure-Rust CPU reference. On the int8 path the GPU output is designed to be **bit-identical** to it.
* Benchmarks against **cuBLAS FP16** and a **naive ternary kernel**, timed with CUDA events.

> **Status: no GPU numbers have been measured yet.** The kernels compile with NVRTC 12.8, and `ptxas` assembles them for sm_80, sm_86, sm_89, sm_90 and sm_120. The CPU reference and a lane-by-lane emulation of the kernel index math pass on the CPU. The machine this was written on has **no GPU**, so every GPU latency or bandwidth figure below is either **NOT MEASURED** or explicitly a **theoretical estimate**. Run `./scripts/bench.sh` on a real GPU to fill them in.

![Fused kernel data flow](docs/img/fused_kernel_dataflow.png)

## Quick start (on the GPU machine, e.g. an RTX 5060)

```bash
git clone https://github.com/shubhxho/ternary-gemm && cd ternary-gemm
./scripts/bench.sh            # build -> GPU tests -> benchmarks -> results/<time>-<gpu>.{md,csv} + measured charts results/*.png
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

To hide launch overhead, record a whole decode step once and replay it as a CUDA graph:

```rust
let graph = gpu.capture(|| {            // launches only: no allocation, no host copies, no sync
    gpu.gemm_i8_fused(&w, &x, batch, Some(&gamma), 1e-5, None, &mut y, Launch::default())?;
    Ok(())
})?;
graph.launch()?;                        // replays every launch in one call; buffer addresses are baked in
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
| `ternary_i8_fused_b{BT}_r{R}` | Single launch. Each block re-quantizes the activations (fp16 → RMSNorm → int8) into shared memory, then runs the same loop. It trades redundant L2 reads of the small activation vector for one fewer launch and one fewer global round trip. Shared memory needed is `batch·K` bytes. At startup the fused kernels opt in to the device's large dynamic shared-memory carve-out (`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN − 1 KiB`, ≈ 99 KiB on sm_86/89/120, ≈ 163 KiB on sm_80, ≈ 227 KiB on sm_90). If the driver refuses, the limit falls back to 47 KiB. The limit in use is `gpu.info.fused_smem_limit`. |
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
* A full **BitNet-b1.58-2B linear-layer stack**: 30 layers × (q, k, v, o, gate, up, down), comparing the ternary path with cuBLAS FP16 as linear-only tokens/s. Each path runs twice, launched op by op and replayed as one **CUDA graph**.
* The same numbers as CSV (`--csv`). `scripts/plot_results.py` turns the CSV into charts titled "Measured on <GPU>": latency and % of peak bandwidth per shape, plus layer-stack tokens/s.

For honest DRAM numbers, weights rotate through enough copies to exceed 3× the L2 size, the layer stack allocates all 30 layers' weights, and buffers are filled with random data, not zeros.

### Measured on the CPU (sanity only, not a GPU number)

When no GPU is found, `cargo run --release --bin bench` times the scalar CPU reference. On the 8-vCPU build box it took 16.4 ms for 2560×2560 and 42.5 ms for 2560×6912 at batch 1, single-threaded. This is an oracle and was deliberately left unoptimized.

### Theoretical estimate (NOT measured): DRAM roofline at batch 1, RTX 5060 at 448 GB/s

![Weight bytes per matrix](docs/img/bytes_per_matrix.png)
![Roofline latency floor, theoretical](docs/img/roofline_latency_floor.png)

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

![Tokens/s ceiling, theoretical](docs/img/tokens_per_s_ceiling.png)

All three charts come from `scripts/make_doc_charts.py` and are pure arithmetic. They are labeled **Theoretical (not measured)**.

These are **upper bounds**, not predictions. Several things will keep the real speedup below 8×:
* Small ternary GEMVs, at 1–10 µs ideal, are close to **kernel-launch overhead** of a few µs each. The layer stack issues 11 launches per layer. The benchmark also times a CUDA-graph replay to measure how much of that overhead graphs remove.
* The tail chunk, where K is not a multiple of 2048, idles some lanes.
* cuBLAS is near roofline on large shapes.
* On small shapes where fp16 weights fit in L2, cuBLAS also benefits if the cache isn't flushed. The benchmark prevents this by rotating weight copies.

Microsoft reports 1.3×–3.6× over BF16 for its W2A8 GEMV on an A100. Expect something in that range, not 8×, until it's measured.

## Novelty check: has someone already done this? (searched Nov 2026)

**Verdict: yes, mostly.** A Rust + CUDA ternary (BitNet b1.58) decode engine with a fused RMSNorm → int8 quantization kernel, bit-exact checks against a scalar reference, and measured consumer-GPU numbers already exists: **[Quitetall/tritium](https://github.com/Quitetall/tritium)**. It is Apache-2.0, has a `tritium-cuda` crate (`rmsnorm_quant_i8`, TQ2_0 IMMA kernels, built with nvcc), and its authors report ~280–300 tok/s end-to-end for BitNet-2B4T on an RTX 4090 ([r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1vbf0nt/)). The core trick (2-bit pack → int8 → `dp4a`) is also in Microsoft's own BitNet GPU kernel. This repo is **not a first**.

What is narrower or different here:
* It is a **small standalone kernel crate**, not a whole inference engine.
* Kernels are compiled at runtime with **NVRTC**, so building needs no nvcc and no CUDA toolkit.
* RMSNorm and quantization are fused into **the same launch as the GEMV**, along with the residual add in the epilogue.
* It handles **any N, any K % 64, and batch 1–16**.
* It ships a cuBLAS-FP16 / naive / CUDA-graph **comparison harness** with CSV output and charts.
* Weakness: **no measurements yet**. Tritium and Cortex-Bitnet both publish real numbers.

| project | what it is | relation to this repo |
|---|---|---|
| [Quitetall/tritium](https://github.com/Quitetall/tritium) | Rust BitNet inference engine with CUDA kernels (`crates/tritium-cuda`): fused RMSNorm + int8 quantization, ternary IMMA/dp4a paths, bit-identical to a scalar reference. Reports ~280–300 tok/s on an RTX 4090. | **Closest prior art.** It does more (a full model) and is measured. This repo is a smaller, NVRTC-only kernel library with fused quant+GEMV in one launch. |
| [Tabakkui/Cortex-Bitnet](https://github.com/Tabakkui/Cortex-Bitnet) ([post](https://www.reddit.com/r/OpenAssistant/comments/1v8veh2/)) | Custom CUDA ternary engine (Python host, sm_86): fused FFN/RMSNorm, CUDA graphs, int8 lm_head. Reports 206–217 tok/s for BitNet-2B on an RTX 3070. PolyForm Noncommercial. | Same goals and also measured. Not Rust. |
| [microsoft/BitNet `gpu/`](https://github.com/microsoft/BitNet/tree/main/gpu) ([bitnet.cpp paper](https://arxiv.org/abs/2502.11880)) | CUDA W2A8 GEMV: 2-bit interleave → int8 → `dp4a`. M = 1 only, hard-coded shapes, quantization done separately in PyTorch. Reports 1.3–3.6× over BF16 on an A100. | Same core idea. This repo adds arbitrary shapes, batch 1–16, fused prologue and epilogue, and Rust with no nvcc. |
| [microsoft/BitBLAS](https://github.com/microsoft/BitBLAS) (Ladder, OSDI '24) | TVM-generated INT2×INT8 GEMV/GEMM. | Likely a stronger baseline. Not benchmarked here yet. |
| [Spectra-1.1 / TriRun](https://arxiv.org/abs/2506.23025), [TernaInfer](https://github.com/Gu-J/TernaInfer) (DAC 2026), [BITCOS](https://arxiv.org/abs/2609.16338) (Intel Xe2) | Research ternary kernels: FP16×INT2 tensor-core, FPGA/accelerator-style, Intel GPU. | Different hardware paths. |
| llama.cpp `TQ1_0` / `TQ2_0` | Ternary GGUF; CUDA `TQ2_0` PRs ([#11183](https://github.com/ggml-org/llama.cpp/pull/11183), draft [#30278](https://github.com/ggml-org/llama.cpp/pull/30278)) still unmerged. | Inference engine, not a library. |
| Other Rust work: [`bitnet-quantize`](https://docs.rs/bitnet-quantize) and `tritter-accel` (CubeCL), [nanochat-rs-ternary](https://github.com/architehc/nanochat-rs-ternary) (claims a CUDA dp4a decode path), [tidynest/bitnet-toy](https://github.com/tidynest/bitnet-toy) (cudarc + NVRTC training, cuBLAS int8), [burn](https://github.com/tracel-ai/burn) (absmean + packed 2-bit quantization), [candle #1956](https://github.com/huggingface/candle/issues/1956) (BitNet still open) | Rust ternary/BitNet pieces at various depths. | I didn't run or benchmark them. |
| [microsoft/T-MAC](https://github.com/microsoft/T-MAC) | Lookup-table low-bit GEMM, CPU only. | Different hardware. |

## Limitations

* **No GPU measurements yet**, as described above.
* `K` must be a multiple of 64. Batch is limited to 1–16, which is decode-oriented; there is no tensor-core path for prefill.
* Scales are per row (or per tensor), with no per-group scales.
* The fused kernel needs `batch·K` bytes of shared memory, up to the opt-in limit (≈ 99 KiB on consumer GPUs, so batch ≤ 9 at K = 11008). Above that, use `quantize` plus `gemm_i8`. A large carve-out lowers occupancy; whether fused still wins at big batch·K is unmeasured.
* There is no split-K, so very small `N` with very large `K` underfills the GPU.
* CUDA graph capture (`Gpu::capture`) is untested on hardware, like everything else on the GPU side. If capture fails, the benchmark prints `n/a` with the error and carries on.
* The layer-stack benchmark is linear layers only. It leaves out attention, the activation function, embeddings and lm_head.
* Not yet compared against BitBLAS, the BitNet GPU kernel, or llama.cpp on the same GPU.

## Layout

```
kernels/ternary.cu        all CUDA kernels (NVRTC, no includes)
src/pack.rs               absmean quantization, 2-bit packing (permuted + natural layouts)
src/reference.rs          CPU reference (oracle)
src/gpu.rs                safe cudarc wrapper, cuBLAS baseline, CUDA-event timer
src/bin/bench.rs          benchmark -> Markdown report (+ --csv)
src/bin/compile_check.rs  NVRTC + ptxas register/spill report (no GPU needed)
tests/                    cpu_reference, kernel_emulation (CPU), gpu (needs GPU)
scripts/bench.sh          one-shot build + verify + benchmark + measured charts
scripts/plot_results.py   bench CSV -> results/*.png ("Measured on <GPU>")
scripts/make_doc_charts.py  docs/img/*.png (theoretical charts + data-flow diagram)
```

License: MIT.
