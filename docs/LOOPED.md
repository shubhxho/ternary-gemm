# Looped ternary transformer

`ternary_gemm::looped` implements a **weight-shared, recurrent-depth** transformer on top of the
ternary W2A8 kernels: one pre-norm block applied `loops` times with the same weights.

```text
h_0 = e                                  e: input embeddings [tokens][d_model], fp16
for t in 1..=loops:
    u_t = inject(h_{t-1}, e)
    h_t = block(u_t)
    if halting and t >= min_loops and ||h_t - h_{t-1}|| / ||h_t|| < tol: stop
```

## The block

BitNet-b1.58-style pre-norm block, all seven projections ternary (`PackedTernary`, 2 bits per
weight, per-row scale):

```text
a   = quant(RMSNorm_attn(h))          int8 absmax, RMSNorm fused into the quantizer
q,k,v = Wq a, Wk a, Wv a
h1  = h + Wo quant(causal_attention(q, k, v))      residual fused into the GEMM epilogue
a2  = quant(RMSNorm_ffn(h1))
m   = relu(Wgate a2)^2 * (Wup a2)                  ReLU^2 gated FFN
out = h1 + Wdown quant(m)
```

Every projection is exactly one `quantize` + one `gemm_i8` with the semantics of
`reference::gemm_i8`, so the same block runs on the CPU (`CpuBackend`) or on the CUDA kernels
(`GpuBackend`). The `Backend` trait is the seam: two methods, `quantize` and `gemm`.

## Input injection

| mode | update | notes |
|---|---|---|
| `Inject::None` | `h_t = block(h_{t-1})` | plain weight tying (Universal Transformer without injection) |
| `Inject::Add` | `h_t = block(h_{t-1} + e)` | re-injects the input every step (deep-equilibrium style); keeps the fixed point tied to the input |
| `Inject::Adapter` | `h_t = block(A [h_{t-1}; e] + h_{t-1})` | learned ternary adapter `A: 2d -> d`, as in the recurrent-depth design of Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (2025) |

With `loops = 1` and `Inject::None` the model is exactly one standard block; the test
`one_loop_equals_one_block` checks this bit for bit.

## Adaptive halting

`Halting { tol, min_loops }` stops once the relative change of the hidden state falls below
`tol` (after at least `min_loops`). The output reports `loops_run` and the per-loop deltas.
This is a simple convergence criterion (no learned halting unit as in ACT); it gives
input-dependent depth at zero parameter cost, and the halted output equals the non-halting
output truncated at the same depth (tested).

## Why looping suits ternary

* **Unique weights shrink by `loops`x.** A model with the effective depth of `L` blocks stores
  one. At 2 bits per weight a BitNet-2B-sized block (hidden 2560, FFN 6912) is 17.4 MB, so a
  looped model's entire weight set fits in an RTX 5090's 96 MB L2 (see `docs/RTX5090.md`); a
  d_model 512 block is ~1 MB and fits in a CPU's L2.
* **Bandwidth was the bottleneck.** Ternary decode is memory bound (the roofline in
  `docs/RTX5090.md`); once the weights are on-chip, every extra loop is served from cache or
  SRAM instead of DRAM. Looping turns "more depth" from a DRAM cost into an on-chip cost.
* **Per-watt.** Off-chip DRAM access costs far more energy per byte than on-chip SRAM, so
  serving the repeated passes from on-chip memory is where the energy advantage is.
  `src/energy.rs` (RAPL / NVML) is the harness for measuring this on real hardware.
* **FPGA path.** A single ternary block fits in on-chip BRAM/URAM of mid-size FPGAs; the
  ternary dot unit in `hw/` needs only adders (no multipliers). Looping means the same
  resident weights and the same datapath are reused for every layer of effective depth.

## Measured on this machine (CPU only)

These are CPU measurements on the development VM (Intel Xeon, AVX2 + AVX-512 VNNI, 8 vCPUs,
single-threaded GEMMs). **No GPU numbers are claimed.**

Criterion (`cargo bench --bench cpu -- looped`), d_model 256, 4 heads, d_ff 1024, 8 tokens,
`Inject::Add`:

| loops | time per forward |
|---|---|
| 1 | 249 us |
| 4 | 989 us |
| 8 | 1.95 ms |

Scaling is linear in loops, as expected with no per-loop setup. Optimisations that landed with
this module (same machine, same bench): fast bit-exact activation quantizer (461 -> 302 us per
loop), f32-widened attention (35.5 -> 13.7 us for the attention step), 4-row register blocking
in the CPU GEMM (302 -> 249 us per loop).

`cargo run --release --example looped` (d_model 512, 8 heads, d_ff 2048, 16 tokens, random
weights): ~1.6-1.8 ms per loop; unique weights 1026 KiB vs 8208 KiB for the unrolled 8-block
equivalent (1154 vs 9232 KiB with the adapter). With random weights and `Inject::Add` the state
change per loop decays from 0.56 to 0.12 over 8 loops, and halting at `tol = 0.05` stopped after
20 loops. Random weights are not trained; these deltas only show the mechanism.

## GPU wiring and what is not done yet

`GpuBackend` runs every projection on the W2A8 kernels (`Gpu::quantize` + `Gpu::gemm_i8`),
weights uploaded once, activations in 16-token chunks. Attention and the ReLU^2 gate run on the
host in this version, so each GEMM round-trips activations over PCIe: it is a correctness path
(GPU vs CPU is tested: bit-exact GEMM without RMSNorm, within 2% relative L2 for the whole
looped forward because the device RMSNorm uses `rsqrtf`), not a fast path. Next steps:

* device-side attention + gate kernels so a loop never leaves the GPU, then capture one loop
  in a CUDA graph and replay it `loops` times;
* a persistent kernel that keeps the block's packed weights in shared memory / L2 across loops
  (the reason for looping in the first place);
* training: this crate does inference only; the weights here are random.
