//! Looped (weight-shared, recurrent-depth) ternary transformer.
//!
//! One pre-norm transformer block (causal multi-head attention + ReLU²-gated FFN, BitNet b1.58
//! style) whose seven projections are ternary [`PackedTernary`] matrices, applied `loops` times
//! with the **same weights**:
//!
//! ```text
//! h_0 = e                                  (e = input embeddings, [tokens][d_model] fp16)
//! for t in 1..=loops:
//!     u_t = inject(h_{t-1}, e)             (None: h | Add: h + e | Adapter: A·[h; e] + h)
//!     h_t = block(u_t)
//!     if halting and t >= min_loops and |h_t - h_{t-1}| / |h_t| < tol: stop
//! ```
//!
//! Every projection is a W2A8 GEMM with the exact semantics of [`crate::reference::gemm_i8`]
//! (absmax int8 activations, optional fused RMSNorm in the quantizer, optional fused residual), so
//! the same model runs on the CPU ([`CpuBackend`], bit-exact with the reference) or on the GPU
//! kernels ([`GpuBackend`], feature `cuda`). Attention and the elementwise FFN gate run on the
//! host in f32 in this version; see `docs/LOOPED.md`.
//!
//! Why loop a ternary model: the unique weights are one block, so a model with the effective depth
//! of `loops` blocks keeps only `1/loops` of the weight bytes, small enough to stay resident in L2 /
//! shared memory / on-chip SRAM (the FPGA path), which is where ternary's bandwidth and energy
//! advantage is largest.

use crate::cpu::CpuTernary;
use crate::pack::{quantize_absmean, PackedTernary, K_ALIGN};
use crate::reference::QuantizedActs;
use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// RMSNorm epsilon used by both norms in the block.
pub const RMS_EPS: f32 = 1e-5;

/// Shape of the shared block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
}

impl BlockConfig {
    /// `d_model` and `d_ff` must be multiples of the packing alignment (64) and `d_model` must
    /// split evenly into heads.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 || self.d_ff == 0 || self.n_heads == 0 {
            return Err("d_model, d_ff and n_heads must be > 0".into());
        }
        if !self.d_model.is_multiple_of(K_ALIGN) || !self.d_ff.is_multiple_of(K_ALIGN) {
            return Err(format!(
                "d_model={} and d_ff={} must be multiples of {K_ALIGN}",
                self.d_model, self.d_ff
            ));
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(format!(
                "d_model={} is not divisible by n_heads={}",
                self.d_model, self.n_heads
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

/// Which projection of the block a GEMM uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightId {
    Q = 0,
    K = 1,
    V = 2,
    O = 3,
    Gate = 4,
    Up = 5,
    Down = 6,
    Adapter = 7,
}

/// The weight-shared pre-norm block.
#[derive(Debug, Clone)]
pub struct TernaryBlock {
    pub cfg: BlockConfig,
    /// RMSNorm gain before attention (fused into the Q/K/V quantizer).
    pub attn_norm: Vec<f16>,
    /// RMSNorm gain before the FFN (fused into the gate/up quantizer).
    pub ffn_norm: Vec<f16>,
    pub wq: PackedTernary,
    pub wk: PackedTernary,
    pub wv: PackedTernary,
    pub wo: PackedTernary,
    pub w_gate: PackedTernary,
    pub w_up: PackedTernary,
    pub w_down: PackedTernary,
}

fn random_ternary(rng: &mut StdRng, n: usize, k: usize) -> Result<PackedTernary, String> {
    // N(0, 1/k)-ish init, then BitNet absmean ternarization (per-tensor scale).
    let std = (1.0 / k as f32).sqrt();
    let w: Vec<f32> = (0..n * k)
        .map(|_| {
            let (a, b): (f32, f32) = (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
            (a + b) * std * 1.2247 // sum of two U(-1,1) has variance 2/3
        })
        .collect();
    let (t, scale) = quantize_absmean(&w, n, k, false);
    PackedTernary::from_ternary(&t, n, k, scale).map_err(|e| e.to_string())
}

impl TernaryBlock {
    /// A randomly initialised block (deterministic for a given seed).
    pub fn random(cfg: BlockConfig, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let (d, f) = (cfg.d_model, cfg.d_ff);
        let ones = vec![f16::ONE; d];
        Ok(Self {
            cfg,
            attn_norm: ones.clone(),
            ffn_norm: ones,
            wq: random_ternary(&mut rng, d, d)?,
            wk: random_ternary(&mut rng, d, d)?,
            wv: random_ternary(&mut rng, d, d)?,
            wo: random_ternary(&mut rng, d, d)?,
            w_gate: random_ternary(&mut rng, f, d)?,
            w_up: random_ternary(&mut rng, f, d)?,
            w_down: random_ternary(&mut rng, d, f)?,
        })
    }

    /// The seven projections in [`WeightId`] order.
    pub fn weights(&self) -> [&PackedTernary; 7] {
        [
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
        ]
    }

    /// Packed weight bytes of the block (2 bits per weight + scales + norm gains).
    pub fn weight_bytes(&self) -> usize {
        self.weights()
            .iter()
            .map(|w| w.weight_bytes())
            .sum::<usize>()
            + 2 * 2 * self.cfg.d_model
    }

    /// One application of the block on the CPU (bit-exact with the reference GEMM).
    pub fn forward_cpu(&self, h: &[f16], tokens: usize) -> Vec<f16> {
        let mut be = CpuBackend::new(self, None);
        block_forward(&mut be, self, h, tokens).expect("CPU backend is infallible")
    }
}

/// How the input embeddings are re-injected at every loop iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inject {
    /// Plain weight tying: `h_t = block(h_{t-1})`.
    None,
    /// `h_t = block(h_{t-1} + e)` (deep-equilibrium / Universal-Transformer style).
    Add,
    /// `h_t = block(A·[h_{t-1}; e] + h_{t-1})` with a learned ternary adapter `A: 2d -> d`
    /// (the recurrent-depth design of Geiping et al. 2025).
    Adapter,
}

/// Adaptive depth: stop once the hidden state stops moving.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halting {
    /// Stop when `||h_t - h_{t-1}||_2 / ||h_t||_2 < tol`.
    pub tol: f32,
    /// Never stop before this many iterations.
    pub min_loops: usize,
}

/// Loop schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopConfig {
    /// Maximum number of block applications (>= 1).
    pub loops: usize,
    pub inject: Inject,
    pub halting: Option<Halting>,
}

/// Output of a looped forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopOutput {
    /// Final hidden state `[tokens][d_model]`.
    pub hidden: Vec<f16>,
    /// Iterations actually run (< `loops` only when halting fired).
    pub loops_run: usize,
    /// Relative state change after each iteration.
    pub deltas: Vec<f32>,
}

/// A weight-shared block applied up to `cfg.loops` times.
#[derive(Debug, Clone)]
pub struct LoopedTransformer {
    pub block: TernaryBlock,
    /// Input-injection adapter `[d_model][2 * d_model]`, present iff `cfg.inject == Adapter`.
    pub adapter: Option<PackedTernary>,
    pub cfg: LoopConfig,
}

impl LoopedTransformer {
    pub fn new(
        block: TernaryBlock,
        adapter: Option<PackedTernary>,
        cfg: LoopConfig,
    ) -> Result<Self, String> {
        block.cfg.validate()?;
        if cfg.loops == 0 {
            return Err("loops must be >= 1".into());
        }
        let d = block.cfg.d_model;
        match (cfg.inject, &adapter) {
            (Inject::Adapter, Some(a)) if a.n == d && a.k == 2 * d => {}
            (Inject::Adapter, Some(a)) => {
                return Err(format!(
                    "adapter must be [{d}][{}], got [{}][{}]",
                    2 * d,
                    a.n,
                    a.k
                ))
            }
            (Inject::Adapter, None) => return Err("Inject::Adapter needs an adapter".into()),
            (_, Some(_)) => return Err("adapter given but inject is not Adapter".into()),
            _ => {}
        }
        if let Some(h) = cfg.halting {
            if h.tol.is_nan() || h.tol < 0.0 || h.min_loops == 0 {
                return Err("halting needs tol >= 0 and min_loops >= 1".into());
            }
        }
        Ok(Self {
            block,
            adapter,
            cfg,
        })
    }

    /// Random block (and adapter if needed), deterministic for a given seed.
    pub fn random(block_cfg: BlockConfig, cfg: LoopConfig, seed: u64) -> Result<Self, String> {
        let block = TernaryBlock::random(block_cfg, seed)?;
        let adapter = match cfg.inject {
            Inject::Adapter => {
                let mut rng = StdRng::seed_from_u64(seed ^ 0x0ada_97e5);
                let d = block_cfg.d_model;
                Some(random_ternary(&mut rng, d, 2 * d)?)
            }
            _ => None,
        };
        Self::new(block, adapter, cfg)
    }

    /// Unique parameter bytes held in memory (one block + adapter, weights and scales).
    pub fn unique_weight_bytes(&self) -> usize {
        self.block.weight_bytes()
            + self
                .adapter
                .as_ref()
                .map_or(0, |a| a.weight_bytes() + 4 * a.row_scale.len())
    }

    /// Weight bytes an unrolled (non-shared) model of the same effective depth would need.
    pub fn unrolled_weight_bytes(&self) -> usize {
        self.unique_weight_bytes() * self.cfg.loops
    }

    /// Forward pass on the CPU.
    pub fn forward_cpu(&self, e: &[f16], tokens: usize) -> LoopOutput {
        let mut be = CpuBackend::new(&self.block, self.adapter.as_ref());
        self.forward(&mut be, e, tokens)
            .expect("CPU backend is infallible")
    }

    /// Forward pass on any [`Backend`].
    pub fn forward<B: Backend>(
        &self,
        be: &mut B,
        e: &[f16],
        tokens: usize,
    ) -> Result<LoopOutput, String> {
        let d = self.block.cfg.d_model;
        if tokens == 0 || e.len() != tokens * d {
            return Err(format!("input must be [tokens={tokens}][d_model={d}]"));
        }
        let mut h = e.to_vec();
        let mut deltas = Vec::with_capacity(self.cfg.loops);
        for t in 1..=self.cfg.loops {
            let u = match self.cfg.inject {
                Inject::None => h.clone(),
                Inject::Add => h
                    .iter()
                    .zip(e)
                    .map(|(a, b)| f16::from_f32(a.to_f32() + b.to_f32()))
                    .collect(),
                Inject::Adapter => {
                    let mut cat = Vec::with_capacity(tokens * 2 * d);
                    for tok in 0..tokens {
                        cat.extend_from_slice(&h[tok * d..(tok + 1) * d]);
                        cat.extend_from_slice(&e[tok * d..(tok + 1) * d]);
                    }
                    let a = be.quantize(&cat, tokens, 2 * d, None)?;
                    be.gemm(WeightId::Adapter, &a, Some(&h))?
                }
            };
            let next = block_forward(be, &self.block, &u, tokens)?;
            let delta = rel_change(&h, &next);
            deltas.push(delta);
            h = next;
            if let Some(halt) = self.cfg.halting {
                if t >= halt.min_loops && delta < halt.tol {
                    break;
                }
            }
        }
        Ok(LoopOutput {
            hidden: h,
            loops_run: deltas.len(),
            deltas,
        })
    }
}

/// `||b - a||_2 / max(||b||_2, 1e-12)` in f64.
fn rel_change(a: &[f16], b: &[f16]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x.to_f32() as f64, y.to_f32() as f64);
        num += (y - x) * (y - x);
        den += y * y;
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

/// Where the ternary GEMMs run. The block's control flow, attention and elementwise ops are
/// shared host code; a backend provides the two W2A8 primitives.
pub trait Backend {
    /// Quantized activations (host or device resident).
    type Acts;
    /// Absmax int8 quantization of `[tokens][k]`, with RMSNorm by `gamma` fused in if given.
    fn quantize(
        &mut self,
        x: &[f16],
        tokens: usize,
        k: usize,
        gamma: Option<&[f16]>,
    ) -> Result<Self::Acts, String>;
    /// `y = W·q / s * scale (+ residual)` as fp16 `[tokens][n]`.
    fn gemm(
        &mut self,
        w: WeightId,
        a: &Self::Acts,
        residual: Option<&[f16]>,
    ) -> Result<Vec<f16>, String>;
}

/// CPU backend: [`CpuTernary`] GEMMs (SIMD, bit-exact with [`crate::reference::gemm_i8`]).
pub struct CpuBackend {
    mats: Vec<Option<CpuTernary>>,
}

impl CpuBackend {
    pub fn new(block: &TernaryBlock, adapter: Option<&PackedTernary>) -> Self {
        let mut mats: Vec<Option<CpuTernary>> = block
            .weights()
            .iter()
            .map(|w| Some(CpuTernary::new(w)))
            .collect();
        mats.push(adapter.map(CpuTernary::new));
        Self { mats }
    }

    /// Split each GEMM's output rows across `threads` threads (results are identical).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.mats = self
            .mats
            .into_iter()
            .map(|m| m.map(|m| m.with_threads(threads)))
            .collect();
        self
    }
}

impl Backend for CpuBackend {
    type Acts = QuantizedActs;
    fn quantize(
        &mut self,
        x: &[f16],
        tokens: usize,
        k: usize,
        gamma: Option<&[f16]>,
    ) -> Result<QuantizedActs, String> {
        Ok(crate::cpu::quantize_acts(x, tokens, k, gamma, RMS_EPS))
    }
    fn gemm(
        &mut self,
        w: WeightId,
        a: &QuantizedActs,
        residual: Option<&[f16]>,
    ) -> Result<Vec<f16>, String> {
        let m = self.mats[w as usize]
            .as_ref()
            .ok_or_else(|| format!("{w:?} weights not loaded"))?;
        Ok(m.gemm_i8(a, residual))
    }
}

/// One pre-norm block: `h1 = h + O·attn(Q,K,V of norm(h))`, `out = h1 + Down·(relu(G)^2 * U)`.
pub fn block_forward<B: Backend>(
    be: &mut B,
    blk: &TernaryBlock,
    h: &[f16],
    tokens: usize,
) -> Result<Vec<f16>, String> {
    let c = blk.cfg;
    let d = c.d_model;
    let a = be.quantize(h, tokens, d, Some(&blk.attn_norm))?;
    let q = be.gemm(WeightId::Q, &a, None)?;
    let k = be.gemm(WeightId::K, &a, None)?;
    let v = be.gemm(WeightId::V, &a, None)?;
    let ctx = causal_attention(&q, &k, &v, tokens, c.n_heads, c.head_dim());
    let a = be.quantize(&ctx, tokens, d, None)?;
    let h1 = be.gemm(WeightId::O, &a, Some(h))?;
    let a = be.quantize(&h1, tokens, d, Some(&blk.ffn_norm))?;
    let g = be.gemm(WeightId::Gate, &a, None)?;
    let u = be.gemm(WeightId::Up, &a, None)?;
    let m: Vec<f16> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| {
            let r = g.to_f32().max(0.0);
            f16::from_f32(r * r * u.to_f32())
        })
        .collect();
    let a = be.quantize(&m, tokens, c.d_ff, None)?;
    be.gemm(WeightId::Down, &a, Some(&h1))
}

/// Causal softmax attention in f32, `[tokens][heads * head_dim]` in and out (fp16 storage).
pub fn causal_attention(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    tokens: usize,
    heads: usize,
    hd: usize,
) -> Vec<f16> {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    // Widen once (fp16 -> f32) instead of per dot product.
    let to32 = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
    let (q, k, v) = (to32(q), to32(k), to32(v));
    let mut out = vec![f16::ZERO; tokens * d];
    let mut p = vec![0f32; tokens];
    let mut acc = vec![0f32; hd];
    for hh in 0..heads {
        let off = hh * hd;
        for i in 0..tokens {
            let qi = &q[i * d + off..i * d + off + hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                let kj = &k[j * d + off..j * d + off + hd];
                let s: f32 = qi.iter().zip(kj).map(|(a, b)| a * b).sum();
                p[j] = s * scale;
                mx = mx.max(p[j]);
            }
            let mut z = 0f32;
            for pj in &mut p[..=i] {
                *pj = (*pj - mx).exp();
                z += *pj;
            }
            acc.iter_mut().for_each(|x| *x = 0.0);
            for j in 0..=i {
                let w = p[j] / z;
                for (a, vv) in acc.iter_mut().zip(&v[j * d + off..j * d + off + hd]) {
                    *a += w * vv;
                }
            }
            for (o, a) in out[i * d + off..i * d + off + hd].iter_mut().zip(&acc) {
                *o = f16::from_f32(*a);
            }
        }
    }
    out
}

/// GPU backend (feature `cuda`): every projection runs on the W2A8 kernels
/// ([`crate::gpu::Gpu::quantize`] + [`crate::gpu::Gpu::gemm_i8`]). Weights are uploaded once;
/// activations move per call, in token chunks of 16 (the kernels' batch tile).
///
/// Without RMSNorm the GEMM results are bit-identical to [`CpuBackend`]; the RMSNorm'ed quantizer
/// uses `rsqrtf` on the device, so Q/K/V and gate/up can differ in the last bit (see the GPU test).
#[cfg(feature = "cuda")]
pub struct GpuBackend<'g> {
    gpu: &'g crate::gpu::Gpu,
    mats: Vec<Option<crate::gpu::GpuTernary>>,
    launch: crate::gpu::Launch,
}

/// Device-resident activations for [`GpuBackend`], one entry per chunk of <= 16 tokens.
#[cfg(feature = "cuda")]
pub struct GpuChunks {
    chunks: Vec<crate::gpu::GpuActs>,
}

/// Tokens per GPU GEMM launch (the largest batch tile of the kernels).
#[cfg(feature = "cuda")]
pub const GPU_TOKEN_CHUNK: usize = 16;

#[cfg(feature = "cuda")]
impl<'g> GpuBackend<'g> {
    pub fn new(
        gpu: &'g crate::gpu::Gpu,
        block: &TernaryBlock,
        adapter: Option<&PackedTernary>,
    ) -> Result<Self, String> {
        let up = |w: &PackedTernary| gpu.upload_weights(w).map_err(|e| e.to_string());
        let mut mats = Vec::with_capacity(8);
        for w in block.weights() {
            mats.push(Some(up(w)?));
        }
        mats.push(adapter.map(up).transpose()?);
        Ok(Self {
            gpu,
            mats,
            launch: crate::gpu::Launch::default(),
        })
    }

    /// Override the kernel launch configuration (e.g. from the autotuner).
    pub fn with_launch(mut self, launch: crate::gpu::Launch) -> Self {
        self.launch = launch;
        self
    }
}

#[cfg(feature = "cuda")]
impl Backend for GpuBackend<'_> {
    type Acts = GpuChunks;
    fn quantize(
        &mut self,
        x: &[f16],
        tokens: usize,
        k: usize,
        gamma: Option<&[f16]>,
    ) -> Result<GpuChunks, String> {
        let e = |e: crate::gpu::Error| e.to_string();
        if x.len() != tokens * k {
            return Err("quantize: x must be [tokens][k]".into());
        }
        let g = gamma
            .map(|g| self.gpu.upload_f16(g))
            .transpose()
            .map_err(e)?;
        let mut chunks = Vec::new();
        for start in (0..tokens).step_by(GPU_TOKEN_CHUNK) {
            let t = GPU_TOKEN_CHUNK.min(tokens - start);
            let xd = self
                .gpu
                .upload_f16(&x[start * k..(start + t) * k])
                .map_err(e)?;
            let mut acts = self.gpu.alloc_acts(t, k).map_err(e)?;
            self.gpu
                .quantize(&xd, g.as_ref(), RMS_EPS, &mut acts)
                .map_err(e)?;
            chunks.push(acts);
        }
        Ok(GpuChunks { chunks })
    }
    fn gemm(
        &mut self,
        w: WeightId,
        a: &GpuChunks,
        residual: Option<&[f16]>,
    ) -> Result<Vec<f16>, String> {
        let e = |e: crate::gpu::Error| e.to_string();
        let m = self.mats[w as usize]
            .as_ref()
            .ok_or_else(|| format!("{w:?} weights not loaded"))?;
        let n = m.n;
        let mut y = Vec::new();
        let mut row = 0;
        for acts in &a.chunks {
            let t = acts.batch;
            let res = residual
                .map(|r| self.gpu.upload_f16(&r[row * n..(row + t) * n]))
                .transpose()
                .map_err(e)?;
            let mut out = self.gpu.alloc_f16(t * n).map_err(e)?;
            self.gpu
                .gemm_i8(m, acts, res.as_ref(), &mut out, self.launch)
                .map_err(e)?;
            y.extend(self.gpu.download(&out).map_err(e)?);
            row += t;
        }
        Ok(y)
    }
}
