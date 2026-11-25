//! Small ternary MLPs for latency-critical scoring (e.g. a market-feature vector in, a signal out).
//!
//! * [`TernaryMlp`]: the model (ternary weights + per-row scale + bias per layer, BitNet absmax
//!   int8 activations between layers) and its scalar [`TernaryMlp::forward_reference`] oracle.
//! * [`CpuEngine`]: an allocation-free CPU path (decoded int8 weights, SIMD dot products with
//!   runtime AVX2 dispatch on x86_64). Bit-identical to the reference.
//! * [`DenseF32Mlp`]: the same architecture with dense f32 weights, as a plain CPU baseline.
//!
//! The GPU engines that run the same model live in [`crate::signal`] and are bit-identical too:
//! every float operation below is a single IEEE round-to-nearest op that the CUDA kernel performs
//! in the same order (`__fdiv_rn`, `__fmul_rn`, `__fadd_rn`, `rintf`).

use crate::pack::quantize_absmean;
use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Largest layer width the single-block GPU kernel supports (shared-memory resident activations).
pub const MLP_MAX_DIM: usize = 4096;
/// Layer input widths must be a multiple of this (one packed `u32` holds 16 weights).
pub const MLP_K_ALIGN: usize = 16;

/// Activation applied after a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Identity = 0,
    Relu = 1,
    /// `relu(x)^2`, as used in BitNet b1.58 FFNs.
    Relu2 = 2,
}

#[inline(always)]
fn act(v: f32, a: Act) -> f32 {
    match a {
        Act::Identity => v,
        Act::Relu => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        Act::Relu2 => {
            let r = if v > 0.0 { v } else { 0.0 };
            r * r
        }
    }
}

/// One ternary layer: `y = act(((W_t · q(x)) / s) * scale + bias)`.
#[derive(Debug, Clone)]
pub struct MlpLayer {
    /// Output width.
    pub n: usize,
    /// Input width (multiple of 16).
    pub k: usize,
    /// Ternary weights `[n][k]`, values in `{-1, 0, 1}`.
    pub w: Vec<i8>,
    pub scale: Vec<f32>,
    pub bias: Vec<f32>,
    pub act: Act,
}

impl MlpLayer {
    /// Packed "k-major" words for the GPU kernel: word `w` of row `r` at `w * n + r`; inside a
    /// word, weight `16w + 4j + i` sits at bits `8i + 2j` as code `weight + 1`.
    pub fn kmajor_words(&self) -> Vec<u32> {
        let kw = self.k / 16;
        let mut out = vec![0u32; kw * self.n];
        for r in 0..self.n {
            for w in 0..kw {
                let mut word = 0u32;
                for j in 0..4 {
                    for i in 0..4 {
                        let c = (self.w[r * self.k + 16 * w + 4 * j + i] + 1) as u32;
                        word |= c << (8 * i + 2 * j);
                    }
                }
                out[w * self.n + r] = word;
            }
        }
        out
    }
}

/// A feed-forward stack of ternary layers.
#[derive(Debug, Clone)]
pub struct TernaryMlp {
    pub layers: Vec<MlpLayer>,
}

/// Per-token int8 quantization exactly as the kernels do it. Returns `(s, sum_q)`.
///
/// Dispatches to AVX2 (8 lanes, `vroundps` nearest-even) or SSE4.1 when available; every path is
/// bit-identical to [`quantize_scalar`], which the reference forward pass uses as the oracle.
pub fn quantize_into(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    assert!(q.len() >= x.len(), "q too short");
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected; lengths checked above.
        return unsafe { quantize_avx2(x, q) };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    if std::is_x86_feature_detected!("sse4.1") {
        // SAFETY: SSE4.1 detected (gives an inline round-to-nearest-even instead of a libm call).
        return unsafe { quantize_sse41(x, q) };
    }
    quantize_generic(x, q)
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[target_feature(enable = "sse4.1")]
unsafe fn quantize_sse41(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    quantize_generic(x, q)
}

/// Scalar quantization: the oracle every SIMD variant must match bit for bit.
pub fn quantize_scalar(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    quantize_generic(x, q)
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let p = x.as_ptr();
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut mv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        mv = _mm256_max_ps(mv, _mm256_and_ps(_mm256_loadu_ps(p.add(i)), abs_mask));
        i += 8;
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), mv);
    let mut m = 0.0f32;
    for v in lanes {
        m = m.max(v);
    }
    for v in &x[i..] {
        m = m.max(v.abs());
    }
    let s = 127.0f32 / m.max(1e-5);
    let sv = _mm256_set1_ps(s);
    let (lo, hi) = (_mm256_set1_ps(-128.0), _mm256_set1_ps(127.0));
    let conv = |j: usize| {
        let t = _mm256_round_ps(
            _mm256_mul_ps(_mm256_loadu_ps(p.add(j)), sv),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(t, lo), hi))
    };
    let mut sum = _mm256_setzero_si256();
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let qp = q.as_mut_ptr();
    let mut j = 0;
    while j + 32 <= n {
        let (a, b, c, d) = (conv(j), conv(j + 8), conv(j + 16), conv(j + 24));
        sum = _mm256_add_epi32(
            sum,
            _mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_add_epi32(c, d)),
        );
        let bytes = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
        _mm256_storeu_si256(
            qp.add(j) as *mut __m256i,
            _mm256_permutevar8x32_epi32(bytes, perm),
        );
        j += 32;
    }
    let mut parts = [0i32; 8];
    _mm256_storeu_si256(parts.as_mut_ptr() as *mut __m256i, sum);
    let mut total: i32 = parts.iter().sum();
    for k in j..n {
        let t = (x[k] * s).round_ties_even().clamp(-128.0, 127.0) as i32;
        total += t;
        q[k] = t as i8;
    }
    (s, total)
}

#[inline(always)]
fn quantize_generic(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    let mut m = 0.0f32;
    for &v in x {
        m = m.max(v.abs());
    }
    let s = 127.0f32 / m.max(1e-5);
    let mut sum = 0i32;
    for (d, &v) in q.iter_mut().zip(x) {
        let t = (v * s).round_ties_even().clamp(-128.0, 127.0) as i32;
        sum += t;
        *d = t as i8;
    }
    (s, sum)
}

impl TernaryMlp {
    /// Validate shapes: widths chain, every K % 16 == 0, every width <= [`MLP_MAX_DIM`].
    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("MLP has no layers".into());
        }
        for (i, l) in self.layers.iter().enumerate() {
            if l.k == 0 || l.k % MLP_K_ALIGN != 0 {
                return Err(format!(
                    "layer {i}: K={} must be a nonzero multiple of 16 (zero-pad the features)",
                    l.k
                ));
            }
            if l.n == 0 || l.n > MLP_MAX_DIM || l.k > MLP_MAX_DIM {
                return Err(format!(
                    "layer {i}: N={} K={} must be in 1..={MLP_MAX_DIM}",
                    l.n, l.k
                ));
            }
            if l.w.len() != l.n * l.k || l.scale.len() != l.n || l.bias.len() != l.n {
                return Err(format!(
                    "layer {i}: buffer lengths don't match N={} K={}",
                    l.n, l.k
                ));
            }
            if l.w.iter().any(|&v| !(-1..=1).contains(&v)) {
                return Err(format!("layer {i}: weights must be in {{-1,0,1}}"));
            }
            if i + 1 < self.layers.len() {
                let next = &self.layers[i + 1];
                if next.k != l.n {
                    return Err(format!(
                        "layer {i}: N={} but layer {} has K={}",
                        l.n,
                        i + 1,
                        next.k
                    ));
                }
            }
        }
        Ok(())
    }

    /// Build from float weights (`weights[i]` is `[n][k]` row-major) with BitNet absmean
    /// quantization, per-row scales. `dims = [in, hidden.., out]`; hidden layers use `hidden_act`,
    /// the last layer is linear.
    pub fn from_f32(
        dims: &[usize],
        weights: &[Vec<f32>],
        biases: &[Vec<f32>],
        hidden_act: Act,
    ) -> Result<Self, String> {
        if dims.len() < 2 || weights.len() != dims.len() - 1 || biases.len() != dims.len() - 1 {
            return Err("need dims.len() - 1 weight and bias tensors".into());
        }
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            let (k, n) = (dims[i], dims[i + 1]);
            if weights[i].len() != n * k {
                return Err(format!("layer {i}: expected {} weights", n * k));
            }
            let (w, scale) = quantize_absmean(&weights[i], n, k, true);
            let act = if i + 2 == dims.len() {
                Act::Identity
            } else {
                hidden_act
            };
            layers.push(MlpLayer {
                n,
                k,
                w,
                scale,
                bias: biases[i].clone(),
                act,
            });
        }
        let m = Self { layers };
        m.validate()?;
        Ok(m)
    }

    /// A random model for tests and benchmarks (synthetic, not trained).
    pub fn random(dims: &[usize], hidden_act: Act, seed: u64) -> Result<Self, String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for i in 0..dims.len() - 1 {
            let (k, n) = (dims[i], dims[i + 1]);
            let a = 1.0 / (k as f32).sqrt();
            ws.push((0..n * k).map(|_| rng.random_range(-a..a)).collect());
            bs.push((0..n).map(|_| rng.random_range(-0.1f32..0.1)).collect());
        }
        Self::from_f32(dims, &ws, &bs, hidden_act)
    }

    pub fn in_dim(&self) -> usize {
        self.layers[0].k
    }
    pub fn out_dim(&self) -> usize {
        self.layers.last().unwrap().n
    }
    pub fn max_dim(&self) -> usize {
        self.layers.iter().map(|l| l.n.max(l.k)).max().unwrap()
    }
    /// Packed weight bytes (2 bits per weight) + scales and biases.
    pub fn weight_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.n * l.k / 4 + 8 * l.n).sum()
    }

    /// Scalar reference forward pass for one sample (allocates; this is the oracle).
    pub fn forward_reference(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim());
        let mut cur = x.to_vec();
        for l in &self.layers {
            let mut q = vec![0i8; l.k];
            let (s, _sum) = quantize_scalar(&cur, &mut q);
            let mut out = vec![0f32; l.n];
            for r in 0..l.n {
                let mut acc = 0i32;
                for c in 0..l.k {
                    acc += l.w[r * l.k + c] as i32 * q[c] as i32;
                }
                let v = (acc as f32 / s) * l.scale[r] + l.bias[r];
                out[r] = act(v, l.act);
            }
            cur = out;
        }
        cur
    }
}

/// `sum(c[i] * q[i])` with codes `c = w + 1` in {0,1,2}. Callers subtract `sum(q)`.
#[inline(always)]
fn dot_codes_generic(c: &[u8], q: &[i8]) -> i32 {
    let n = c.len().min(q.len());
    let (cc, cr) = c[..n].as_chunks::<16>();
    let (qc, qr) = q[..n].as_chunks::<16>();
    let mut acc = [0i32; 16];
    for (x, y) in cc.iter().zip(qc) {
        for i in 0..16 {
            acc[i] += x[i] as i32 * y[i] as i32;
        }
    }
    let mut s: i32 = acc.iter().sum();
    for (x, y) in cr.iter().zip(qr) {
        s += *x as i32 * *y as i32;
    }
    s
}

/// AVX2: `vpmaddubsw` (u8 codes x i8 activations -> i16 pairs, at most 2*2*128 so no saturation)
/// then `vpmaddwd` with ones -> i32. 32 multiply-adds per instruction pair.
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_codes_avx2(c: &[u8], q: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = c.len().min(q.len());
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0;
    while i + 64 <= n {
        let c0 = _mm256_loadu_si256(c.as_ptr().add(i) as *const __m256i);
        let q0 = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
        let c1 = _mm256_loadu_si256(c.as_ptr().add(i + 32) as *const __m256i);
        let q1 = _mm256_loadu_si256(q.as_ptr().add(i + 32) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(c0, q0), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(c1, q1), ones));
        i += 64;
    }
    while i + 32 <= n {
        let c0 = _mm256_loadu_si256(c.as_ptr().add(i) as *const __m256i);
        let q0 = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(c0, q0), ones));
        i += 32;
    }
    let v = _mm256_add_epi32(acc0, acc1);
    let h = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0b01_00_11_10));
    let h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0b10_11_00_01));
    let mut s = _mm_cvtsi128_si32(h);
    while i < n {
        s += c[i] as i32 * q[i] as i32;
        i += 1;
    }
    s
}

/// Four rows at once against the same activations: each activation load is reused 4x.
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_codes4_avx2(c: [&[u8]; 4], q: &[i8]) -> [i32; 4] {
    use std::arch::x86_64::*;
    let n = q.len();
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut i = 0;
    while i + 32 <= n {
        let qv = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
        for r in 0..4 {
            let cv = _mm256_loadu_si256(c[r].as_ptr().add(i) as *const __m256i);
            acc[r] = _mm256_add_epi32(
                acc[r],
                _mm256_madd_epi16(_mm256_maddubs_epi16(cv, qv), ones),
            );
        }
        i += 32;
    }
    // Horizontal sums of the four accumulators in one go.
    let a01 = _mm256_hadd_epi32(acc[0], acc[1]);
    let a23 = _mm256_hadd_epi32(acc[2], acc[3]);
    let a = _mm256_hadd_epi32(a01, a23); // [r0 r1 r2 r3 | r0 r1 r2 r3] partials
    let h = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    let mut out = [0i32; 4];
    _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, h);
    while i < n {
        for r in 0..4 {
            out[r] += c[r][i] as i32 * q[i] as i32;
        }
        i += 1;
    }
    out
}

/// AVX-512 VNNI: `vpdpbusd` multiplies u8 codes by i8 activations and accumulates groups of 4
/// straight into i32 lanes, 64 multiply-adds per instruction; the K tail uses a masked load.
/// Four rows share each activation load.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[clippy::msrv = "1.89"] // AVX-512 intrinsics; the avx512 feature documents this requirement
unsafe fn dot_codes4_vnni(c: [&[u8]; 4], q: &[i8]) -> [i32; 4] {
    use std::arch::x86_64::*;
    let n = q.len();
    let mut acc = [_mm512_setzero_si512(); 4];
    let mut i = 0;
    while i < n {
        let rem = n - i;
        let mask: u64 = if rem >= 64 {
            u64::MAX
        } else {
            (1u64 << rem) - 1
        };
        let qv = _mm512_maskz_loadu_epi8(mask, q.as_ptr().add(i));
        for r in 0..4 {
            let cv = _mm512_maskz_loadu_epi8(mask, c[r].as_ptr().add(i) as *const i8);
            acc[r] = _mm512_dpbusd_epi32(acc[r], cv, qv);
        }
        i += 64;
    }
    [
        _mm512_reduce_add_epi32(acc[0]),
        _mm512_reduce_add_epi32(acc[1]),
        _mm512_reduce_add_epi32(acc[2]),
        _mm512_reduce_add_epi32(acc[3]),
    ]
}

#[inline]
pub(crate) fn dot_codes(avx2: bool, c: &[u8], q: &[i8]) -> i32 {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    if avx2 {
        // SAFETY: callers only pass `avx2 = true` after runtime detection.
        return unsafe { dot_codes_avx2(c, q) };
    }
    let _ = avx2;
    dot_codes_generic(c, q)
}

/// Dot products of four code rows against the same activations (each activation load reused 4x).
#[inline]
pub(crate) fn dot_codes4(avx2: bool, c: [&[u8]; 4], q: &[i8]) -> [i32; 4] {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    if avx2 {
        // SAFETY: callers only pass `avx2 = true` after runtime detection; rows are >= q.len().
        return unsafe { dot_codes4_avx2(c, q) };
    }
    let _ = avx2;
    c.map(|row| dot_codes_generic(row, q))
}

#[inline(always)]
fn dot_f32_generic(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (ac, ar) = a[..n].as_chunks::<16>();
    let (bc, br) = b[..n].as_chunks::<16>();
    let mut acc = [0f32; 16];
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for j in 0..4 {
            let x = _mm256_loadu_ps(a.as_ptr().add(i + 8 * j));
            let y = _mm256_loadu_ps(b.as_ptr().add(i + 8 * j));
            acc[j] = _mm256_fmadd_ps(x, y, acc[j]);
        }
        i += 32;
    }
    while i + 8 <= n {
        acc[0] = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc[0],
        );
        i += 8;
    }
    let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    let mut s = _mm_cvtss_f32(h);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
fn dot_f32(avx2: bool, a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    if avx2 {
        // SAFETY: callers only pass `avx2 = true` after runtime detection of AVX2 + FMA.
        return unsafe { dot_f32_avx2(a, b) };
    }
    let _ = avx2;
    dot_f32_generic(a, b)
}

/// Which SIMD path the CPU engines use.
pub fn cpu_simd_level() -> &'static str {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if has_vnni() {
        return "avx512-vnni";
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return "avx2";
    }
    "portable (autovectorized baseline ISA)"
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn has_vnni() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vnni")
}

/// Allocation-free CPU inference for a [`TernaryMlp`]. Bit-identical to the reference.
pub struct CpuEngine {
    model: TernaryMlp,
    /// Per layer, codes `w + 1` row-major `[n][k]` as u8 (one byte per weight; small models fit
    /// in L1/L2 either way, and bytes feed `vpmaddubsw` directly).
    codes: Vec<Vec<u8>>,
    bufs: [Vec<f32>; 2],
    q: Vec<i8>,
    avx2: bool,
    #[cfg_attr(not(all(target_arch = "x86_64", feature = "avx512")), allow(dead_code))]
    vnni: bool,
    /// 2-bit packed weights decoded in registers ([`crate::cpu_packed`]), when the CPU has
    /// AVX-512 VNNI + VBMI; `None` falls back to the byte-code paths.
    packed: Option<Vec<crate::cpu_packed::Packed2>>,
    /// Row sums scratch for the packed path (`n` padded to 16).
    acc: Vec<i32>,
    /// Optional spinning worker pool for the packed path ([`Self::with_threads`]).
    pool: Option<crate::spin_pool::SpinPool>,
}

impl CpuEngine {
    pub fn new(model: &TernaryMlp) -> Result<Self, String> {
        model.validate()?;
        let d = model.max_dim();
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        let avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        let avx2 = false;
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        let vnni = has_vnni();
        #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
        let vnni = false;
        let codes = model
            .layers
            .iter()
            .map(|l| l.w.iter().map(|&v| (v + 1) as u8).collect())
            .collect();
        let packed = crate::cpu_packed::available().then(|| {
            model
                .layers
                .iter()
                .map(|l| crate::cpu_packed::Packed2::from_ternary(&l.w, l.n, l.k))
                .collect::<Vec<_>>()
        });
        let qlen = d.div_ceil(64) * 64;
        let acc_len = d.div_ceil(crate::cpu_packed::ROWS) * crate::cpu_packed::ROWS;
        Ok(Self {
            model: model.clone(),
            codes,
            bufs: [vec![0.0; d], vec![0.0; d]],
            q: vec![0; qlen],
            avx2,
            vnni,
            packed,
            acc: vec![0; acc_len],
            pool: None,
        })
    }

    /// Split each layer's rows across `threads` threads (caller included) using a spinning
    /// [`SpinPool`](crate::spin_pool::SpinPool). Only the packed AVX-512 path is threaded; results
    /// are identical for any thread count. Idle workers spin, so dedicate cores to this.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.pool = (threads > 1).then(|| crate::spin_pool::SpinPool::new(threads));
        self
    }

    /// Force the byte-code SIMD path even when the packed AVX-512 path is available (used to
    /// benchmark one against the other; results are identical).
    pub fn without_packed(mut self) -> Self {
        self.packed = None;
        self
    }

    fn forward_packed(&mut self, x: &[f32], out: &mut [f32]) {
        let packed = self.packed.as_ref().expect("packed path");
        let [b0, b1] = &mut self.bufs;
        let (mut cur, mut nxt) = (b0, b1);
        cur[..x.len()].copy_from_slice(x);
        let nl = self.model.layers.len();
        for (li, l) in self.model.layers.iter().enumerate() {
            let p = &packed[li];
            let (s, sum_q) = quantize_into(&cur[..l.k], &mut self.q[..l.k]);
            let (q, acc) = (&self.q[..p.q_len()], &mut self.acc[..p.n_padded()]);
            match &self.pool {
                // Small layers finish faster than a fork-join round trip; measured on an 8-vCPU
                // Xeon VM, the break-even is a few hundred thousand weights.
                Some(pool)
                    if l.n * l.k >= crate::cpu_packed::PAR_MIN_WEIGHTS && pool.threads() > 1 =>
                {
                    crate::cpu_packed::gemv_parallel(p, q, acc, pool)
                }
                _ => crate::cpu_packed::gemv(p, q, acc),
            }
            let dst: &mut [f32] = if li + 1 == nl {
                &mut *out
            } else {
                &mut nxt[..l.n]
            };
            crate::cpu_packed::epilogue(&self.acc, sum_q, s, &l.scale, &l.bias, l.act, dst);
            std::mem::swap(&mut cur, &mut nxt);
        }
    }

    /// `x: [in_dim]` -> `out: [out_dim]`. No heap allocation.
    pub fn forward_into(&mut self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.model.in_dim());
        assert_eq!(out.len(), self.model.out_dim());
        if self.packed.is_some() {
            return self.forward_packed(x, out);
        }
        let [b0, b1] = &mut self.bufs;
        let (mut cur, mut nxt) = (b0, b1);
        cur[..x.len()].copy_from_slice(x);
        let nl = self.model.layers.len();
        for (li, l) in self.model.layers.iter().enumerate() {
            let q = &mut self.q[..l.k];
            let (s, sum_q) = quantize_into(&cur[..l.k], q);
            let codes = &self.codes[li];
            let q = &self.q[..l.k];
            let dst: &mut [f32] = if li + 1 == nl {
                &mut *out
            } else {
                &mut nxt[..l.n]
            };
            let row = |r: usize| &codes[r * l.k..(r + 1) * l.k];
            // Integer row sums first, then one vectorized epilogue pass over the layer.
            let acc = &mut self.acc[..l.n];
            #[allow(unused_mut)] // only advanced by the SIMD 4-row paths
            let mut r0 = 0;
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if self.vnni {
                while r0 + 4 <= l.n {
                    // SAFETY: AVX-512 F/BW/VNNI detected at construction.
                    let a = unsafe {
                        dot_codes4_vnni([row(r0), row(r0 + 1), row(r0 + 2), row(r0 + 3)], q)
                    };
                    acc[r0..r0 + 4].copy_from_slice(&a);
                    r0 += 4;
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
            if self.avx2 {
                while r0 + 4 <= l.n {
                    // SAFETY: AVX2 detected at construction.
                    let a = unsafe {
                        dot_codes4_avx2([row(r0), row(r0 + 1), row(r0 + 2), row(r0 + 3)], q)
                    };
                    acc[r0..r0 + 4].copy_from_slice(&a);
                    r0 += 4;
                }
            }
            for r in r0..l.n {
                acc[r] = dot_codes(self.avx2, row(r), q);
            }
            crate::cpu_packed::epilogue(acc, sum_q, s, &l.scale, &l.bias, l.act, dst);
            std::mem::swap(&mut cur, &mut nxt);
        }
    }

    pub fn model(&self) -> &TernaryMlp {
        &self.model
    }

    /// The SIMD path this engine dispatches to on this CPU.
    pub fn simd(&self) -> &'static str {
        if self.packed.is_some() {
            if crate::cpu_packed::available() {
                "avx512-vnni-vbmi packed2"
            } else {
                "avx2 packed2"
            }
        } else if self.vnni {
            "avx512-vnni"
        } else if self.avx2 {
            "avx2"
        } else {
            "portable"
        }
    }
}

/// One dense layer: (n, k, weights `[n][k]`, bias, activation).
type DenseLayer = (usize, usize, Vec<f32>, Vec<f32>, Act);

/// Same architecture with dense f32 weights (the dequantized ternary weights), plain CPU baseline.
pub struct DenseF32Mlp {
    layers: Vec<DenseLayer>,
    bufs: [Vec<f32>; 2],
    avx2: bool,
}

impl DenseF32Mlp {
    pub fn new(model: &TernaryMlp) -> Self {
        let layers = model
            .layers
            .iter()
            .map(|l| {
                let w = (0..l.n * l.k)
                    .map(|i| l.w[i] as f32 * l.scale[i / l.k])
                    .collect();
                (l.n, l.k, w, l.bias.clone(), l.act)
            })
            .collect();
        let d = model.max_dim();
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        let avx2 = false;
        Self {
            layers,
            bufs: [vec![0.0; d], vec![0.0; d]],
            avx2,
        }
    }

    /// A dense f32 model straight from float weights (`weights[i]` is `[n][k]` row-major), no
    /// quantization: the baseline for a model trained in full precision. `dims = [in, hidden..,
    /// out]`; hidden layers use `hidden_act`, the last layer is linear.
    pub fn from_f32(
        dims: &[usize],
        weights: &[Vec<f32>],
        biases: &[Vec<f32>],
        hidden_act: Act,
    ) -> Result<Self, String> {
        if dims.len() < 2 || weights.len() != dims.len() - 1 || biases.len() != dims.len() - 1 {
            return Err("need dims.len() - 1 weight and bias tensors".into());
        }
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            let (k, n) = (dims[i], dims[i + 1]);
            if weights[i].len() != n * k || biases[i].len() != n {
                return Err(format!(
                    "layer {i}: expected {} weights and {n} biases",
                    n * k
                ));
            }
            let a = if i + 2 == dims.len() {
                Act::Identity
            } else {
                hidden_act
            };
            layers.push((n, k, weights[i].clone(), biases[i].clone(), a));
        }
        let d = *dims.iter().max().unwrap();
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        let avx2 = false;
        Ok(Self {
            layers,
            bufs: [vec![0.0; d], vec![0.0; d]],
            avx2,
        })
    }

    /// No heap allocation.
    pub fn forward_into(&mut self, x: &[f32], out: &mut [f32]) {
        let [b0, b1] = &mut self.bufs;
        let (mut cur, mut nxt) = (b0, b1);
        cur[..x.len()].copy_from_slice(x);
        let nl = self.layers.len();
        for (li, (n, k, w, b, a)) in self.layers.iter().enumerate() {
            let dst: &mut [f32] = if li + 1 == nl {
                &mut *out
            } else {
                &mut nxt[..*n]
            };
            for r in 0..*n {
                let row = &w[r * k..(r + 1) * k];
                let d = dot_f32(self.avx2, row, &cur[..*k]);
                dst[r] = act(d + b[r], *a);
            }
            std::mem::swap(&mut cur, &mut nxt);
        }
    }
}

/// Dense f32 MLP with k-major (transposed) weights: for each input `k`, broadcast `x[k]` and FMA
/// a contiguous row of `n` weights into `n / 16` AVX-512 accumulators (no horizontal sums).
/// A stronger dense baseline than [`DenseF32Mlp`]'s per-row dot products; with the `avx512`
/// feature on an AVX-512F CPU it uses 512-bit FMAs, otherwise a portable loop the compiler
/// autovectorizes. Not bit-identical to [`DenseF32Mlp`] (different summation order).
/// (n, n padded to 16, k, weights `[k][n_pad]`, bias `[n_pad]`, act)
type KMajorLayer = (usize, usize, usize, Vec<f32>, Vec<f32>, Act);

pub struct DenseF32KMajor {
    layers: Vec<KMajorLayer>,
    bufs: [Vec<f32>; 2],
    avx512: bool,
}

impl DenseF32KMajor {
    pub fn from_f32(
        dims: &[usize],
        weights: &[Vec<f32>],
        biases: &[Vec<f32>],
        hidden_act: Act,
    ) -> Result<Self, String> {
        if dims.len() < 2 || weights.len() != dims.len() - 1 || biases.len() != dims.len() - 1 {
            return Err("need dims.len() - 1 weight and bias tensors".into());
        }
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            let (k, n) = (dims[i], dims[i + 1]);
            if weights[i].len() != n * k || biases[i].len() != n {
                return Err(format!(
                    "layer {i}: expected {} weights and {n} biases",
                    n * k
                ));
            }
            let np = n.div_ceil(16) * 16;
            let mut w = vec![0f32; k * np];
            for r in 0..n {
                for c in 0..k {
                    w[c * np + r] = weights[i][r * k + c];
                }
            }
            let mut b = vec![0f32; np];
            b[..n].copy_from_slice(&biases[i]);
            let a = if i + 2 == dims.len() {
                Act::Identity
            } else {
                hidden_act
            };
            layers.push((n, np, k, w, b, a));
        }
        let d = layers.iter().map(|l| l.1.max(l.2)).max().unwrap();
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        let avx512 = std::is_x86_feature_detected!("avx512f");
        #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
        let avx512 = false;
        Ok(Self {
            layers,
            bufs: [vec![0.0; d], vec![0.0; d]],
            avx512,
        })
    }

    pub fn simd(&self) -> &'static str {
        if self.avx512 {
            "avx512f k-major fma"
        } else {
            "portable k-major (autovectorized)"
        }
    }

    /// No heap allocation.
    pub fn forward_into(&mut self, x: &[f32], out: &mut [f32]) {
        let [b0, b1] = &mut self.bufs;
        let (mut cur, mut nxt) = (b0, b1);
        cur[..x.len()].copy_from_slice(x);
        for (n, np, k, w, b, a) in self.layers.iter() {
            let dst = &mut nxt[..*np];
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if self.avx512 {
                // SAFETY: AVX-512F detected; w is [k][np], b and dst are np long, cur >= k.
                unsafe { kmajor_avx512(w, b, &cur[..*k], dst, *np) };
            } else {
                kmajor_generic(w, b, &cur[..*k], dst, *np);
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
            kmajor_generic(w, b, &cur[..*k], dst, *np);
            for v in dst[..*n].iter_mut() {
                *v = act(*v, *a);
            }
            std::mem::swap(&mut cur, &mut nxt);
        }
        let last = self.layers.last().unwrap().0;
        out[..last].copy_from_slice(&cur[..last]);
    }
}

fn kmajor_generic(w: &[f32], b: &[f32], x: &[f32], dst: &mut [f32], np: usize) {
    dst.copy_from_slice(b);
    for (c, &xv) in x.iter().enumerate() {
        let row = &w[c * np..(c + 1) * np];
        for (d, &wv) in dst.iter_mut().zip(row) {
            *d += wv * xv;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
unsafe fn kmajor_avx512(w: &[f32], b: &[f32], x: &[f32], dst: &mut [f32], np: usize) {
    // Blocks of 16 accumulators (256 outputs) stay in registers across the whole K loop; the
    // block width is a const generic so the accumulator array is register-allocated.
    let mut n0 = 0;
    while n0 + 256 <= np {
        kmajor_block::<16>(w, b, x, dst, np, n0);
        n0 += 256;
    }
    while n0 + 64 <= np {
        kmajor_block::<4>(w, b, x, dst, np, n0);
        n0 += 64;
    }
    while n0 < np {
        kmajor_block::<1>(w, b, x, dst, np, n0);
        n0 += 16;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[inline]
// The `avx512` feature already requires rustc >= 1.89 (see Cargo.toml).
#[allow(clippy::incompatible_msrv)]
unsafe fn kmajor_block<const NB: usize>(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    dst: &mut [f32],
    np: usize,
    n0: usize,
) {
    use std::arch::x86_64::*;
    let mut acc = [_mm512_setzero_ps(); NB];
    for j in 0..NB {
        acc[j] = _mm512_loadu_ps(b.as_ptr().add(n0 + 16 * j));
    }
    for (c, &xv) in x.iter().enumerate() {
        let xb = _mm512_set1_ps(xv);
        let row = w.as_ptr().add(c * np + n0);
        for j in 0..NB {
            acc[j] = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16 * j)), xb, acc[j]);
        }
    }
    for j in 0..NB {
        _mm512_storeu_ps(dst.as_mut_ptr().add(n0 + 16 * j), acc[j]);
    }
}
