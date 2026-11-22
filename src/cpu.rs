//! CPU backend for the W2A8 GEMM: the no-GPU fallback for the LLM path.
//!
//! Same math as [`crate::reference::gemm_i8`] and bit-identical to it (integer dot products, then
//! `acc / s * scale (+ residual)` in f32, rounded once to fp16), but with weights decoded once to
//! byte codes, SIMD dot products (AVX2 when the `avx2` feature is on and the CPU has it), **any
//! batch size** (no 16-token tile limit) and optional multi-threading over output rows.

use crate::pack::PackedTernary;
use crate::reference::QuantizedActs;
use half::f16;

/// Ternary weights prepared for the CPU: codes `w + 1` as bytes, row-major `[n][k]`.
#[derive(Debug, Clone)]
pub struct CpuTernary {
    pub n: usize,
    pub k: usize,
    codes: Vec<u8>,
    row_scale: Vec<f32>,
    avx2: bool,
    threads: usize,
}

impl CpuTernary {
    /// Decode packed weights for CPU execution (single-threaded; see [`Self::with_threads`]).
    pub fn new(w: &PackedTernary) -> Self {
        let codes = w.unpack().into_iter().map(|v| (v + 1) as u8).collect();
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        let avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        let avx2 = false;
        Self {
            n: w.n,
            k: w.k,
            codes,
            row_scale: w.row_scale.clone(),
            avx2,
            threads: 1,
        }
    }

    /// Split output rows across `threads` scoped threads (1 = no threading). Results are identical
    /// for any thread count: every output element is computed independently.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// `y[b][n] = (sum_k w[n][k] * q[b][k]) / s[b] * row_scale[n] (+ residual)` as fp16, any batch.
    pub fn gemm_i8(&self, a: &QuantizedActs, residual: Option<&[f16]>) -> Vec<f16> {
        let mut y = vec![f16::ZERO; a.batch * self.n];
        self.gemm_i8_into(a, residual, &mut y);
        y
    }

    /// Allocation-free form of [`Self::gemm_i8`]; `y` is `[batch][n]`.
    pub fn gemm_i8_into(&self, a: &QuantizedActs, residual: Option<&[f16]>, y: &mut [f16]) {
        assert_eq!(a.k, self.k, "activation K does not match weights");
        assert_eq!(y.len(), a.batch * self.n, "output must be [batch][n]");
        if let Some(r) = residual {
            assert_eq!(r.len(), a.batch * self.n, "residual must be [batch][n]");
        }
        let (n, k, batch) = (self.n, self.k, a.batch);
        // Row-major [n][batch] scratch so each thread owns a contiguous block of rows.
        let mut yt = vec![f16::ZERO; n * batch];
        let epilogue = |r: usize, b: usize, acc: i32| {
            let mut t = (acc - a.sum[b]) as f32;
            t /= a.s[b];
            t *= self.row_scale[r];
            if let Some(res) = residual {
                t += res[b * n + r].to_f32();
            }
            f16::from_f32(t)
        };
        let rows = |r0: usize, out: &mut [f16]| {
            let nrows = out.len() / batch.max(1);
            let row = |r: usize| &self.codes[r * k..(r + 1) * k];
            let mut i = 0;
            // Four rows per pass so every activation vector load feeds four dot products.
            while i + 4 <= nrows {
                let r = r0 + i;
                let c = [row(r), row(r + 1), row(r + 2), row(r + 3)];
                for b in 0..batch {
                    let acc = crate::mlp::dot_codes4(self.avx2, c, &a.q[b * k..(b + 1) * k]);
                    for j in 0..4 {
                        out[(i + j) * batch + b] = epilogue(r + j, b, acc[j]);
                    }
                }
                i += 4;
            }
            for i in i..nrows {
                let r = r0 + i;
                for b in 0..batch {
                    let acc = crate::mlp::dot_codes(self.avx2, row(r), &a.q[b * k..(b + 1) * k]);
                    out[i * batch + b] = epilogue(r, b, acc);
                }
            }
        };
        let threads = self.threads.min(n);
        if threads <= 1 || batch == 0 {
            rows(0, &mut yt);
        } else {
            let per = n.div_ceil(threads);
            std::thread::scope(|s| {
                for (t, chunk) in yt.chunks_mut(per * batch).enumerate() {
                    let rows = &rows;
                    s.spawn(move || rows(t * per, chunk));
                }
            });
        }
        for r in 0..n {
            for b in 0..batch {
                y[b * n + r] = yt[r * batch + b];
            }
        }
    }
}

/// Fast activation quantization, bit-identical to [`crate::reference::quantize_acts`].
///
/// Converts fp16 -> f32 with the vectorised `half` slice conversion, applies the RMSNorm scale in
/// the same operation order as the reference (`(x * g) * inv`, with the sum of squares in f64), and
/// then reuses the SIMD absmax quantizer [`crate::mlp::quantize_into`]. That is exact: the
/// reference takes `absmax = max|x * g| * inv`, and because rounding is monotone that equals
/// `max|(x * g) * inv|`, which is what the SIMD quantizer computes.
pub fn quantize_acts(
    x: &[f16],
    batch: usize,
    k: usize,
    gamma: Option<&[f16]>,
    eps: f32,
) -> QuantizedActs {
    use half::slice::HalfFloatSliceExt;
    assert_eq!(x.len(), batch * k, "x must be [batch][k]");
    if let Some(g) = gamma {
        assert!(g.len() >= k, "gamma too short");
    }
    let mut out = QuantizedActs {
        batch,
        k,
        q: vec![0; batch * k],
        s: vec![0.0; batch],
        sum: vec![0; batch],
    };
    let mut xf = vec![0f32; k];
    let gf: Option<Vec<f32>> = gamma.map(|g| g[..k].iter().map(|v| v.to_f32()).collect());
    for b in 0..batch {
        x[b * k..(b + 1) * k].convert_to_f32_slice(&mut xf);
        if let Some(g) = &gf {
            let ss: f64 = xf.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv = (1.0 / ((ss / k as f64) + eps as f64).sqrt()) as f32;
            for (v, &gi) in xf.iter_mut().zip(g) {
                *v = (*v * gi) * inv;
            }
        }
        let (s, sum) = crate::mlp::quantize_into(&xf, &mut out.q[b * k..(b + 1) * k]);
        out.s[b] = s;
        out.sum[b] = sum;
    }
    out
}
