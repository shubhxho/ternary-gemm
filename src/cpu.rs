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
        let rows = |r0: usize, out: &mut [f16]| {
            for (i, o) in out.chunks_mut(batch).enumerate() {
                let r = r0 + i;
                let row = &self.codes[r * k..(r + 1) * k];
                for (b, ob) in o.iter_mut().enumerate() {
                    let xb = &a.q[b * k..(b + 1) * k];
                    let acc = crate::mlp::dot_codes(self.avx2, row, xb) - a.sum[b];
                    let mut t = acc as f32;
                    t /= a.s[b];
                    t *= self.row_scale[r];
                    if let Some(res) = residual {
                        t += res[b * n + r].to_f32();
                    }
                    *ob = f16::from_f32(t);
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
