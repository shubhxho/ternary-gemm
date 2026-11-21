//! Pure-Rust CPU reference implementation (the correctness oracle for the GPU kernels).
//!
//! The int8 path mirrors the GPU arithmetic exactly (same quantization formula, exact integer
//! accumulation, same rounding-mode-pinned epilogue), so without RMSNorm the GPU output is
//! expected to be **bit-identical**. With RMSNorm the GPU uses `rsqrtf` and a different summation
//! order, so a small tolerance applies.

use crate::pack::PackedTernary;
use half::f16;

/// Per-token int8 activations (BitNet absmax scheme).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedActs {
    pub batch: usize,
    pub k: usize,
    /// `[batch][k]`
    pub q: Vec<i8>,
    /// Quantization scale per token: `s = 127 / absmax`; dequantize with `q / s`.
    pub s: Vec<f32>,
    /// `sum(q)` per token (used to undo the `code = w + 1` offset).
    pub sum: Vec<i32>,
}

/// Quantize `[batch][k]` fp16 activations, optionally applying RMSNorm with `gamma` first.
pub fn quantize_acts(
    x: &[f16],
    batch: usize,
    k: usize,
    gamma: Option<&[f16]>,
    eps: f32,
) -> QuantizedActs {
    assert_eq!(x.len(), batch * k);
    let mut out = QuantizedActs {
        batch,
        k,
        q: vec![0; batch * k],
        s: vec![0.0; batch],
        sum: vec![0; batch],
    };
    for b in 0..batch {
        let xb = &x[b * k..(b + 1) * k];
        let (inv, m) = match gamma {
            Some(g) => {
                let ss: f64 = xb.iter().map(|v| (v.to_f32() as f64).powi(2)).sum();
                let inv = (1.0 / ((ss / k as f64) + eps as f64).sqrt()) as f32;
                let m = xb
                    .iter()
                    .zip(g)
                    .map(|(v, g)| (v.to_f32() * g.to_f32()).abs())
                    .fold(0f32, f32::max);
                (inv, m)
            }
            None => (
                1.0,
                xb.iter().map(|v| v.to_f32().abs()).fold(0f32, f32::max),
            ),
        };
        let absmax = (if gamma.is_some() { m * inv } else { m }).max(1e-5);
        let s = 127.0f32 / absmax;
        let mut sum = 0i32;
        for i in 0..k {
            let mut v = xb[i].to_f32();
            if let Some(g) = gamma {
                v = (v * g[i].to_f32()) * inv;
            }
            let q = (v * s).round_ties_even().clamp(-128.0, 127.0) as i32;
            sum += q;
            out.q[b * k + i] = q as i8;
        }
        out.s[b] = s;
        out.sum[b] = sum;
    }
    out
}

/// `y[b][n] = (sum_k w[n][k] * q[b][k]) / s[b] * row_scale[n] (+ residual)`, output fp16.
pub fn gemm_i8(w: &PackedTernary, a: &QuantizedActs, residual: Option<&[f16]>) -> Vec<f16> {
    assert_eq!(a.k, w.k);
    let (n, k, batch) = (w.n, w.k, a.batch);
    let mut y = vec![f16::ZERO; batch * n];
    let mut row = vec![0i8; k];
    for r in 0..n {
        for (c, v) in row.iter_mut().enumerate() {
            *v = w.get(r, c);
        }
        for b in 0..batch {
            let xb = &a.q[b * k..(b + 1) * k];
            let acc: i32 = row.iter().zip(xb).map(|(&w, &x)| w as i32 * x as i32).sum();
            let mut t = acc as f32;
            t /= a.s[b];
            t *= w.row_scale[r];
            if let Some(res) = residual {
                t += res[b * n + r].to_f32();
            }
            y[b * n + r] = f16::from_f32(t);
        }
    }
    y
}

/// W2A16: `y[b][n] = (sum_k w[n][k] * x[b][k]) * row_scale[n] (+ residual)` accumulated in f64.
/// Returns f32 so callers can measure the GPU error against a high-precision reference.
pub fn gemm_f16(w: &PackedTernary, x: &[f16], batch: usize, residual: Option<&[f16]>) -> Vec<f32> {
    let (n, k) = (w.n, w.k);
    assert_eq!(x.len(), batch * k);
    let mut y = vec![0f32; batch * n];
    let mut row = vec![0i8; k];
    for r in 0..n {
        for (c, v) in row.iter_mut().enumerate() {
            *v = w.get(r, c);
        }
        for b in 0..batch {
            let xb = &x[b * k..(b + 1) * k];
            let acc: f64 = row
                .iter()
                .zip(xb)
                .map(|(&w, x)| w as f64 * x.to_f32() as f64)
                .sum();
            let mut t = acc * w.row_scale[r] as f64;
            if let Some(res) = residual {
                t += res[b * n + r].to_f32() as f64;
            }
            y[b * n + r] = t as f32;
        }
    }
    y
}
