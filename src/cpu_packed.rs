//! 2-bit packed ternary GEMV for the CPU, decoded in registers (AVX-512 VNNI + VBMI).
//!
//! The byte-code engines spend one byte per weight, so the `medium` signal model (3 MiB of codes)
//! falls out of a 2 MiB L2 and the dot products wait on L3. This layout keeps the weights at
//! 2 bits (4x less traffic) and unpacks them with one `vpmultishiftqb` + one `vpandd` per 64
//! weights, which run on port 5 while `vpdpbusd` keeps port 0 busy.
//!
//! Layout ([`Packed2`]): rows in blocks of [`ROWS`] = 16, K in chunks of [`CHUNK`] = 256 weights.
//! Each (block, chunk, row) owns 64 bytes; byte `j` holds the codes `w + 1` of the four weights
//! at chunk offsets `j`, `j + 64`, `j + 128`, `j + 192` in bits `0-1`, `2-3`, `4-5`, `6-7`. So
//! sub-vector `p` of a chunk is `(bytes >> 2p) & 3`, lined up with activations `q[64p..64p+64]`.
//! Padding (rows past `n`, weights past `k`) is code 0, which contributes nothing whatever the
//! activation, so no activation masking is needed.
//!
//! The 16 row accumulators of a block are reduced to one vector of 16 sums with a transpose-add
//! tree, and the dequant epilogue (`(acc - sum_q) / s * scale + bias`, then the activation) runs
//! 16 lanes wide with exactly the scalar operation order, so results stay bit-identical.

use crate::mlp::Act;

/// Rows per block (one accumulator register each).
pub const ROWS: usize = 16;
/// Weights per 64-byte row chunk.
pub const CHUNK: usize = 256;

/// Ternary weights packed for [`gemv`] (see the module docs for the layout).
#[derive(Debug, Clone)]
pub struct Packed2 {
    pub n: usize,
    pub k: usize,
    blocks: usize,
    chunks: usize,
    data: Vec<u8>,
}

impl Packed2 {
    /// Pack ternary weights `w` (`[n][k]`, values in `{-1, 0, 1}`).
    pub fn from_ternary(w: &[i8], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k, "weights must be [n][k]");
        assert!(n > 0 && k > 0, "empty layer");
        let blocks = n.div_ceil(ROWS);
        let chunks = k.div_ceil(CHUNK);
        let mut data = vec![0u8; blocks * chunks * ROWS * 64];
        for r in 0..n {
            for c in 0..k {
                let v = w[r * k + c];
                debug_assert!((-1..=1).contains(&v));
                let code = (v + 1) as u8;
                let (ch, off) = (c / CHUNK, c % CHUNK);
                let idx = ((r / ROWS * chunks + ch) * ROWS + r % ROWS) * 64 + off % 64;
                data[idx] |= code << (2 * (off / 64));
            }
        }
        Self {
            n,
            k,
            blocks,
            chunks,
            data,
        }
    }

    /// Rows computed by [`gemv`] (`n` rounded up to [`ROWS`]); the extra sums are padding.
    pub fn n_padded(&self) -> usize {
        self.blocks * ROWS
    }

    /// Activation length [`gemv`] reads: `k` rounded up to 64 (the tail must be readable; its
    /// values are ignored because the matching codes are 0).
    pub fn q_len(&self) -> usize {
        self.k.div_ceil(64) * 64
    }

    /// Packed bytes.
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    fn parts_last(&self) -> usize {
        (self.k - (self.chunks - 1) * CHUNK).div_ceil(64)
    }

    /// Scalar decode of the layout: `out[r] = sum_c code[r][c] * q[c]` for `r < n_padded()`.
    pub fn gemv_scalar(&self, q: &[i8], out: &mut [i32]) {
        assert!(q.len() >= self.q_len() && out.len() >= self.n_padded());
        for b in 0..self.blocks {
            for r in 0..ROWS {
                let mut acc = 0i32;
                for ch in 0..self.chunks {
                    let parts = if ch + 1 == self.chunks {
                        self.parts_last()
                    } else {
                        4
                    };
                    let row = &self.data[((b * self.chunks + ch) * ROWS + r) * 64..][..64];
                    for p in 0..parts {
                        for (j, &byte) in row.iter().enumerate() {
                            let c = ((byte >> (2 * p)) & 3) as i32;
                            acc += c * q[ch * CHUNK + 64 * p + j] as i32;
                        }
                    }
                }
                out[b * ROWS + r] = acc;
            }
        }
    }
}

/// Whether this CPU can run the packed AVX-512 kernels.
pub fn available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        false
    }
}

/// `out[r] = sum_c code[r][c] * q[c]` for every padded row, using AVX-512 when [`available`].
pub fn gemv(p: &Packed2, q: &[i8], out: &mut [i32]) {
    assert!(q.len() >= p.q_len(), "q shorter than Packed2::q_len()");
    assert!(
        out.len() >= p.n_padded(),
        "out shorter than Packed2::n_padded()"
    );
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if available() {
        // SAFETY: features detected; lengths checked above.
        return unsafe { gemv_avx512(p, q, out) };
    }
    p.gemv_scalar(q, out)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
#[clippy::msrv = "1.89"]
unsafe fn gemv_avx512(p: &Packed2, q: &[i8], out: &mut [i32]) {
    use std::arch::x86_64::*;
    // vpmultishiftqb control: output byte j of each qword takes bits [8j + s, 8j + s + 8).
    let ctl = |s: u64| {
        let mut c = 0u64;
        for j in 0..8 {
            c |= (8 * j + s) << (8 * j);
        }
        _mm512_set1_epi64(c as i64)
    };
    let (c0, c2, c4, c6) = (ctl(0), ctl(2), ctl(4), ctl(6));
    let m3 = _mm512_set1_epi8(3);
    let parts_last = p.parts_last();
    let qp = q.as_ptr();
    for b in 0..p.blocks {
        let mut acc = [_mm512_setzero_si512(); ROWS];
        let base = p.data.as_ptr().add(b * p.chunks * ROWS * 64);
        for ch in 0..p.chunks {
            let wp = base.add(ch * ROWS * 64);
            let qc = qp.add(ch * CHUNK);
            let parts = if ch + 1 == p.chunks { parts_last } else { 4 };
            if parts == 4 {
                let q0 = _mm512_loadu_si512(qc as *const _);
                let q1 = _mm512_loadu_si512(qc.add(64) as *const _);
                let q2 = _mm512_loadu_si512(qc.add(128) as *const _);
                let q3 = _mm512_loadu_si512(qc.add(192) as *const _);
                for r in 0..ROWS {
                    let v = _mm512_loadu_si512(wp.add(r * 64) as *const _);
                    let mut a = acc[r];
                    a = _mm512_dpbusd_epi32(a, _mm512_and_si512(v, m3), q0);
                    a = _mm512_dpbusd_epi32(
                        a,
                        _mm512_and_si512(_mm512_multishift_epi64_epi8(c2, v), m3),
                        q1,
                    );
                    a = _mm512_dpbusd_epi32(
                        a,
                        _mm512_and_si512(_mm512_multishift_epi64_epi8(c4, v), m3),
                        q2,
                    );
                    a = _mm512_dpbusd_epi32(
                        a,
                        _mm512_and_si512(_mm512_multishift_epi64_epi8(c6, v), m3),
                        q3,
                    );
                    acc[r] = a;
                }
            } else {
                for part in 0..parts {
                    let qv = _mm512_loadu_si512(qc.add(64 * part) as *const _);
                    let ctlp = match part {
                        0 => c0,
                        1 => c2,
                        2 => c4,
                        _ => c6,
                    };
                    for r in 0..ROWS {
                        let v = _mm512_loadu_si512(wp.add(r * 64) as *const _);
                        let c = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctlp, v), m3);
                        acc[r] = _mm512_dpbusd_epi32(acc[r], c, qv);
                    }
                }
            }
        }
        _mm512_storeu_si512(out.as_mut_ptr().add(b * ROWS) as *mut _, reduce16(&acc));
    }
}

/// Transpose-add 16 accumulators into one vector whose lane `r` is the sum of `acc[r]`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[clippy::msrv = "1.89"]
#[inline]
unsafe fn reduce16(acc: &[std::arch::x86_64::__m512i; ROWS]) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    // Pairs of rows: per 128-bit lane [a0+a2, b0+b2, a1+a3, b1+b3].
    let mut x = [_mm512_setzero_si512(); 8];
    for i in 0..8 {
        let (a, b) = (acc[2 * i], acc[2 * i + 1]);
        x[i] = _mm512_add_epi32(_mm512_unpacklo_epi32(a, b), _mm512_unpackhi_epi32(a, b));
    }
    // Quads: per 128-bit lane [r0, r1, r2, r3] partial sums for rows 4i..4i+3.
    let mut y = [_mm512_setzero_si512(); 4];
    for i in 0..4 {
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        y[i] = _mm512_add_epi32(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
    }
    // Fold the four 128-bit lanes and line the quads up in row order.
    let s01 = _mm512_add_epi32(
        _mm512_shuffle_i32x4(y[0], y[1], 0b10_00_10_00),
        _mm512_shuffle_i32x4(y[0], y[1], 0b11_01_11_01),
    );
    let s23 = _mm512_add_epi32(
        _mm512_shuffle_i32x4(y[2], y[3], 0b10_00_10_00),
        _mm512_shuffle_i32x4(y[2], y[3], 0b11_01_11_01),
    );
    _mm512_add_epi32(
        _mm512_shuffle_i32x4(s01, s23, 0b10_00_10_00),
        _mm512_shuffle_i32x4(s01, s23, 0b11_01_11_01),
    )
}

/// Dequant epilogue, scalar order: `act(((acc - sum_q) as f32 / s) * scale + bias)`.
#[inline(always)]
fn epilogue_scalar(acc: i32, sum_q: i32, s: f32, scale: f32, bias: f32, a: Act) -> f32 {
    let v = ((acc - sum_q) as f32 / s) * scale + bias;
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

/// `dst[r] = act(((acc[r] - sum_q) / s) * scale[r] + bias[r])` for `r < dst.len()`, 16 lanes
/// wide when AVX-512 is available, bit-identical to the scalar form either way.
pub fn epilogue(
    acc: &[i32],
    sum_q: i32,
    s: f32,
    scale: &[f32],
    bias: &[f32],
    a: Act,
    dst: &mut [f32],
) {
    let n = dst.len();
    assert!(acc.len() >= n && scale.len() >= n && bias.len() >= n);
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if available() {
        // SAFETY: AVX-512F detected; lengths checked above.
        return unsafe { epilogue_avx512(acc, sum_q, s, scale, bias, a, dst) };
    }
    for r in 0..n {
        dst[r] = epilogue_scalar(acc[r], sum_q, s, scale[r], bias[r], a);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[clippy::msrv = "1.89"]
unsafe fn epilogue_avx512(
    acc: &[i32],
    sum_q: i32,
    s: f32,
    scale: &[f32],
    bias: &[f32],
    a: Act,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let (sq, sv, zero) = (
        _mm512_set1_epi32(sum_q),
        _mm512_set1_ps(s),
        _mm512_setzero_ps(),
    );
    let mut i = 0;
    while i < n {
        let m: u16 = if n - i >= 16 {
            u16::MAX
        } else {
            (1u16 << (n - i)) - 1
        };
        let x = _mm512_maskz_loadu_epi32(m, acc.as_ptr().add(i));
        // Same single-rounding ops as the scalar path; no FMA contraction.
        let mut v = _mm512_cvtepi32_ps(_mm512_sub_epi32(x, sq));
        v = _mm512_div_ps(v, sv);
        v = _mm512_mul_ps(v, _mm512_maskz_loadu_ps(m, scale.as_ptr().add(i)));
        v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(m, bias.as_ptr().add(i)));
        v = match a {
            Act::Identity => v,
            // max(v, 0) returns the second operand when v is NaN or both are zero, which matches
            // `if v > 0.0 { v } else { 0.0 }` exactly (NaN -> 0, -0.0 -> +0.0).
            Act::Relu => _mm512_max_ps(v, zero),
            Act::Relu2 => {
                let r = _mm512_max_ps(v, zero);
                _mm512_mul_ps(r, r)
            }
        };
        _mm512_mask_storeu_ps(dst.as_mut_ptr().add(i), m, v);
        i += 16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn packed_gemv_matches_naive() {
        let mut rng = StdRng::seed_from_u64(9);
        for &(n, k) in &[
            (1, 16),
            (5, 64),
            (16, 256),
            (17, 80),
            (33, 1024),
            (64, 1040),
            (3, 4096),
        ] {
            let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
            let p = Packed2::from_ternary(&w, n, k);
            let mut q: Vec<i8> = (0..p.q_len()).map(|_| rng.random()).collect();
            // Garbage past k must not matter.
            for v in &mut q[k..] {
                *v = -128;
            }
            let mut fast = vec![0; p.n_padded()];
            let mut slow = vec![0; p.n_padded()];
            gemv(&p, &q, &mut fast);
            p.gemv_scalar(&q, &mut slow);
            for r in 0..n {
                let want: i32 = (0..k)
                    .map(|c| (w[r * k + c] as i32 + 1) * q[c] as i32)
                    .sum();
                assert_eq!(slow[r], want, "scalar {n}x{k} row {r}");
                assert_eq!(fast[r], want, "simd {n}x{k} row {r}");
            }
        }
    }

    #[test]
    fn epilogue_matches_scalar_including_edge_values() {
        let mut rng = StdRng::seed_from_u64(3);
        for a in [Act::Identity, Act::Relu, Act::Relu2] {
            for n in [1usize, 15, 16, 17, 100] {
                let acc: Vec<i32> = (0..n).map(|_| rng.random_range(-40000..40000)).collect();
                let scale: Vec<f32> = (0..n).map(|_| rng.random_range(-0.1..0.1)).collect();
                let mut bias: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
                bias[0] = -0.0;
                let s = rng.random_range(1.0f32..300.0);
                let sum_q = rng.random_range(-500..500);
                let mut got = vec![0f32; n];
                epilogue(&acc, sum_q, s, &scale, &bias, a, &mut got);
                for r in 0..n {
                    let want = epilogue_scalar(acc[r], sum_q, s, scale[r], bias[r], a);
                    assert_eq!(got[r].to_bits(), want.to_bits(), "{a:?} n{n} r{r}");
                }
            }
        }
    }
}
