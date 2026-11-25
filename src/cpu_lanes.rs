//! Rows-in-lanes 2-bit ternary GEMV for batch-1 latency (AVX-512 VNNI, GFNI or VBMI decode).
//!
//! [`crate::cpu_packed::Packed2`] puts 64 weights of **one row** in a vector, so every 16-row
//! block ends in a 16-way transpose-add (`reduce16`) and each 256 weights cost one shift + one
//! AND per sub-vector to decode. For the small layers of a signal model (K = 64 or 256) that
//! reduction and decode dominate. This layout puts **16 rows in the 16 lanes** of a vector:
//! lane `r` of vector `(block, g)` is the u32 holding the 16 codes `w + 1` of row `16·block + r`
//! for `k = 16g .. 16g + 16`, with byte `j`, bits `2p..2p+1` = code of `k = 16g + 4p + j`.
//! Sub-vector `p` (`(v >> 2p) & 3` per byte, one `vgf2p8affineqb` with GFNI or `vpmultishiftqb` +
//! AND without) lines up with the four activations `q[16g + 4p ..][..4]` broadcast to every
//! lane, so one `vpdpbusd` adds them into all 16 row sums at once. No horizontal reduction is
//! needed: after the last `g` the four per-part accumulators are added and lane `r` *is* row `r`.
//!
//! [`layer`] fuses the dequant epilogue (`act(((acc - sum_q) / s) * scale + bias)`, same IEEE
//! operations and order as the scalar reference, no FMA contraction) and the absmax needed by the
//! next layer's int8 quantizer into the same pass, so the `vdivps` of block `b` overlaps the dot
//! products of block `b + 1` instead of running as a separate pass.

use crate::mlp::Act;

/// Rows per block (lanes of one vector).
pub const ROWS: usize = 16;
/// Weights per row per vector (one u32 of 2-bit codes).
pub const KG: usize = 16;

/// Ternary weights in the rows-in-lanes layout (see the module docs).
#[derive(Debug, Clone)]
pub struct PackedLanes {
    pub n: usize,
    pub k: usize,
    blocks: usize,
    groups: usize,
    /// `[block][group][lane]` u32 words.
    data: Vec<u32>,
}

impl PackedLanes {
    /// Pack ternary weights `w` (`[n][k]`, values in `{-1, 0, 1}`). Padding (rows past `n`,
    /// weights past `k`) is code 0, which contributes nothing whatever the activation.
    pub fn from_ternary(w: &[i8], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k, "weights must be [n][k]");
        assert!(n > 0 && k > 0, "empty layer");
        let blocks = n.div_ceil(ROWS);
        let groups = k.div_ceil(KG);
        let mut data = vec![0u32; blocks * groups * ROWS];
        for r in 0..n {
            for c in 0..k {
                let v = w[r * k + c];
                debug_assert!((-1..=1).contains(&v));
                let code = (v + 1) as u32;
                let (g, off) = (c / KG, c % KG);
                let (p, j) = (off / 4, off % 4);
                data[((r / ROWS) * groups + g) * ROWS + r % ROWS] |= code << (8 * j + 2 * p);
            }
        }
        Self {
            n,
            k,
            blocks,
            groups,
            data,
        }
    }

    /// Rows computed by [`gemv`] (`n` rounded up to [`ROWS`]).
    pub fn n_padded(&self) -> usize {
        self.blocks * ROWS
    }

    /// Activation length the kernels read: `k` rounded up to 16 (the tail must be readable; its
    /// values are ignored because the matching codes are 0).
    pub fn q_len(&self) -> usize {
        self.groups * KG
    }

    /// Number of 16-row blocks.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Packed bytes.
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }

    /// Scalar decode of the layout: `out[r] = sum_c code[r][c] * q[c]` for `r < n_padded()`.
    pub fn gemv_scalar(&self, q: &[i8], out: &mut [i32]) {
        assert!(q.len() >= self.q_len() && out.len() >= self.n_padded());
        for b in 0..self.blocks {
            for lane in 0..ROWS {
                let mut acc = 0i32;
                for g in 0..self.groups {
                    let word = self.data[(b * self.groups + g) * ROWS + lane];
                    for p in 0..4 {
                        for j in 0..4 {
                            let c = ((word >> (8 * j + 2 * p)) & 3) as i32;
                            acc += c * q[g * KG + 4 * p + j] as i32;
                        }
                    }
                }
                out[b * ROWS + lane] = acc;
            }
        }
    }
}

/// Whether the AVX-512 kernels can run on this CPU (F, BW, VNNI and one of GFNI / VBMI).
pub fn available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
            && (std::is_x86_feature_detected!("gfni")
                || std::is_x86_feature_detected!("avx512vbmi"))
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        false
    }
}

/// `true` when the decode uses GFNI (`vgf2p8affineqb`), else VBMI `vpmultishiftqb` + AND.
pub fn uses_gfni() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        std::is_x86_feature_detected!("gfni")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        false
    }
}

/// Which kernel [`gemv`] / [`layer`] dispatch to on this CPU.
pub fn level() -> &'static str {
    if !available() {
        "scalar"
    } else if uses_gfni() {
        "avx512-vnni-gfni lanes"
    } else {
        "avx512-vnni-vbmi lanes"
    }
}

/// `out[r] = sum_c code[r][c] * q[c]` for every padded row (codes `w + 1`; callers subtract
/// `sum(q)`). Uses AVX-512 when [`available`], else the scalar decode.
pub fn gemv(p: &PackedLanes, q: &[i8], out: &mut [i32]) {
    assert!(q.len() >= p.q_len(), "q shorter than PackedLanes::q_len()");
    assert!(out.len() >= p.n_padded(), "out shorter than n_padded()");
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if available() {
        // SAFETY: features detected; lengths checked above.
        unsafe {
            if uses_gfni() {
                gemv_avx512_gfni(p, q, out.as_mut_ptr(), 0..p.blocks)
            } else {
                gemv_avx512_vbmi(p, q, out.as_mut_ptr(), 0..p.blocks)
            }
        }
        return;
    }
    p.gemv_scalar(q, out)
}

/// Epilogue parameters of one layer.
#[derive(Clone, Copy)]
pub struct Epi<'a> {
    pub sum_q: i32,
    pub s: f32,
    pub scale: &'a [f32],
    pub bias: &'a [f32],
    pub act: Act,
}

/// One fused layer: `dst[r] = act(((acc[r] - sum_q) / s) * scale[r] + bias[r])` for
/// `r < dst.len()` (`dst.len() == p.n`), and returns `max_r |dst[r]|` (the next layer's
/// quantizer input), computed in the same pass. Bit-identical to [`gemv`] followed by
/// [`crate::cpu_packed::epilogue`]. With a `pool`, blocks are split across its threads.
pub fn layer(
    p: &PackedLanes,
    q: &[i8],
    e: Epi<'_>,
    dst: &mut [f32],
    pool: Option<&crate::spin_pool::SpinPool>,
) -> f32 {
    assert!(q.len() >= p.q_len(), "q shorter than PackedLanes::q_len()");
    assert_eq!(dst.len(), p.n, "dst must have n entries");
    assert!(e.scale.len() >= p.n && e.bias.len() >= p.n);
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if available() {
        let gf = uses_gfni();
        let run = |blocks: std::ops::Range<usize>, dst: *mut f32| -> f32 {
            // SAFETY: features detected; q, scale, bias and dst lengths checked above; each
            // caller passes a block range whose rows lie inside dst.
            unsafe {
                if gf {
                    layer_avx512_gfni(p, q, &e, dst, blocks)
                } else {
                    layer_avx512_vbmi(p, q, &e, dst, blocks)
                }
            }
        };
        if let Some(pool) = pool {
            let t = pool.threads().min(p.blocks);
            if t > 1 {
                // Per-thread maxima on the stack (no allocation on the hot path).
                const MAXT: usize = 64;
                let t = t.min(MAXT);
                let per = p.blocks.div_ceil(t);
                let maxes: [std::sync::atomic::AtomicU32; MAXT] =
                    std::array::from_fn(|_| std::sync::atomic::AtomicU32::new(0));
                let d = SyncPtr(dst.as_mut_ptr());
                pool.run(|i| {
                    let (b0, b1) = ((i * per).min(p.blocks), ((i + 1) * per).min(p.blocks));
                    if i < t && b0 < b1 {
                        let dd = &d;
                        let m = run(b0..b1, dd.0);
                        maxes[i].store(m.to_bits(), std::sync::atomic::Ordering::Relaxed);
                    }
                });
                return maxes
                    .iter()
                    .map(|m| f32::from_bits(m.load(std::sync::atomic::Ordering::Relaxed)))
                    .fold(0.0f32, f32::max);
            }
        }
        return run(0..p.blocks, dst.as_mut_ptr());
    }
    let _ = pool;
    let mut acc = vec![0i32; p.n_padded()];
    p.gemv_scalar(q, &mut acc);
    crate::cpu_packed::epilogue(&acc, e.sum_q, e.s, e.scale, e.bias, e.act, dst);
    dst.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
struct SyncPtr(*mut f32);
// SAFETY: used only for disjoint per-thread block ranges inside `layer`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl Sync for SyncPtr {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[clippy::msrv = "1.89"]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// `vgf2p8affineqb` matrices for sub-vector `p`: output bit `i` of each byte is the parity
    /// of (matrix byte `7 - i`) AND input, so bits 0 and 1 select input bits `2p` and `2p + 1`.
    const GF_MATS: [u64; 4] = {
        let mut m = [0u64; 4];
        let mut p = 0;
        while p < 4 {
            m[p] = ((1u64 << (2 * p)) << 56) | ((1u64 << (2 * p + 1)) << 48);
            p += 1;
        }
        m
    };

    /// `vpmultishiftqb` controls: output byte `j` of each qword takes bits `[8j + 2p, +8)`.
    const MS_CTLS: [u64; 4] = {
        let mut c = [0u64; 4];
        let mut p = 0;
        while p < 4 {
            let mut j = 0;
            while j < 8 {
                c[p] |= ((8 * j + 2 * p) as u64) << (8 * j);
                j += 1;
            }
            p += 1;
        }
        c
    };

    /// Decoders for sub-vector `p` of a packed vector: codes `(v >> 2p) & 3` per byte.
    pub(super) trait Decode {
        unsafe fn new() -> Self;
        unsafe fn part(&self, v: __m512i, p: usize) -> __m512i;
    }

    pub(super) struct Gfni {
        m: [__m512i; 4],
    }
    impl Decode for Gfni {
        #[inline(always)]
        unsafe fn new() -> Self {
            Self {
                m: GF_MATS.map(|m| _mm512_set1_epi64(m as i64)),
            }
        }
        #[inline(always)]
        unsafe fn part(&self, v: __m512i, p: usize) -> __m512i {
            if p == 0 {
                // A plain AND is as cheap and runs on either vector port.
                _mm512_and_si512(v, _mm512_set1_epi8(3))
            } else {
                _mm512_gf2p8affine_epi64_epi8::<0>(v, self.m[p])
            }
        }
    }

    pub(super) struct Vbmi {
        c: [__m512i; 4],
        m3: __m512i,
    }
    impl Decode for Vbmi {
        #[inline(always)]
        unsafe fn new() -> Self {
            Self {
                c: MS_CTLS.map(|c| _mm512_set1_epi64(c as i64)),
                m3: _mm512_set1_epi8(3),
            }
        }
        #[inline(always)]
        unsafe fn part(&self, v: __m512i, p: usize) -> __m512i {
            if p == 0 {
                _mm512_and_si512(v, self.m3)
            } else {
                _mm512_and_si512(_mm512_multishift_epi64_epi8(self.c[p], v), self.m3)
            }
        }
    }

    /// Row sums (codes · q) of `NB` consecutive blocks starting at `b0`, lane `r` = row `r`.
    #[inline(always)]
    pub(super) unsafe fn blocks<const NB: usize, D: Decode>(
        d: &D,
        p: &PackedLanes,
        q: *const i8,
        b0: usize,
    ) -> [__m512i; NB] {
        let g_n = p.groups;
        let base = p.data.as_ptr();
        let mut acc = [[_mm512_setzero_si512(); 4]; NB];
        for g in 0..g_n {
            let qp = q.add(g * KG) as *const i32;
            let qb = [
                _mm512_set1_epi32(qp.read_unaligned()),
                _mm512_set1_epi32(qp.add(1).read_unaligned()),
                _mm512_set1_epi32(qp.add(2).read_unaligned()),
                _mm512_set1_epi32(qp.add(3).read_unaligned()),
            ];
            for i in 0..NB {
                let v = _mm512_loadu_si512(base.add(((b0 + i) * g_n + g) * ROWS) as *const _);
                for (part, qv) in qb.iter().enumerate() {
                    acc[i][part] = _mm512_dpbusd_epi32(acc[i][part], d.part(v, part), *qv);
                }
            }
        }
        let mut out = [_mm512_setzero_si512(); NB];
        for i in 0..NB {
            out[i] = _mm512_add_epi32(
                _mm512_add_epi32(acc[i][0], acc[i][1]),
                _mm512_add_epi32(acc[i][2], acc[i][3]),
            );
        }
        out
    }

    #[inline(always)]
    unsafe fn step1<D: Decode>(
        d: &D,
        acc: &mut [__m512i; 4],
        base: *const u32,
        q: *const i8,
        g: usize,
    ) {
        let qp = q.add(g * KG) as *const i32;
        let v = _mm512_loadu_si512(base.add(g * ROWS) as *const _);
        for (part, a) in acc.iter_mut().enumerate() {
            let qv = _mm512_set1_epi32(qp.add(part).read_unaligned());
            *a = _mm512_dpbusd_epi32(*a, d.part(v, part), qv);
        }
    }

    /// One block with two accumulator sets (even / odd groups), so the `vpdpbusd` latency
    /// chains are half as long when there is no other block to interleave with.
    #[inline(always)]
    pub(super) unsafe fn block1<D: Decode>(
        d: &D,
        p: &PackedLanes,
        q: *const i8,
        b0: usize,
    ) -> __m512i {
        let g_n = p.groups;
        let base = p.data.as_ptr().add(b0 * g_n * ROWS);
        let mut acc = [[_mm512_setzero_si512(); 4]; 2];
        let mut g = 0;
        while g + 2 <= g_n {
            step1(d, &mut acc[0], base, q, g);
            step1(d, &mut acc[1], base, q, g + 1);
            g += 2;
        }
        if g < g_n {
            step1(d, &mut acc[0], base, q, g);
        }
        let a0 = _mm512_add_epi32(acc[0][0], acc[1][0]);
        let a1 = _mm512_add_epi32(acc[0][1], acc[1][1]);
        let a2 = _mm512_add_epi32(acc[0][2], acc[1][2]);
        let a3 = _mm512_add_epi32(acc[0][3], acc[1][3]);
        _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3))
    }

    #[inline(always)]
    pub(super) unsafe fn gemv_range<D: Decode>(
        p: &PackedLanes,
        q: &[i8],
        out: *mut i32,
        range: std::ops::Range<usize>,
    ) {
        let d = D::new();
        let qp = q.as_ptr();
        let mut b = range.start;
        while b + 4 <= range.end {
            let s = blocks::<4, D>(&d, p, qp, b);
            for (i, v) in s.iter().enumerate() {
                _mm512_storeu_si512(out.add((b + i) * ROWS) as *mut _, *v);
            }
            b += 4;
        }
        while b < range.end {
            _mm512_storeu_si512(out.add(b * ROWS) as *mut _, block1::<D>(&d, p, qp, b));
            b += 1;
        }
    }

    /// First half of the epilogue (same operations as `cpu_packed::epilogue_avx512`):
    /// `(acc - sum_q) as f32 / s`. Split out so the long-latency `vdivps` of one group of blocks
    /// can be issued before the next group's dot products and overlap with them.
    #[inline(always)]
    unsafe fn epi_div(acc: __m512i, e: &Epi<'_>) -> __m512 {
        let v = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc, _mm512_set1_epi32(e.sum_q)));
        _mm512_div_ps(v, _mm512_set1_ps(e.s))
    }

    /// Second half: `* scale + bias`, activation, masked store; returns lane-wise |result|
    /// (zero in masked-off lanes).
    #[inline(always)]
    unsafe fn epi_finish(v: __m512, e: &Epi<'_>, row0: usize, n: usize, dst: *mut f32) -> __m512 {
        let rem = n - row0;
        let m: u16 = if rem >= ROWS {
            u16::MAX
        } else {
            (1u16 << rem) - 1
        };
        let zero = _mm512_setzero_ps();
        let mut v = _mm512_mul_ps(v, _mm512_maskz_loadu_ps(m, e.scale.as_ptr().add(row0)));
        v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(m, e.bias.as_ptr().add(row0)));
        v = match e.act {
            Act::Identity => v,
            // max(v, 0) returns the second operand when v is NaN or both are zero, which matches
            // `if v > 0.0 { v } else { 0.0 }` exactly (NaN -> 0, -0.0 -> +0.0).
            Act::Relu => _mm512_max_ps(v, zero),
            Act::Relu2 => {
                let r = _mm512_max_ps(v, zero);
                _mm512_mul_ps(r, r)
            }
        };
        _mm512_mask_storeu_ps(dst.add(row0), m, v);
        _mm512_mask_abs_ps(zero, m, v)
    }

    #[inline(always)]
    pub(super) unsafe fn layer_range<D: Decode>(
        p: &PackedLanes,
        q: &[i8],
        e: &Epi<'_>,
        dst: *mut f32,
        range: std::ops::Range<usize>,
    ) -> f32 {
        let d = D::new();
        let qp = q.as_ptr();
        let mut mx = _mm512_setzero_ps();
        let mut b = range.start;
        // Software pipeline over groups of 4 blocks: the divisions of group i are issued right
        // before group i + 1's dot products, and finished (scale, bias, act, store) after them.
        let mut pv = [_mm512_setzero_ps(); 4];
        let mut pb = usize::MAX;
        while b + 4 <= range.end {
            let s = blocks::<4, D>(&d, p, qp, b);
            if pb != usize::MAX {
                for i in 0..4 {
                    mx = _mm512_max_ps(mx, epi_finish(pv[i], e, (pb + i) * ROWS, p.n, dst));
                }
            }
            for i in 0..4 {
                pv[i] = epi_div(s[i], e);
            }
            pb = b;
            b += 4;
        }
        if pb != usize::MAX {
            for i in 0..4 {
                mx = _mm512_max_ps(mx, epi_finish(pv[i], e, (pb + i) * ROWS, p.n, dst));
            }
        }
        while b + 2 <= range.end {
            let s = blocks::<2, D>(&d, p, qp, b);
            for (i, v) in s.iter().enumerate() {
                let v = epi_div(*v, e);
                mx = _mm512_max_ps(mx, epi_finish(v, e, (b + i) * ROWS, p.n, dst));
            }
            b += 2;
        }
        if b < range.end {
            let v = epi_div(block1::<D>(&d, p, qp, b), e);
            mx = _mm512_max_ps(mx, epi_finish(v, e, b * ROWS, p.n, dst));
        }
        // Max is exact and order independent for non-NaN values.
        _mm512_reduce_max_ps(mx)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,gfni")]
#[clippy::msrv = "1.89"]
unsafe fn gemv_avx512_gfni(p: &PackedLanes, q: &[i8], out: *mut i32, r: std::ops::Range<usize>) {
    x86::gemv_range::<x86::Gfni>(p, q, out, r)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
#[clippy::msrv = "1.89"]
unsafe fn gemv_avx512_vbmi(p: &PackedLanes, q: &[i8], out: *mut i32, r: std::ops::Range<usize>) {
    x86::gemv_range::<x86::Vbmi>(p, q, out, r)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,gfni")]
#[clippy::msrv = "1.89"]
unsafe fn layer_avx512_gfni(
    p: &PackedLanes,
    q: &[i8],
    e: &Epi<'_>,
    dst: *mut f32,
    r: std::ops::Range<usize>,
) -> f32 {
    x86::layer_range::<x86::Gfni>(p, q, e, dst, r)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
#[clippy::msrv = "1.89"]
unsafe fn layer_avx512_vbmi(
    p: &PackedLanes,
    q: &[i8],
    e: &Epi<'_>,
    dst: *mut f32,
    r: std::ops::Range<usize>,
) -> f32 {
    x86::layer_range::<x86::Vbmi>(p, q, e, dst, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    const SHAPES: &[(usize, usize)] = &[
        (1, 16),
        (1, 256),
        (5, 64),
        (16, 256),
        (17, 80),
        (33, 1024),
        (64, 1040),
        (100, 336),
        (256, 64),
        (3, 4096),
    ];

    fn case(rng: &mut StdRng, n: usize, k: usize) -> (Vec<i8>, PackedLanes, Vec<i8>) {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let p = PackedLanes::from_ternary(&w, n, k);
        let mut q: Vec<i8> = (0..p.q_len()).map(|_| rng.random()).collect();
        // Garbage past k must not matter.
        for v in &mut q[k..] {
            *v = -128;
        }
        (w, p, q)
    }

    fn naive(w: &[i8], q: &[i8], n: usize, k: usize) -> Vec<i32> {
        (0..n)
            .map(|r| {
                (0..k)
                    .map(|c| (w[r * k + c] as i32 + 1) * q[c] as i32)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn lanes_gemv_matches_naive() {
        let mut rng = StdRng::seed_from_u64(21);
        for &(n, k) in SHAPES {
            let (w, p, q) = case(&mut rng, n, k);
            let want = naive(&w, &q, n, k);
            let mut fast = vec![0; p.n_padded()];
            let mut slow = vec![0; p.n_padded()];
            gemv(&p, &q, &mut fast);
            p.gemv_scalar(&q, &mut slow);
            assert_eq!(&slow[..n], &want[..], "scalar {n}x{k}");
            assert_eq!(&fast[..n], &want[..], "simd {n}x{k}");
            assert!(fast[n..].iter().all(|&v| v == 0), "padding rows {n}x{k}");
        }
    }

    /// Both decoders (GFNI and VBMI) when the CPU has them, whichever `gemv` picks.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn both_decoders_match_naive() {
        if !available() {
            eprintln!("skipped: no AVX-512 VNNI");
            return;
        }
        let mut rng = StdRng::seed_from_u64(22);
        for &(n, k) in SHAPES {
            let (w, p, q) = case(&mut rng, n, k);
            let want = naive(&w, &q, n, k);
            if std::is_x86_feature_detected!("gfni") {
                let mut out = vec![0; p.n_padded()];
                // SAFETY: features detected above; buffer lengths match the layout.
                unsafe { gemv_avx512_gfni(&p, &q, out.as_mut_ptr(), 0..p.blocks) };
                assert_eq!(&out[..n], &want[..], "gfni {n}x{k}");
            }
            if std::is_x86_feature_detected!("avx512vbmi") {
                let mut out = vec![0; p.n_padded()];
                // SAFETY: features detected above; buffer lengths match the layout.
                unsafe { gemv_avx512_vbmi(&p, &q, out.as_mut_ptr(), 0..p.blocks) };
                assert_eq!(&out[..n], &want[..], "vbmi {n}x{k}");
            }
        }
    }

    #[test]
    fn fused_layer_matches_gemv_plus_epilogue() {
        let mut rng = StdRng::seed_from_u64(23);
        let pool = crate::spin_pool::SpinPool::new(3);
        for &(n, k) in SHAPES {
            let (_, p, q) = case(&mut rng, n, k);
            let scale: Vec<f32> = (0..n).map(|_| rng.random_range(0.001f32..0.1)).collect();
            let bias: Vec<f32> = (0..n).map(|_| rng.random_range(-1f32..1.0)).collect();
            let sum_q: i32 = q[..k].iter().map(|&v| v as i32).sum();
            let s = rng.random_range(1f32..200.0);
            for act in [Act::Identity, Act::Relu, Act::Relu2] {
                let mut acc = vec![0; p.n_padded()];
                p.gemv_scalar(&q, &mut acc);
                let mut want = vec![0f32; n];
                crate::cpu_packed::epilogue(&acc, sum_q, s, &scale, &bias, act, &mut want);
                let want_max = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                let e = Epi {
                    sum_q,
                    s,
                    scale: &scale,
                    bias: &bias,
                    act,
                };
                for pl in [None, Some(&pool)] {
                    let mut got = vec![f32::NAN; n];
                    let m = layer(&p, &q, e, &mut got, pl);
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&got),
                        bits(&want),
                        "{n}x{k} {act:?} pool {}",
                        pl.is_some()
                    );
                    assert_eq!(m.to_bits(), want_max.to_bits(), "max {n}x{k} {act:?}");
                }
            }
        }
    }

    #[test]
    fn quantize_with_max_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(24);
        for len in [1usize, 15, 16, 17, 64, 100, 256, 1024] {
            for scale in [1e-7f32, 0.01, 1.0, 300.0] {
                let x: Vec<f32> = (0..len)
                    .map(|_| rng.random_range(-1f32..1.0) * scale)
                    .collect();
                let mut a = vec![0i8; len];
                let mut b = vec![0i8; len];
                let mut c = vec![0i8; len];
                let ra = crate::mlp::quantize_scalar(&x, &mut a);
                let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                let rb = crate::mlp::quantize_with_max(&x, &mut b, m);
                let rc = crate::mlp::quantize_into(&x, &mut c);
                assert_eq!(
                    (ra.0.to_bits(), ra.1),
                    (rb.0.to_bits(), rb.1),
                    "with_max {len}"
                );
                assert_eq!((ra.0.to_bits(), ra.1), (rc.0.to_bits(), rc.1), "into {len}");
                assert_eq!(a, b, "with_max q {len}");
                assert_eq!(a, c, "into q {len}");
            }
        }
    }
}
