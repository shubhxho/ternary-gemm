//! CPU reference for the tensor-core path.
//!
//! Unpacks the natural 2-bit packed layout into int8 / fp16 `mma.sync` fragments (m16n8k16,
//! the same mapping as PTX on sm_80+) and accumulates a tiled GEMM/GEMV. This is the
//! correctness oracle for [`kernels/tensor_core.cu`](../../kernels/tensor_core.cu) when no
//! GPU is present. The int8 path is bit-identical to [`crate::reference::gemm_i8`].
//!
//! The 8× vs fp16 figure at batch 1 is a theoretical bandwidth ceiling (2-bit vs 16-bit
//! weight traffic), not a measured GPU result. TriRun and BitBLAS are the prior art that
//! reached near that ceiling on GEMV.

use crate::pack::PackedTernary;
use crate::reference::QuantizedActs;
use half::f16;

/// MMA M tile (output rows per warp).
pub const MMA_M: usize = 16;
/// MMA N tile (batch columns per MMA).
pub const MMA_N: usize = 8;
/// MMA K tile.
pub const MMA_K: usize = 16;

/// Decode 16 ternary weights from one natural-layout word. Codes are `w + 1` in `{0,1,2}`.
#[inline]
pub fn unpack_natural_word(word: u32) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (t, slot) in out.iter_mut().enumerate() {
        *slot = ((word >> (2 * t)) & 3) as i8 - 1;
    }
    out
}

/// Pack four consecutive weights `w[t0..t0+4]` as signed bytes in a `u32` (little-endian),
/// matching the int8 A/B fragment nibble the kernel feeds to `mma.sync`.
#[inline]
pub fn pack_w4(word: u32, t0: usize) -> u32 {
    let w = unpack_natural_word(word);
    let mut r = 0u32;
    for i in 0..4 {
        r |= (w[t0 + i] as u8 as u32) << (8 * i);
    }
    r
}

/// A-fragment for one lane of `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`.
/// Returns `(a0, a1)`: 4 signed bytes at rows `group` and `group+8`, columns `tidg*4..+4`.
#[inline]
pub fn pack_a_s8(a: &[[i8; MMA_K]; MMA_M], lane: usize) -> (u32, u32) {
    let group = lane >> 2;
    let tidg = lane & 3;
    let mut a0 = 0u32;
    let mut a1 = 0u32;
    for i in 0..4 {
        a0 |= (a[group][tidg * 4 + i] as u8 as u32) << (8 * i);
        a1 |= (a[group + 8][tidg * 4 + i] as u8 as u32) << (8 * i);
    }
    (a0, a1)
}

/// B-fragment for one lane: 4 signed bytes at `k = tidg*4..+4`, `n = group`.
#[inline]
pub fn pack_b_s8(b: &[[i8; MMA_N]; MMA_K], lane: usize) -> u32 {
    let group = lane >> 2;
    let tidg = lane & 3;
    let mut b0 = 0u32;
    for i in 0..4 {
        b0 |= (b[tidg * 4 + i][group] as u8 as u32) << (8 * i);
    }
    b0
}

/// Scatter one lane's four s32 accumulators into a 16×8 C tile.
#[inline]
pub fn scatter_c_s32(c: &mut [[i32; MMA_N]; MMA_M], lane: usize, d: [i32; 4]) {
    let group = lane >> 2;
    let tidg = lane & 3;
    let col0 = tidg * 2;
    // PTX ISA m16n8k16 accumulator layout: d0,d1 on row `group`, d2,d3 on row `group+8`.
    c[group][col0] = d[0];
    c[group][col0 + 1] = d[1];
    c[group + 8][col0] = d[2];
    c[group + 8][col0 + 1] = d[3];
}

/// Gather one lane's four s32 accumulators from a 16×8 C tile.
#[inline]
pub fn gather_c_s32(c: &[[i32; MMA_N]; MMA_M], lane: usize) -> [i32; 4] {
    let group = lane >> 2;
    let tidg = lane & 3;
    let col0 = tidg * 2;
    [
        c[group][col0],
        c[group][col0 + 1],
        c[group + 8][col0],
        c[group + 8][col0 + 1],
    ]
}

/// Rebuild the 16×16 A tile from the 32 lanes' s8 A fragments.
pub fn unpack_a_s8(frags: &[(u32, u32); 32]) -> [[i8; MMA_K]; MMA_M] {
    let mut a = [[0i8; MMA_K]; MMA_M];
    for lane in 0..32 {
        let group = lane >> 2;
        let tidg = lane & 3;
        let (a0, a1) = frags[lane];
        for i in 0..4 {
            a[group][tidg * 4 + i] = (a0 >> (8 * i)) as u8 as i8;
            a[group + 8][tidg * 4 + i] = (a1 >> (8 * i)) as u8 as i8;
        }
    }
    a
}

/// Rebuild the 16×8 B tile (K×N) from the 32 lanes' s8 B fragments.
pub fn unpack_b_s8(frags: &[u32; 32]) -> [[i8; MMA_N]; MMA_K] {
    let mut b = [[0i8; MMA_N]; MMA_K];
    for lane in 0..32 {
        let group = lane >> 2;
        let tidg = lane & 3;
        let b0 = frags[lane];
        for i in 0..4 {
            b[tidg * 4 + i][group] = (b0 >> (8 * i)) as u8 as i8;
        }
    }
    b
}

/// Dense 16×8×16 integer MMA (the warp-wide operation `mma.sync` computes).
pub fn mma_s8_dense(
    a: &[[i8; MMA_K]; MMA_M],
    b: &[[i8; MMA_N]; MMA_K],
    c: &mut [[i32; MMA_N]; MMA_M],
) {
    for m in 0..MMA_M {
        for n in 0..MMA_N {
            let mut acc = c[m][n];
            for k in 0..MMA_K {
                acc += a[m][k] as i32 * b[k][n] as i32;
            }
            c[m][n] = acc;
        }
    }
}

/// W2A8 GEMM with the tensor-core 16×8×16 tiling, unpacked from the natural layout.
/// Bit-identical to [`crate::reference::gemm_i8`].
pub fn gemm_i8(w: &PackedTernary, a: &QuantizedActs, residual: Option<&[f16]>) -> Vec<f16> {
    assert_eq!(a.k, w.k);
    let (n, k, batch) = (w.n, w.k, a.batch);
    let mut y = vec![f16::ZERO; batch * n];
    let kw = k / 16;
    let mut row0 = 0;
    while row0 < n {
        let mut b0 = 0;
        while b0 < batch {
            let mut acc = [[0i32; MMA_N]; MMA_M];
            for kb in 0..kw {
                let mut a_tile = [[0i8; MMA_K]; MMA_M];
                let mut b_tile = [[0i8; MMA_N]; MMA_K];
                for r in 0..MMA_M {
                    let row = row0 + r;
                    let word = if row < n {
                        w.natural[row * kw + kb]
                    } else {
                        0x5555_5555
                    };
                    a_tile[r] = unpack_natural_word(word);
                }
                for kk in 0..MMA_K {
                    for c in 0..MMA_N {
                        let b = b0 + c;
                        b_tile[kk][c] = if b < batch {
                            a.q[b * k + kb * 16 + kk]
                        } else {
                            0
                        };
                    }
                }
                // Round-trip through fragments so the test exercises the same mapping as the kernel.
                let mut a_frags = [(0u32, 0u32); 32];
                let mut b_frags = [0u32; 32];
                for lane in 0..32 {
                    a_frags[lane] = pack_a_s8(&a_tile, lane);
                    b_frags[lane] = pack_b_s8(&b_tile, lane);
                }
                let a_rt = unpack_a_s8(&a_frags);
                let b_rt = unpack_b_s8(&b_frags);
                mma_s8_dense(&a_rt, &b_rt, &mut acc);
            }
            for r in 0..MMA_M {
                let row = row0 + r;
                if row >= n {
                    continue;
                }
                let ws = w.row_scale[row];
                for c in 0..MMA_N {
                    let b = b0 + c;
                    if b >= batch {
                        continue;
                    }
                    let mut t = acc[r][c] as f32;
                    t /= a.s[b];
                    t *= ws;
                    if let Some(res) = residual {
                        t += res[b * n + row].to_f32();
                    }
                    y[b * n + row] = f16::from_f32(t);
                }
            }
            b0 += MMA_N;
        }
        row0 += MMA_M;
    }
    y
}

/// W2A16 GEMM with the same tiling, weights unpacked to fp16 (exact for {-1,0,+1}),
/// accumulated in f32. The final values are f32 (no rounding to f16) so a caller can
/// compare against [`crate::reference::gemm_f16`].
pub fn gemm_f16(w: &PackedTernary, x: &[f16], batch: usize, residual: Option<&[f16]>) -> Vec<f32> {
    let (n, k) = (w.n, w.k);
    assert_eq!(x.len(), batch * k);
    let mut y = vec![0f32; batch * n];
    let kw = k / 16;
    let mut row0 = 0;
    while row0 < n {
        let mut b0 = 0;
        while b0 < batch {
            let mut acc = [[0f32; MMA_N]; MMA_M];
            for kb in 0..kw {
                for r in 0..MMA_M {
                    let row = row0 + r;
                    let word = if row < n {
                        w.natural[row * kw + kb]
                    } else {
                        0x5555_5555
                    };
                    let wt = unpack_natural_word(word);
                    for c in 0..MMA_N {
                        let b = b0 + c;
                        if b >= batch {
                            continue;
                        }
                        for kk in 0..MMA_K {
                            // f16-exact weights; multiply in f32, same as mma.f32.f16.f16.f32.
                            acc[r][c] += f16::from_f32(wt[kk] as f32).to_f32()
                                * x[b * k + kb * 16 + kk].to_f32();
                        }
                    }
                }
            }
            for r in 0..MMA_M {
                let row = row0 + r;
                if row >= n {
                    continue;
                }
                let ws = w.row_scale[row];
                for c in 0..MMA_N {
                    let b = b0 + c;
                    if b >= batch {
                        continue;
                    }
                    let mut t = acc[r][c] * ws;
                    if let Some(res) = residual {
                        t += res[b * n + row].to_f32();
                    }
                    y[b * n + row] = t;
                }
            }
            b0 += MMA_N;
        }
        row0 += MMA_M;
    }
    y
}

/// Parse `"compute_80"` / `"sm_120a"` into `(major, minor)`.
pub fn parse_cuda_arch(arch: &str) -> Option<(i32, i32)> {
    let s = arch.trim();
    let s = s
        .strip_prefix("compute_")
        .or_else(|| s.strip_prefix("sm_"))
        .unwrap_or(s);
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    match digits.len() {
        2 => Some((digits[..1].parse().ok()?, digits[1..].parse().ok()?)),
        3 => Some((digits[..2].parse().ok()?, digits[2..].parse().ok()?)),
        _ => None,
    }
}

/// True for virtual or real arch sm_80 / compute_80 and newer (mma.sync required).
pub fn arch_has_mma(arch: &str) -> bool {
    parse_cuda_arch(arch).is_some_and(|(maj, _)| maj >= 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_natural_word_covers_all_codes() {
        // 16 codes of 0,1,2 packed naturally.
        let mut word = 0u32;
        let mut expect = [0i8; 16];
        for t in 0..16 {
            let c = (t % 3) as u32;
            word |= c << (2 * t);
            expect[t] = c as i8 - 1;
        }
        assert_eq!(unpack_natural_word(word), expect);
        assert_eq!(unpack_natural_word(0x5555_5555), [0i8; 16]);
        // pack_w4 bytes are the signed weights.
        for t0 in [0, 4, 8, 12] {
            let p = pack_w4(word, t0);
            for i in 0..4 {
                assert_eq!((p >> (8 * i)) as u8 as i8, expect[t0 + i]);
            }
        }
    }

    #[test]
    fn s8_fragment_roundtrip() {
        let mut a = [[0i8; MMA_K]; MMA_M];
        let mut b = [[0i8; MMA_N]; MMA_K];
        for m in 0..MMA_M {
            for k in 0..MMA_K {
                a[m][k] = (((m * 3 + k) % 3) as i8) - 1;
            }
        }
        for k in 0..MMA_K {
            for n in 0..MMA_N {
                b[k][n] = (((k + 2 * n) % 5) as i8) - 2; // int8 activations, not just ternary
            }
        }
        let mut a_frags = [(0u32, 0u32); 32];
        let mut b_frags = [0u32; 32];
        for lane in 0..32 {
            a_frags[lane] = pack_a_s8(&a, lane);
            b_frags[lane] = pack_b_s8(&b, lane);
        }
        assert_eq!(unpack_a_s8(&a_frags), a);
        assert_eq!(unpack_b_s8(&b_frags), b);
        let mut c = [[0i32; MMA_N]; MMA_M];
        mma_s8_dense(&a, &b, &mut c);
        // Scatter / gather of C is a bijection.
        let mut c2 = [[0i32; MMA_N]; MMA_M];
        for lane in 0..32 {
            scatter_c_s32(&mut c2, lane, gather_c_s32(&c, lane));
        }
        assert_eq!(c, c2);
    }

    #[test]
    fn parse_cuda_arch_sm80_plus() {
        assert_eq!(parse_cuda_arch("compute_80"), Some((8, 0)));
        assert_eq!(parse_cuda_arch("sm_89"), Some((8, 9)));
        assert_eq!(parse_cuda_arch("compute_120"), Some((12, 0)));
        assert_eq!(parse_cuda_arch("compute_120a"), Some((12, 0)));
        assert_eq!(parse_cuda_arch("compute_61"), Some((6, 1)));
        assert!(arch_has_mma("compute_80"));
        assert!(arch_has_mma("sm_120"));
        assert!(!arch_has_mma("compute_75"));
        assert!(!arch_has_mma("compute_61"));
    }
}
