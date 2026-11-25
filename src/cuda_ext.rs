//! CPU side of the extra GEMV kernels in
//! [`kernels/ternary_ext.cu`](../../kernels/ternary_ext.cu): their exact arithmetic contract
//! ([`reference_gemv`], [`reference_gemm`], with a fused activation), bit-level emulations of the
//! kernels whose data flow differs from the main dp4a kernel ([`emulate_lut`], [`emulate_splitk`],
//! [`emulate_batched`]), a model of the warp-specialized kernel's barrier protocol
//! ([`simulate_warpspec`]), and the launch heuristics the GPU wrapper uses ([`splitk_factor`],
//! [`pick_ext_kernel`], shared-memory sizes). The bulk kernel has the same math as cp.async r2
//! (only the prefetch instruction changes), so it is covered by [`reference_gemv`].
//!
//! No GPU is needed for anything here, so the tests run everywhere; the CUDA kernels themselves
//! are compile-checked with NVRTC/ptxas in CI and only run on a real device (`tests/gpu_ext.rs`).

use crate::mlp::Act;
use crate::pack::PackedTernary;
use crate::reference::QuantizedActs;
use half::f16;

/// The CUDA source of the extra kernels.
pub const EXT_SOURCE: &str = include_str!("../kernels/ternary_ext.cu");

/// The hand-written PTX GEMV ([`kernels/ternary_gemv.ptx`](../../kernels/ternary_gemv.ptx)):
/// batch-1, permuted layout, one warp per row, identity epilogue. Entry point
/// [`HAND_PTX_KERNEL`]; assembles for sm_61+.
pub const HAND_PTX: &str = include_str!("../kernels/ternary_gemv.ptx");
/// Entry point of [`HAND_PTX`].
pub const HAND_PTX_KERNEL: &str = "ternary_gemv_ptx";

/// Every `extern "C"` entry point in [`EXT_SOURCE`].
pub const EXT_KERNELS: &[&str] = &[
    "tgx_gemv_cpasync_r1",
    "tgx_gemv_cpasync_r2",
    "tgx_gemv_cpasync_r4",
    "tgx_gemv_splitk_r2",
    "tgx_splitk_finalize",
    "tgx_gemv_lut",
    "tgx_gemv_warpspec",
    "tgx_gemv_batched",
    "tgx_gemv_bulk_r2",
];

/// Words per permuted chunk (`TGX_CHUNK_WORDS`, same as the main kernels).
pub const CHUNK_WORDS: usize = 128;
/// Warps per block of the cp.async kernel (`TGX_CP_WARPS`).
pub const CP_WARPS: usize = 4;
/// Ring depth of the cp.async kernel (`TGX_CP_STAGES`).
pub const CP_STAGES: usize = 4;
/// Rows per block of the LUT kernel (`TGX_LUT_ROWS`).
pub const LUT_ROWS: usize = 64;
/// K tile of the LUT kernel (`TGX_LUT_KT`).
pub const LUT_KT: usize = 2048;
/// Rows per block of the warp-specialized kernel (`TGX_WS_ROWS`).
pub const WS_ROWS: usize = 8;
/// Threads per block of the warp-specialized kernel (1 producer + 4 consumer warps).
pub const WS_THREADS: usize = 160;
/// Warps per block of the batched kernel (`TGX_BAT_WARPS`).
pub const BAT_WARPS: usize = 4;
/// Rows per warp of the batched kernel (`TGX_BAT_R`).
pub const BAT_R: usize = 2;
/// Ring depth of the batched kernel (`TGX_BAT_STAGES`).
pub const BAT_STAGES: usize = 2;
/// Largest batch `tgx_gemv_batched` accepts (`TGX_BAT_MAX`).
pub const BAT_MAX: usize = 16;

/// The kernels' epilogue for one output: `act((acc - sum) / s * scale) (+ residual)` as fp16.
#[inline]
pub fn epilogue(
    acc_codes: i32,
    sum: i32,
    s: f32,
    scale: f32,
    act: Act,
    residual: Option<f16>,
) -> f16 {
    let mut t = (acc_codes - sum) as f32;
    t /= s;
    t *= scale;
    t = match act {
        Act::Identity => t,
        Act::Relu => {
            if t > 0.0 {
                t
            } else {
                0.0
            }
        }
        Act::Relu2 => {
            let r = if t > 0.0 { t } else { 0.0 };
            r * r
        }
    };
    if let Some(r) = residual {
        t += r.to_f32();
    }
    f16::from_f32(t)
}

/// Batch-1 reference for every extra kernel: `y[r] = epilogue(sum_k w[r][k] * q[k])`.
/// With `Act::Identity` it equals [`crate::reference::gemm_i8`] bit for bit.
pub fn reference_gemv(
    w: &PackedTernary,
    a: &QuantizedActs,
    residual: Option<&[f16]>,
    act: Act,
) -> Vec<f16> {
    assert_eq!((a.batch, a.k), (1, w.k), "batch-1 activations matching K");
    (0..w.n)
        .map(|r| {
            let acc: i32 = (0..w.k).map(|c| w.get(r, c) as i32 * a.q[c] as i32).sum();
            // Kernels accumulate codes (w + 1) and subtract sum(q); same integer.
            epilogue(
                acc + a.sum[0],
                a.sum[0],
                a.s[0],
                w.row_scale[r],
                act,
                residual.map(|v| v[r]),
            )
        })
        .collect()
}

/// Batched W2A8 GEMM reference: `y[b*n + r] = epilogue(sum_k w[r][k] * q[b][k])`.
/// Batch-1 equals [`reference_gemv`] bit for bit. Residual, if any, is laid out `[batch][n]`.
pub fn reference_gemm(
    w: &PackedTernary,
    a: &QuantizedActs,
    residual: Option<&[f16]>,
    act: Act,
) -> Vec<f16> {
    assert_eq!(a.k, w.k, "activations matching K");
    let mut out = vec![f16::from_f32(0.0); a.batch * w.n];
    for b in 0..a.batch {
        let q = &a.q[b * a.k..(b + 1) * a.k];
        for r in 0..w.n {
            let acc: i32 = (0..w.k).map(|c| w.get(r, c) as i32 * q[c] as i32).sum();
            out[b * w.n + r] = epilogue(
                acc + a.sum[b],
                a.sum[b],
                a.s[b],
                w.row_scale[r],
                act,
                residual.map(|v| v[b * w.n + r]),
            );
        }
    }
    out
}

/// dp4a over one permuted chunk for one row, exactly as `tgx_chunk_dot` walks it.
fn chunk_dot(words: &[u32], q: &[i8], base: usize, l: usize) -> i32 {
    let mut acc = 0i32;
    for lane in 0..l {
        for qq in 0..4 {
            let word = words[base + 4 * lane + qq];
            let xi = 16 * (base + qq * l + lane);
            for j in 0..4 {
                let codes = (word >> (2 * j)) & 0x0303_0303;
                for b in 0..4 {
                    let c = ((codes >> (8 * b)) & 0xff) as i32;
                    acc += c * q[xi + 4 * j + b] as i32;
                }
            }
        }
    }
    acc
}

/// Emulates `tgx_gemv_splitk_r2` + `tgx_splitk_finalize`: `splits` chunk ranges summed into
/// int32 partials (order independent), then the epilogue.
pub fn emulate_splitk(
    w: &PackedTernary,
    a: &QuantizedActs,
    splits: usize,
    residual: Option<&[f16]>,
    act: Act,
) -> Vec<f16> {
    assert_eq!((a.batch, a.k), (1, w.k));
    let kw = w.k / 16;
    let chunks = kw.div_ceil(CHUNK_WORDS);
    let per = chunks.div_ceil(splits.max(1));
    let mut part = vec![0i32; w.n];
    // Visit splits in reverse to make the point that the atomics' order does not matter.
    for s in (0..splits.max(1)).rev() {
        let (c0, c1) = (s * per, ((s + 1) * per).min(chunks));
        for (r, p) in part.iter_mut().enumerate() {
            let words = &w.permuted[r * kw..(r + 1) * kw];
            for c in c0..c1 {
                let base = c * CHUNK_WORDS;
                let l = CHUNK_WORDS.min(kw - base) / 4;
                *p += chunk_dot(words, &a.q, base, l);
            }
        }
    }
    (0..w.n)
        .map(|r| {
            epilogue(
                part[r],
                a.sum[0],
                a.s[0],
                w.row_scale[r],
                act,
                residual.map(|v| v[r]),
            )
        })
        .collect()
}

/// The 16-entry table `tgx_gemv_lut` builds for activations `(x0, x1)`: `T[c0 + 4*c1]`.
pub fn lut16(x0: i8, x1: i8) -> [i16; 16] {
    let mut t = [0i16; 16];
    for (idx, v) in t.iter_mut().enumerate() {
        let (c0, c1) = ((idx & 3) as i32, (idx >> 2) as i32);
        *v = (c0 * x0 as i32 + c1 * x1 as i32) as i16;
    }
    t
}

/// Emulates `tgx_gemv_lut` on the natural layout: per K tile, build int16 tables for every
/// activation pair, then one lookup per weight nibble.
pub fn emulate_lut(
    w: &PackedTernary,
    a: &QuantizedActs,
    residual: Option<&[f16]>,
    act: Act,
) -> Vec<f16> {
    assert_eq!((a.batch, a.k), (1, w.k));
    let kw = w.k / 16;
    let mut acc = vec![0i32; w.n];
    let mut k0 = 0;
    while k0 < w.k {
        let kt = LUT_KT.min(w.k - k0);
        let tables: Vec<[i16; 16]> = (0..kt / 2)
            .map(|p| lut16(a.q[k0 + 2 * p], a.q[k0 + 2 * p + 1]))
            .collect();
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let row = &w.natural[r * kw + k0 / 16..r * kw + (k0 + kt) / 16];
            for (wi, &word) in row.iter().enumerate() {
                for m in 0..8 {
                    *acc_r += tables[wi * 8 + m][((word >> (4 * m)) & 15) as usize] as i32;
                }
            }
        }
        k0 += kt;
    }
    (0..w.n)
        .map(|r| {
            epilogue(
                acc[r],
                a.sum[0],
                a.s[0],
                w.row_scale[r],
                act,
                residual.map(|v| v[r]),
            )
        })
        .collect()
}

/// Outcome of [`simulate_warpspec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpspecTrace {
    /// Chunk order the consumers saw.
    pub consumed: Vec<usize>,
    /// Arrivals left pending on any named barrier at exit (must be zero: a leftover arrival
    /// would corrupt the next kernel's use of that barrier id).
    pub dangling: usize,
}

/// Steps the producer/consumer protocol of `tgx_gemv_warpspec` (2-stage ring, FULL ids 1-2,
/// EMPTY ids 3-4) as two interleaved state machines. Returns `Err` on deadlock or if the
/// producer could overwrite a stage the consumers have not finished.
pub fn simulate_warpspec(chunks: usize) -> Result<WarpspecTrace, String> {
    // Each named barrier completes when both sides have reached it; count per-id phases.
    #[derive(Default, Clone, Copy)]
    struct Bar {
        prod: usize,
        cons: usize,
    }
    let mut bars = [Bar::default(); 5];
    let (mut pc, mut cc) = (0usize, 0usize); // next chunk for producer / consumer
    let mut p_waiting_empty = false;
    let mut c_waiting_full = false;
    let mut stage_owner: [Option<usize>; 2] = [None, None]; // chunk currently in each stage
    let mut consumed = Vec::new();
    for _ in 0..(16 * chunks + 16) {
        let mut progressed = false;
        // Producer step.
        if pc < chunks {
            let s = pc & 1;
            if pc >= 2 && !p_waiting_empty {
                bars[3 + s].prod += 1; // bar.sync EMPTY[s]
                p_waiting_empty = true;
            }
            let empty_ok = pc < 2 || bars[3 + s].cons >= bars[3 + s].prod;
            if empty_ok {
                if let Some(old) = stage_owner[s] {
                    if !consumed.contains(&old) {
                        return Err(format!("producer overwrote chunk {old} in stage {s}"));
                    }
                }
                stage_owner[s] = Some(pc);
                bars[1 + s].prod += 1; // bar.arrive FULL[s]
                p_waiting_empty = false;
                pc += 1;
                progressed = true;
            }
        }
        // Consumer step.
        if cc < chunks {
            let s = cc & 1;
            if !c_waiting_full {
                bars[1 + s].cons += 1; // bar.sync FULL[s]
                c_waiting_full = true;
            }
            if bars[1 + s].prod >= bars[1 + s].cons {
                if stage_owner[s] != Some(cc) {
                    return Err(format!("consumer read stage {s} expecting chunk {cc}"));
                }
                consumed.push(cc);
                if cc + 2 < chunks {
                    bars[3 + s].cons += 1; // bar.arrive EMPTY[s]
                }
                c_waiting_full = false;
                cc += 1;
                progressed = true;
            }
        }
        if pc == chunks && cc == chunks {
            let dangling = bars.iter().map(|b| b.prod.abs_diff(b.cons)).sum();
            return Ok(WarpspecTrace { consumed, dangling });
        }
        if !progressed {
            return Err(format!(
                "deadlock at producer chunk {pc}, consumer chunk {cc}"
            ));
        }
    }
    Err("did not finish".into())
}

/// Split-K factor for an `n x k` batch-1 GEMV on `sm_count` SMs: split only when row blocks
/// (`n / 16` per 256-thread block of 2-row warps) cannot give every SM two blocks, and never
/// split below one 128-word chunk per split.
pub fn splitk_factor(sm_count: usize, n: usize, k: usize) -> usize {
    let row_blocks = n.div_ceil(16);
    let chunks = (k / 16).div_ceil(CHUNK_WORDS);
    let want = (2 * sm_count).div_ceil(row_blocks.max(1));
    want.clamp(1, chunks.max(1))
}

/// Dynamic shared memory of `tgx_gemv_cpasync_r{r}` for a given K.
pub fn cpasync_smem_bytes(k: usize, r: usize) -> usize {
    k + CP_WARPS * CP_STAGES * r * 512
}

/// Dynamic shared memory of `tgx_gemv_warpspec` for a given K.
pub fn warpspec_smem_bytes(k: usize) -> usize {
    k + 2 * WS_ROWS * 512
}

/// Dynamic shared memory of `tgx_gemv_batched` for a given batch (K-chunk of acts is 2048 B).
pub fn batched_smem_bytes(batch: usize) -> usize {
    batch * (CHUNK_WORDS * 16) + BAT_WARPS * BAT_STAGES * BAT_R * 512
}

/// Dynamic shared memory of `tgx_gemv_bulk_r2` (same footprint as cp.async r2).
pub fn bulk_smem_bytes(k: usize) -> usize {
    cpasync_smem_bytes(k, 2)
}

/// Emulates `tgx_gemv_batched`: walk K in 128-word chunks, reuse each chunk of weights across
/// the batch, then the same epilogue. Int32 adds commute, so this is bit-identical to
/// [`reference_gemm`] for any chunking.
pub fn emulate_batched(
    w: &PackedTernary,
    a: &QuantizedActs,
    residual: Option<&[f16]>,
    act: Act,
) -> Vec<f16> {
    assert_eq!(a.k, w.k);
    assert!(
        (1..=BAT_MAX).contains(&a.batch),
        "batched kernel supports batch 1..={BAT_MAX}"
    );
    let kw = w.k / 16;
    let chunks = kw.div_ceil(CHUNK_WORDS);
    let mut acc = vec![0i32; a.batch * w.n];
    for c in 0..chunks {
        let base = c * CHUNK_WORDS;
        let l = CHUNK_WORDS.min(kw - base) / 4;
        for r in 0..w.n {
            let words = &w.permuted[r * kw..(r + 1) * kw];
            // chunk_dot indexes q at 16*(base + ...). For the kernel, smem holds only this
            // chunk (base' = 0); use the full-q path here, which is the same integers.
            for b in 0..a.batch {
                acc[b * w.n + r] += chunk_dot(words, &a.q[b * a.k..(b + 1) * a.k], base, l);
            }
        }
    }
    (0..a.batch * w.n)
        .map(|i| {
            let (b, r) = (i / w.n, i % w.n);
            epilogue(
                acc[i],
                a.sum[b],
                a.s[b],
                w.row_scale[r],
                act,
                residual.map(|v| v[i]),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{gemm_i8, quantize_acts};
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    fn case(n: usize, k: usize, seed: u64) -> (PackedTernary, QuantizedActs, Vec<f16>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.1)).collect();
        let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-2.0..2.0)))
            .collect();
        let res: Vec<f16> = (0..n)
            .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        (p, quantize_acts(&x, 1, k, None, 1e-5), res)
    }

    fn bits(v: &[f16]) -> Vec<u16> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn identity_reference_matches_gemm_i8() {
        for &(n, k) in &[(3, 64), (40, 2112), (17, 4096)] {
            let (p, a, res) = case(n, k, 1);
            for r in [None, Some(&res[..])] {
                assert_eq!(
                    bits(&reference_gemv(&p, &a, r, Act::Identity)),
                    bits(&gemm_i8(&p, &a, r))
                );
            }
        }
    }

    #[test]
    fn splitk_and_lut_emulations_are_bit_exact() {
        for &(n, k) in &[(1, 64), (9, 2048), (33, 2112), (64, 4160), (5, 6144)] {
            let (p, a, res) = case(n, k, (n * k) as u64);
            for act in [Act::Identity, Act::Relu, Act::Relu2] {
                for r in [None, Some(&res[..])] {
                    let want = bits(&reference_gemv(&p, &a, r, act));
                    for splits in [1, 2, 3, 7, 64] {
                        assert_eq!(
                            want,
                            bits(&emulate_splitk(&p, &a, splits, r, act)),
                            "{n}x{k} s{splits}"
                        );
                    }
                    assert_eq!(want, bits(&emulate_lut(&p, &a, r, act)), "lut {n}x{k}");
                }
            }
        }
    }

    #[test]
    fn lut_entries_fit_int16() {
        for (x0, x1) in [(-128i8, -128i8), (127, 127), (-128, 127)] {
            let t = lut16(x0, x1);
            assert_eq!(t[0], 0);
            assert_eq!(t[1 + 4], x0 as i16 + x1 as i16);
            assert_eq!(t[2 + 8], 2 * x0 as i16 + 2 * x1 as i16);
        }
    }

    #[test]
    fn warpspec_protocol_has_no_deadlock_or_overwrite() {
        for chunks in 1..40 {
            let t = simulate_warpspec(chunks).unwrap();
            assert_eq!(t.consumed, (0..chunks).collect::<Vec<_>>());
            assert_eq!(t.dangling, 0, "chunks {chunks}");
        }
    }

    #[test]
    fn splitk_factor_fills_the_gpu_without_oversplitting() {
        // 170 SMs (RTX 5090). Wide layers need no split; narrow long-K layers do.
        assert_eq!(splitk_factor(170, 16384, 4096), 1);
        assert!(splitk_factor(170, 1024, 16384) > 1);
        assert!(splitk_factor(170, 16, 2048) <= 1);
        assert_eq!(splitk_factor(170, 16, 64), 1);
    }

    #[test]
    fn smem_budgets_fit_the_default_48k_for_k_up_to_8k() {
        for r in [1, 2, 4] {
            assert!(cpasync_smem_bytes(8192, r) <= 48 * 1024, "r{r}");
        }
        assert!(warpspec_smem_bytes(8192) <= 48 * 1024);
        for b in 1..=BAT_MAX {
            assert!(batched_smem_bytes(b) <= 48 * 1024, "batch {b}");
        }
        assert!(bulk_smem_bytes(8192) <= 48 * 1024);
    }

    #[test]
    fn reference_gemm_batch1_matches_gemv() {
        let (p, a, res) = case(17, 2112, 9);
        for act in [Act::Identity, Act::Relu, Act::Relu2] {
            for r in [None, Some(&res[..])] {
                assert_eq!(
                    bits(&reference_gemv(&p, &a, r, act)),
                    bits(&reference_gemm(&p, &a, r, act))
                );
            }
        }
    }

    #[test]
    fn batched_emulation_is_bit_exact() {
        use crate::reference::quantize_acts;
        let mut rng = StdRng::seed_from_u64(42);
        for &(n, k, batch) in &[
            (3usize, 64usize, 1usize),
            (9, 2048, 4),
            (33, 2112, 8),
            (16, 4160, 16),
            (5, 6144, 3),
        ] {
            let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
            let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.1)).collect();
            let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
            let x: Vec<f16> = (0..batch * k)
                .map(|_| f16::from_f32(rng.random_range(-2.0..2.0)))
                .collect();
            let res: Vec<f16> = (0..batch * n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
                .collect();
            let a = quantize_acts(&x, batch, k, None, 1e-5);
            for act in [Act::Identity, Act::Relu, Act::Relu2] {
                for r in [None, Some(&res[..])] {
                    let want = bits(&reference_gemm(&p, &a, r, act));
                    assert_eq!(
                        want,
                        bits(&emulate_batched(&p, &a, r, act)),
                        "{n}x{k} batch {batch} {act:?}"
                    );
                }
            }
        }
    }
}
