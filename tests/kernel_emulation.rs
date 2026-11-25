#![allow(clippy::needless_range_loop)]
//! Line-by-line Rust transliteration of the CUDA main loops (`ternary_i8_warp`,
//! `ternary_f16_warp`), executed lane by lane on the CPU. This checks the kernel's index math
//! (chunking, permutation, dp4a byte lanes, the PRMT magic-number decode) against the reference
//! without a GPU. It does not prove the CUDA code compiles to the same thing; the GPU tests do that.

use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use ternary_gemm::pack::{PackedTernary, CHUNK_WORDS};
use ternary_gemm::reference::{gemm_i8, quantize_acts};

/// CUDA `__byte_perm(x, y, s)`.
fn byte_perm(x: u32, y: u32, s: u32) -> u32 {
    let bytes = ((y as u64) << 32) | x as u64;
    let mut r = 0u32;
    for n in 0..4 {
        let sel = (s >> (4 * n)) & 0x7;
        r |= (((bytes >> (8 * sel)) & 0xff) as u32) << (8 * n);
    }
    r
}

/// CUDA `__dp4a(int a, int b, int c)` (signed bytes).
fn dp4a(a: i32, b: i32, c: i32) -> i32 {
    let mut acc = c;
    for i in 0..4 {
        acc += ((a >> (8 * i)) as i8 as i32) * ((b >> (8 * i)) as i8 as i32);
    }
    acc
}

fn emulate_i8_row(w: &PackedTernary, xq: &[i8], k: usize, row: usize) -> i32 {
    let kw = k / 16;
    let mut lane_acc = [0i32; 32];
    let mut base = 0;
    while base < kw {
        let wc = CHUNK_WORDS.min(kw - base);
        let l = wc / 4;
        for lane in 0..32 {
            if lane >= l {
                continue;
            }
            let off = row * kw + base + 4 * lane;
            let wv = [
                w.permuted[off],
                w.permuted[off + 1],
                w.permuted[off + 2],
                w.permuted[off + 3],
            ];
            let mut codes = [0i32; 16];
            for q in 0..4 {
                for j in 0..4 {
                    codes[4 * q + j] = ((wv[q] >> (2 * j)) & 0x0303_0303) as i32;
                }
            }
            for q in 0..4 {
                let blk = base + q * l + lane; // int4 index into xq
                let xi: Vec<i32> = (0..4)
                    .map(|t| {
                        i32::from_le_bytes(std::array::from_fn(|u| xq[16 * blk + 4 * t + u] as u8))
                    })
                    .collect();
                let mut a = lane_acc[lane];
                for t in 0..4 {
                    a = dp4a(codes[4 * q + t], xi[t], a);
                }
                lane_acc[lane] = a;
            }
        }
        base += CHUNK_WORDS;
    }
    lane_acc.iter().sum()
}

#[test]
fn i8_kernel_index_math() {
    let mut rng = StdRng::seed_from_u64(7);
    for &(n, k) in &[
        (3usize, 64usize),
        (3, 2048),
        (3, 2560),
        (2, 6912),
        (2, 11008),
        (2, 4096),
    ] {
        let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let p = PackedTernary::from_ternary(&wt, n, k, vec![1.0; n]).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect();
        let a = quantize_acts(&x, 1, k, None, 1e-5);
        let reference = gemm_i8(&p, &a, None);
        for r in 0..n {
            let acc = emulate_i8_row(&p, &a.q, k, r) - a.sum[0];
            let direct: i32 = (0..k).map(|c| wt[r * k + c] as i32 * a.q[c] as i32).sum();
            assert_eq!(acc, direct, "n={n} k={k} row={r}");
            let y = f16::from_f32(acc as f32 / a.s[0] * 1.0);
            assert_eq!(y.to_bits(), reference[r].to_bits());
        }
    }
}

#[test]
fn f16_decode_magic() {
    // every possible 4-code byte group decodes to w = c - 1
    for c0 in 0..3u32 {
        for c1 in 0..3u32 {
            for c2 in 0..3u32 {
                for c3 in 0..3u32 {
                    let c = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
                    for (i, ci) in [c0, c1, c2, c3].iter().enumerate() {
                        let bits = byte_perm(c, 0x4B00_0000, 0x7650 | i as u32);
                        let w = f32::from_bits(bits) - 8_388_609.0;
                        assert_eq!(w, *ci as f32 - 1.0);
                    }
                }
            }
        }
    }
}

#[test]
fn f16_kernel_index_math() {
    let mut rng = StdRng::seed_from_u64(8);
    let (n, k) = (2usize, 2560usize);
    let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
    let p = PackedTernary::from_ternary(&wt, n, k, vec![1.0; n]).unwrap();
    let x: Vec<f16> = (0..k)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
        .collect();
    let kw = k / 16;
    for row in 0..n {
        let mut acc = 0f64;
        let mut base = 0;
        while base < kw {
            let wc = CHUNK_WORDS.min(kw - base);
            let l = wc / 4;
            for lane in 0..l {
                let off = row * kw + base + 4 * lane;
                for q in 0..4 {
                    let word = p.permuted[off + q];
                    let kb = base + q * l + lane;
                    let mut wf = [0f32; 16];
                    for j in 0..4 {
                        let c = (word >> (2 * j)) & 0x0303_0303;
                        for i in 0..4 {
                            wf[4 * j + i] =
                                f32::from_bits(byte_perm(c, 0x4B00_0000, 0x7650 | i as u32))
                                    - 8_388_609.0;
                        }
                    }
                    for h in 0..16 {
                        acc += wf[h] as f64 * x[16 * kb + h].to_f32() as f64;
                    }
                }
            }
            base += CHUNK_WORDS;
        }
        let direct: f64 = (0..k)
            .map(|c| wt[row * k + c] as f64 * x[c].to_f32() as f64)
            .sum();
        assert!((acc - direct).abs() < 1e-9);
    }
}
