//! The CPU GEMM backend must be bit-identical to the reference for any batch and thread count.

use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ternary_gemm::cpu::CpuTernary;
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference::{gemm_i8, quantize_acts};

fn bits(v: &[f16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cpu_gemm_bit_exact_any_batch_and_threads() {
    let mut rng = StdRng::seed_from_u64(5);
    for &(n, k) in &[
        (1usize, 64usize),
        (7, 128),
        (64, 2048),
        (33, 2112),
        (130, 4096),
    ] {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.1)).collect();
        let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
        for batch in [1usize, 3, 16, 17, 40] {
            let x: Vec<f16> = (0..batch * k)
                .map(|_| f16::from_f32(rng.random_range(-2.0..2.0)))
                .collect();
            let res: Vec<f16> = (0..batch * n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
                .collect();
            let a = quantize_acts(&x, batch, k, None, 1e-5);
            for residual in [None, Some(&res[..])] {
                let want = gemm_i8(&p, &a, residual);
                for threads in [1usize, 2, 5] {
                    let got = CpuTernary::new(&p)
                        .with_threads(threads)
                        .gemm_i8(&a, residual);
                    assert_eq!(bits(&want), bits(&got), "{n}x{k} b{batch} t{threads}");
                    let got = CpuTernary::new(&p)
                        .without_packed()
                        .with_threads(threads)
                        .gemm_i8(&a, residual);
                    assert_eq!(bits(&want), bits(&got), "bytes {n}x{k} b{batch} t{threads}");
                }
            }
        }
    }
}

#[test]
fn fast_quantize_matches_reference() {
    use half::f16;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(77);
    for &(batch, k) in &[(1usize, 64usize), (3, 100), (16, 2560), (5, 1024), (2, 7)] {
        for scale in [1e-6f32, 1.0, 300.0] {
            let x: Vec<f16> = (0..batch * k)
                .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0) * scale))
                .collect();
            let g: Vec<f16> = (0..k)
                .map(|_| f16::from_f32(rng.random_range(0.1f32..2.0)))
                .collect();
            for gamma in [None, Some(&g[..])] {
                let want = ternary_gemm::reference::quantize_acts(&x, batch, k, gamma, 1e-5);
                let got = ternary_gemm::cpu::quantize_acts(&x, batch, k, gamma, 1e-5);
                assert_eq!(
                    got,
                    want,
                    "batch={batch} k={k} scale={scale} gamma={}",
                    gamma.is_some()
                );
            }
        }
    }
}

#[test]
fn zero_padding_k_is_exact() {
    use half::f16;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use ternary_gemm::pack::{pad_k, pad_k_acts};
    let mut rng = StdRng::seed_from_u64(9);
    for &(n, k, batch) in &[(5usize, 100usize, 3usize), (3, 1, 1), (7, 130, 17)] {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let x: Vec<f16> = (0..batch * k)
            .map(|_| f16::from_f32(rng.random_range(-2.0f32..2.0)))
            .collect();
        // Unpadded oracle: plain integer math with the reference epilogue.
        let q = ternary_gemm::reference::quantize_acts(&x, batch, k, None, 1e-5);
        let scale: Vec<f32> = (0..n).map(|r| 0.01 + r as f32 * 0.002).collect();
        let (wp, kp) = pad_k(&w, n, k);
        assert_eq!(kp % 64, 0);
        let p = ternary_gemm::PackedTernary::from_ternary(&wp, n, kp, scale.clone()).unwrap();
        let xp = pad_k_acts(&x, batch, k, kp);
        let qp = ternary_gemm::reference::quantize_acts(&xp, batch, kp, None, 1e-5);
        assert_eq!(qp.s, q.s);
        assert_eq!(qp.sum, q.sum);
        let y = ternary_gemm::cpu::CpuTernary::new(&p).gemm_i8(&qp, None);
        for b in 0..batch {
            for r in 0..n {
                let acc: i32 = (0..k)
                    .map(|c| w[r * k + c] as i32 * q.q[b * k + c] as i32)
                    .sum();
                let want = f16::from_f32(acc as f32 / q.s[b] * scale[r]);
                assert_eq!(
                    y[b * n + r].to_bits(),
                    want.to_bits(),
                    "n={n} k={k} b={b} r={r}"
                );
            }
        }
    }
}
