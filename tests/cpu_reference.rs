//! CPU-only tests: packing, quantization and the reference matmuls agree with a direct
//! unpacked computation. Run anywhere (no GPU needed).

use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use ternary_gemm::pack::{quantize_absmean, PackedTernary};
use ternary_gemm::reference::{gemm_f16, gemm_i8, quantize_acts};

fn rand_ternary(rng: &mut StdRng, n: usize, k: usize) -> Vec<i8> {
    (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect()
}
fn rand_f16(rng: &mut StdRng, len: usize, amp: f32) -> Vec<f16> {
    (0..len)
        .map(|_| f16::from_f32(rng.random_range(-amp..amp)))
        .collect()
}

#[test]
fn i8_reference_matches_direct_dot() {
    let mut rng = StdRng::seed_from_u64(1);
    for &(n, k, batch) in &[
        (33usize, 64usize, 1usize),
        (17, 2560, 3),
        (9, 6912, 16),
        (5, 4096 + 64, 2),
    ] {
        let wt = rand_ternary(&mut rng, n, k);
        let scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.5f32..2.0)).collect();
        let p = PackedTernary::from_ternary(&wt, n, k, scales.clone()).unwrap();
        let x = rand_f16(&mut rng, batch * k, 3.0);
        let a = quantize_acts(&x, batch, k, None, 1e-5);
        let y = gemm_i8(&p, &a, None);
        for b in 0..batch {
            for r in 0..n {
                let acc: i32 = (0..k)
                    .map(|c| wt[r * k + c] as i32 * a.q[b * k + c] as i32)
                    .sum();
                let expect = f16::from_f32(acc as f32 / a.s[b] * scales[r]);
                assert_eq!(y[b * n + r].to_bits(), expect.to_bits());
            }
            // the offset trick: sum((w+1) * x) - sum(x) == sum(w * x)
            let s: i32 = a.q[b * k..(b + 1) * k].iter().map(|&v| v as i32).sum();
            assert_eq!(s, a.sum[b]);
        }
    }
}

#[test]
fn activation_quantization_matches_bitnet() {
    let mut rng = StdRng::seed_from_u64(2);
    let k = 2560;
    let x = rand_f16(&mut rng, k, 4.0);
    let a = quantize_acts(&x, 1, k, None, 1e-5);
    let absmax = x.iter().map(|v| v.to_f32().abs()).fold(0f32, f32::max);
    assert_eq!(a.s[0], 127.0 / absmax);
    // the element with max |x| quantizes to +-127
    assert!(a.q.iter().any(|&q| q == 127 || q == -127));
    // dequantized error <= half a step
    for (q, v) in a.q.iter().zip(&x) {
        assert!((*q as f32 / a.s[0] - v.to_f32()).abs() <= 0.5 / a.s[0] + 1e-6);
    }
}

#[test]
fn rmsnorm_path_is_close_to_unfused() {
    let mut rng = StdRng::seed_from_u64(3);
    let k = 2560;
    let x = rand_f16(&mut rng, k, 2.0);
    let g = rand_f16(&mut rng, k, 1.5);
    let fused = quantize_acts(&x, 1, k, Some(&g), 1e-5);
    let ms: f64 = x.iter().map(|v| (v.to_f32() as f64).powi(2)).sum::<f64>() / k as f64;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let normed: Vec<f16> = x
        .iter()
        .zip(&g)
        .map(|(v, g)| f16::from_f64(v.to_f32() as f64 * g.to_f32() as f64 * inv))
        .collect();
    let unfused = quantize_acts(&normed, 1, k, None, 1e-5);
    let diff = fused
        .q
        .iter()
        .zip(&unfused.q)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1)
        .count();
    assert_eq!(diff, 0);
}

#[test]
fn f16_reference_and_absmean() {
    let mut rng = StdRng::seed_from_u64(4);
    let (n, k) = (16, 2048);
    let wf: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let (wt, scales) = quantize_absmean(&wf, n, k, false);
    assert!(wt.iter().all(|v| (-1..=1).contains(v)));
    assert!(scales.windows(2).all(|w| w[0] == w[1]));
    let (_, row_scales) = quantize_absmean(&wf, n, k, true);
    assert_eq!(row_scales.len(), n);
    let p = PackedTernary::from_ternary(&wt, n, k, scales.clone()).unwrap();
    let x = rand_f16(&mut rng, 2 * k, 1.0);
    let y = gemm_f16(&p, &x, 2, None);
    for b in 0..2 {
        for r in 0..n {
            let acc: f64 = (0..k)
                .map(|c| wt[r * k + c] as f64 * x[b * k + c].to_f32() as f64)
                .sum();
            assert!((y[b * n + r] as f64 - acc * scales[r] as f64).abs() < 1e-3);
        }
    }
}
