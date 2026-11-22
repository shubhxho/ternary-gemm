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
                }
            }
        }
    }
}
