//! CPU-side correctness of the tensor-core path: unpack 2-bit natural-layout weights
//! into m16n8k16 fragments and accumulate a tiled GEMM. Runs without a GPU. The GPU
//! test in `tests/gpu.rs` is skipped when no CUDA device is present.

use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference::{self, quantize_acts};
use ternary_gemm::tensor_core;

fn bits_eq(a: &[f16], b: &[f16]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn tensor_core_i8_is_bit_exact_with_the_cpu_reference() {
    let mut rng = StdRng::seed_from_u64(21);
    // Include N and batch that are not multiples of the 16×8 MMA tile.
    for &(n, k) in &[
        (1usize, 64usize),
        (16, 64),
        (17, 64),
        (37, 128),
        (3, 2048),
        (2, 2560),
        (2, 4096),
    ] {
        for &batch in &[1usize, 3, 8, 16] {
            let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
            let scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.05)).collect();
            let p = PackedTernary::from_ternary(&wt, n, k, scales).unwrap();
            let x: Vec<f16> = (0..batch * k)
                .map(|_| f16::from_f32(rng.random_range(-2.0f32..2.0)))
                .collect();
            let res: Vec<f16> = (0..batch * n)
                .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
                .collect();
            let a = quantize_acts(&x, batch, k, None, 1e-5);
            let expect = reference::gemm_i8(&p, &a, Some(&res));
            let got = tensor_core::gemm_i8(&p, &a, Some(&res));
            assert_eq!(
                bits_eq(&got, &expect),
                0,
                "tc i8 vs reference {n}x{k} b{batch}"
            );
        }
    }
}

#[test]
fn tensor_core_f16_matches_the_high_precision_reference() {
    let mut rng = StdRng::seed_from_u64(22);
    let (n, k, batch) = (17usize, 256usize, 3usize);
    let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
    let p = PackedTernary::from_ternary(&wt, n, k, vec![0.03; n]).unwrap();
    let x: Vec<f16> = (0..batch * k)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
        .collect();
    let expect = reference::gemm_f16(&p, &x, batch, None);
    let got = tensor_core::gemm_f16(&p, &x, batch, None);
    let max_ref = expect.iter().copied().map(f32::abs).fold(0f32, f32::max);
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Weights are exact in f16; residual error is f32 vs f64 accumulation.
    assert!(
        max_err <= 1e-4 * max_ref + 1e-6,
        "tc f16 err {max_err} vs max {max_ref}"
    );
}

#[test]
fn tensor_core_source_has_no_includes_and_uses_mma_sync() {
    let src = include_str!("../kernels/tensor_core.cu");
    assert!(
        !src.lines().any(|l| l.trim_start().starts_with("#include")),
        "tensor_core.cu must stay self-contained for NVRTC"
    );
    assert!(src.contains("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"));
    assert!(src.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
    assert!(src.contains("extern \"C\" __global__ void"));
    assert!(src.contains("ternary_tc_i8"));
    assert!(src.contains("ternary_tc_f16"));
    // Must mention the theoretical (not measured) 8x ceiling so the source stays honest.
    assert!(src.to_lowercase().contains("theoretical bandwidth ceiling"));
}
