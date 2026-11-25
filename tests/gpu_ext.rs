//! The extra batch-1 kernels (cp.async, split-K, LUT, warp-specialized) on a real GPU, checked
//! bit for bit against `cuda_ext::reference_gemv`. Skipped without a CUDA device (or on pre-sm_80
//! parts) unless `TERNARY_REQUIRE_GPU=1`.

use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use ternary_gemm::cuda_ext::reference_gemv;
use ternary_gemm::gpu::{ExtKernel, Gpu};
use ternary_gemm::mlp::Act;
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference::quantize_acts;

fn gpu() -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) if g.info.compute_capability >= (8, 0) => Some(g),
        Ok(g) => {
            eprintln!(
                "SKIPPED (needs sm_80+, found {:?})",
                g.info.compute_capability
            );
            None
        }
        Err(e) => {
            if std::env::var("TERNARY_REQUIRE_GPU").as_deref() == Ok("1") {
                panic!("TERNARY_REQUIRE_GPU=1 but no usable GPU: {e}");
            }
            eprintln!("SKIPPED (no GPU): {e}");
            None
        }
    }
}

#[test]
fn ext_kernels_bit_exact_vs_reference() {
    let Some(g) = gpu() else { return };
    let mut rng = StdRng::seed_from_u64(77);
    let kinds = [
        ExtKernel::CpAsync { rows_per_warp: 1 },
        ExtKernel::CpAsync { rows_per_warp: 2 },
        ExtKernel::CpAsync { rows_per_warp: 4 },
        ExtKernel::SplitK { splits: 0 },
        ExtKernel::SplitK { splits: 3 },
        ExtKernel::Lut,
        ExtKernel::WarpSpecialized,
    ];
    for &(n, k) in &[
        (1usize, 64usize),
        (37, 2112),
        (2560, 2560),
        (640, 6912),
        (4096, 8192),
    ] {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.1)).collect();
        let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-2.0..2.0)))
            .collect();
        let res: Vec<f16> = (0..n)
            .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let a = quantize_acts(&x, 1, k, None, 1e-5);
        let gw = g.upload_weights(&p).unwrap();
        let gx = g.upload_f16(&x).unwrap();
        let gres = g.upload_f16(&res).unwrap();
        let mut ga = g.alloc_acts(1, k).unwrap();
        g.quantize(&gx, None, 1e-5, &mut ga).unwrap();
        let mut out = g.alloc_f16(n).unwrap();
        let mut part = g.stream.alloc_zeros::<i32>(n).unwrap();
        for act in [Act::Identity, Act::Relu2] {
            for residual in [None, Some(&gres)] {
                let want = reference_gemv(&p, &a, residual.map(|_| &res[..]), act);
                for kind in kinds {
                    g.gemv_ext(kind, &gw, &ga, residual, &mut out, act, Some(&mut part))
                        .unwrap();
                    let got: Vec<f16> = g.download(&out).unwrap();
                    assert_eq!(
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{kind:?} {n}x{k} {act:?} residual={}",
                        residual.is_some()
                    );
                }
            }
        }
    }
}

#[test]
fn hand_written_ptx_bit_exact_vs_reference() {
    let Some(g) = gpu() else { return };
    let mut rng = StdRng::seed_from_u64(78);
    for &(n, k) in &[(1usize, 64usize), (37, 2112), (2560, 2560), (4096, 11008)] {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.1)).collect();
        let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-2.0..2.0)))
            .collect();
        let a = quantize_acts(&x, 1, k, None, 1e-5);
        let gw = g.upload_weights(&p).unwrap();
        let mut ga = g.alloc_acts(1, k).unwrap();
        g.quantize(&g.upload_f16(&x).unwrap(), None, 1e-5, &mut ga)
            .unwrap();
        let mut out = g.alloc_f16(n).unwrap();
        g.gemv_hand_ptx(&gw, &ga, &mut out).unwrap();
        let got: Vec<f16> = g.download(&out).unwrap();
        let want = reference_gemv(&p, &a, None, Act::Identity);
        assert_eq!(
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "hand PTX {n}x{k}"
        );
    }
}
