//! Looped transformer on the GPU kernels vs the CPU backend. Skipped without a CUDA device unless
//! `TERNARY_REQUIRE_GPU=1`.

use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use ternary_gemm::gpu::Gpu;
use ternary_gemm::looped::*;

fn gpu() -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) => {
            if std::env::var("TERNARY_REQUIRE_GPU").as_deref() == Ok("1") {
                panic!("TERNARY_REQUIRE_GPU=1 but no usable GPU: {e}");
            }
            eprintln!("SKIPPED (no GPU): {e}");
            None
        }
    }
}

const CFG: BlockConfig = BlockConfig {
    d_model: 256,
    n_heads: 4,
    d_ff: 512,
};

fn input(tokens: usize, seed: u64) -> Vec<f16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..tokens * CFG.d_model)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
        .collect()
}

#[test]
fn gemm_without_norm_is_bit_exact() {
    let Some(g) = gpu() else { return };
    let m = LoopedTransformer::random(
        CFG,
        LoopConfig {
            loops: 1,
            inject: Inject::None,
            halting: None,
        },
        3,
    )
    .unwrap();
    let mut cpu = CpuBackend::new(&m.block, None);
    let mut dev = GpuBackend::new(&g, &m.block, None).unwrap();
    // 37 tokens: exercises 16 + 16 + 5 chunking.
    let x = input(37, 4);
    let res = input(37, 5);
    let a = cpu.quantize(&x, 37, CFG.d_model, None).unwrap();
    let want = cpu.gemm(WeightId::O, &a, Some(&res)).unwrap();
    let a = dev.quantize(&x, 37, CFG.d_model, None).unwrap();
    let got = dev.gemm(WeightId::O, &a, Some(&res)).unwrap();
    assert_eq!(got, want);
}

#[test]
fn looped_forward_matches_cpu_within_tolerance() {
    let Some(g) = gpu() else { return };
    for inject in [Inject::None, Inject::Add, Inject::Adapter] {
        let m = LoopedTransformer::random(
            CFG,
            LoopConfig {
                loops: 3,
                inject,
                halting: None,
            },
            9,
        )
        .unwrap();
        let x = input(20, 10);
        let want = m.forward_cpu(&x, 20);
        let mut dev = GpuBackend::new(&g, &m.block, m.adapter.as_ref()).unwrap();
        let got = m.forward(&mut dev, &x, 20).unwrap();
        assert_eq!(got.loops_run, want.loops_run);
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in got.hidden.iter().zip(&want.hidden) {
            let (a, b) = (a.to_f32() as f64, b.to_f32() as f64);
            num += (a - b) * (a - b);
            den += b * b;
        }
        let rel = (num / den.max(1e-30)).sqrt();
        // RMSNorm uses rsqrtf on the device: a last-bit difference in the scale can flip an int8
        // code, and three loops amplify it a little. 2% relative L2 is far above that noise.
        assert!(rel < 2e-2, "{inject:?}: rel L2 {rel}");
    }
}
