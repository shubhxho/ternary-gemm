//! Criterion micro-benchmarks for the CPU paths (run with `cargo bench --bench cpu`).
//! Numbers depend entirely on the machine; nothing here is a GPU measurement.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::hint::black_box;
use ternary_gemm::mlp::{quantize_into, Act, CpuEngine, DenseF32Mlp, TernaryMlp};
use ternary_gemm::pack::PackedTernary;

fn quantize(c: &mut Criterion) {
    let mut g = c.benchmark_group("quantize_into");
    let mut rng = StdRng::seed_from_u64(1);
    for k in [64usize, 256, 1024, 4096] {
        let x: Vec<f32> = (0..k).map(|_| rng.random_range(-3.0..3.0)).collect();
        let mut q = vec![0i8; k];
        g.throughput(Throughput::Elements(k as u64));
        g.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| black_box(quantize_into(black_box(&x), &mut q)))
        });
    }
    g.finish();
}

fn signal_models(c: &mut Criterion) {
    let mut g = c.benchmark_group("signal_model_forward");
    for dims in [
        &[64usize, 256, 256, 1][..],
        &[256, 1024, 1024, 16],
        &[1024, 1024, 1024, 1024, 4],
    ] {
        let name = dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("-");
        let m = TernaryMlp::random(dims, Act::Relu, 7).unwrap();
        let x: Vec<f32> = (0..m.in_dim()).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut out = vec![0f32; m.out_dim()];
        let macs: usize = m.layers.iter().map(|l| l.n * l.k).sum();
        g.throughput(Throughput::Elements(macs as u64));
        let mut e = CpuEngine::new(&m).unwrap();
        g.bench_function(
            BenchmarkId::new(format!("ternary-{}", e.simd()), &name),
            |b| b.iter(|| e.forward_into(black_box(&x), &mut out)),
        );
        let mut d = DenseF32Mlp::new(&m);
        g.bench_function(BenchmarkId::new("dense-f32", &name), |b| {
            b.iter(|| d.forward_into(black_box(&x), &mut out))
        });
    }
    g.finish();
}

fn packing(c: &mut Criterion) {
    let mut g = c.benchmark_group("pack_from_ternary");
    let mut rng = StdRng::seed_from_u64(2);
    for (n, k) in [(256usize, 256usize), (2560, 2560)] {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        g.throughput(Throughput::Elements((n * k) as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("{n}x{k}")), |b| {
            b.iter(|| PackedTernary::from_ternary(black_box(&w), n, k, vec![1.0; n]).unwrap())
        });
    }
    g.finish();
}

fn looped(c: &mut Criterion) {
    use ternary_gemm::looped::*;
    let cfg = BlockConfig {
        d_model: 256,
        n_heads: 4,
        d_ff: 1024,
    };
    let tokens = 8;
    let x: Vec<half::f16> = (0..tokens * cfg.d_model)
        .map(|i| half::f16::from_f32(((i * 31) % 200) as f32 / 100.0 - 1.0))
        .collect();
    let mut g = c.benchmark_group("looped_forward_d256_t8");
    for loops in [1usize, 4, 8] {
        let m = LoopedTransformer::random(
            cfg,
            LoopConfig {
                loops,
                inject: Inject::Add,
                halting: None,
            },
            1,
        )
        .unwrap();
        let mut be = CpuBackend::new(&m.block, None);
        g.bench_with_input(BenchmarkId::from_parameter(loops), &loops, |b, _| {
            b.iter(|| m.forward(&mut be, &x, tokens).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, quantize, signal_models, packing, looped);
criterion_main!(benches);
