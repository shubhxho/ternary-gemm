//! A market-feature -> signal inference loop with a small ternary MLP.
//!
//! EVERYTHING HERE IS SYNTHETIC: the "features" are a random walk turned into returns, a spread
//! proxy and an order-book-imbalance proxy, and the model is random and untrained. The point is
//! the inference loop (preallocated, allocation-free, one request at a time) and its latency, not
//! the signal.
//!
//! cargo run --release --example signal_model [-- --steps 20000]

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use ternary_gemm::gpu::Gpu;
use ternary_gemm::mlp::{Act, CpuEngine, TernaryMlp};
use ternary_gemm::signal::SignalEngine;

const FEATURES: usize = 64; // 16 lags x {return, spread, imbalance, volume} (synthetic)

fn main() {
    let steps: usize = std::env::args()
        .skip_while(|a| a != "--steps")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000);
    let model = TernaryMlp::random(&[FEATURES, 256, 256, 1], Act::Relu, 2026).expect("model");
    println!(
        "model {FEATURES}-256-256-1 ternary, {:.1} KiB of weights (random, untrained)",
        model.weight_bytes() as f64 / 1024.0
    );

    let mut cpu = CpuEngine::new(&model).expect("cpu engine");
    let gpu = Gpu::new(0)
        .map_err(|e| println!("no GPU ({e}): CPU engine only"))
        .ok();
    let mut gpu_eng = gpu.as_ref().and_then(|g| {
        SignalEngine::new(g, &model, 1, None)
            .map_err(|e| println!("GPU engine unavailable: {e}"))
            .ok()
    });

    // Synthetic tick stream: price random walk + noisy spread / imbalance / volume.
    let mut rng = StdRng::seed_from_u64(1);
    let mut price = 100.0f32;
    let mut hist = vec![[0f32; 4]; 16];
    let mut feats = vec![0f32; FEATURES];
    let (mut sig_cpu, mut sig_gpu) = ([0f32; 1], [0f32; 1]);
    let (mut lat_cpu, mut lat_gpu) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
    let mut mismatches = 0usize;
    let mut position = 0i32;
    for _ in 0..steps {
        let ret = rng.random_range(-1.0f32..1.0) * 1e-3;
        price *= 1.0 + ret;
        hist.rotate_right(1);
        hist[0] = [
            ret * 1e3,
            rng.random_range(0.5f32..2.0),
            rng.random_range(-1.0f32..1.0),
            rng.random_range(0.0f32..3.0),
        ];
        for (i, h) in hist.iter().enumerate() {
            feats[4 * i..4 * i + 4].copy_from_slice(h);
        }

        let t = Instant::now();
        cpu.forward_into(&feats, &mut sig_cpu);
        lat_cpu.push(t.elapsed().as_nanos() as f64 / 1e3);

        if let Some(e) = gpu_eng.as_mut() {
            let t = Instant::now();
            e.infer(&feats, &mut sig_gpu).expect("gpu infer");
            lat_gpu.push(t.elapsed().as_nanos() as f64 / 1e3);
            mismatches += (sig_gpu[0].to_bits() != sig_cpu[0].to_bits()) as usize;
        }
        // Toy decision rule, only to show where a signal would be consumed.
        position = if sig_cpu[0] > 0.05 {
            1
        } else if sig_cpu[0] < -0.05 {
            -1
        } else {
            position
        };
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
    };
    println!("{steps} synthetic ticks, final price {price:.2}, last signal {:.4}, toy position {position}", sig_cpu[0]);
    println!(
        "CPU engine latency: p50 {:.2} us, p99 {:.2} us, max {:.2} us (measured on this machine)",
        pct(&mut lat_cpu, 0.5),
        pct(&mut lat_cpu, 0.99),
        pct(&mut lat_cpu, 1.0)
    );
    if !lat_gpu.is_empty() {
        println!(
            "GPU engine latency (host -> signal -> host): p50 {:.2} us, p99 {:.2} us, max {:.2} us; CPU/GPU bit mismatches: {mismatches}",
            pct(&mut lat_gpu, 0.5),
            pct(&mut lat_gpu, 0.99),
            pct(&mut lat_gpu, 1.0)
        );
    }
}
