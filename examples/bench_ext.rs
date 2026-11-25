//! Time the extra batch-1 kernels (cp.async, split-K, LUT, warp-specialized, hand-written PTX)
//! against the default dp4a kernel on a real GPU, after checking each one bit for bit against
//! the CPU reference. Prints a Markdown table of measured microseconds and effective GB/s.
//! Needs a CUDA device (sm_80+ for the ext kernels); `scripts/bench.sh --ext` runs it.
//!
//! Usage: `cargo run --release --example bench_ext [-- --iters N]`

use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use ternary_gemm::cuda_ext::reference_gemv;
use ternary_gemm::gpu::{ExtKernel, Gpu, Launch};
use ternary_gemm::mlp::Act;
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference::quantize_acts;

const SHAPES: &[(usize, usize)] = &[
    (2560, 2560),
    (6912, 2560),
    (2560, 6912),
    (4096, 11008),
    (1024, 16384),
];

fn main() {
    let iters: usize = std::env::args()
        .skip_while(|a| a != "--iters")
        .nth(1)
        .map_or(500, |v| v.parse().expect("--iters N"));
    let g = Gpu::new(0).expect("no usable GPU: this benchmark measures real hardware only");
    println!(
        "GPU: {} (sm_{}{}), {iters} launches per cell, wall time with one sync at the end\n",
        g.info.name, g.info.compute_capability.0, g.info.compute_capability.1
    );
    println!("| N x K | kernel | us / GEMV | effective GB/s (2-bit weights) |\n|---|---|---|---|");
    let mut rng = StdRng::seed_from_u64(5);
    for &(n, k) in SHAPES {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.1)).collect();
        let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-2.0..2.0)))
            .collect();
        let a = quantize_acts(&x, 1, k, None, 1e-5);
        let want: Vec<u16> = reference_gemv(&p, &a, None, Act::Identity)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let gw = g.upload_weights(&p).unwrap();
        let mut ga = g.alloc_acts(1, k).unwrap();
        g.quantize(&g.upload_f16(&x).unwrap(), None, 1e-5, &mut ga)
            .unwrap();
        let mut out = g.alloc_f16(n).unwrap();
        let mut part = g.stream.alloc_zeros::<i32>(n).unwrap();
        let bytes = (n * k / 4) as f64;
        // None = default dp4a kernel, Some(None) = hand PTX, Some(Some(kind)) = ext kernel.
        let mut cells: Vec<Option<Option<ExtKernel>>> = vec![None, Some(None)];
        if g.info.compute_capability >= (8, 0) {
            for kind in [
                ExtKernel::CpAsync { rows_per_warp: 1 },
                ExtKernel::CpAsync { rows_per_warp: 2 },
                ExtKernel::CpAsync { rows_per_warp: 4 },
                ExtKernel::SplitK { splits: 0 },
                ExtKernel::Lut,
                ExtKernel::WarpSpecialized,
            ] {
                cells.push(Some(Some(kind)));
            }
        }
        let mut run = |cell: Option<Option<ExtKernel>>, o: &mut _| match cell {
            None => g.gemm_i8(&gw, &ga, None, o, Launch::default()).unwrap(),
            Some(None) => g.gemv_hand_ptx(&gw, &ga, o).unwrap(),
            Some(Some(kind)) => g
                .gemv_ext(kind, &gw, &ga, None, o, Act::Identity, Some(&mut part))
                .unwrap(),
        };
        for cell in cells {
            let name = match cell {
                None => "dp4a default (ternary_i8)".to_string(),
                Some(None) => "hand-written PTX".to_string(),
                Some(Some(kind)) => format!("{kind:?}"),
            };
            let mut f = |o: &mut _| run(cell, o);
            f(&mut out);
            let got: Vec<u16> = g
                .download(&out)
                .unwrap()
                .iter()
                .map(|v: &f16| v.to_bits())
                .collect();
            assert_eq!(got, want, "{name} {n}x{k}: not bit-exact");
            for _ in 0..20 {
                f(&mut out);
            }
            g.synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..iters {
                f(&mut out);
            }
            g.synchronize().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("| {n} x {k} | {name} | {us:.2} | {:.0} |", bytes / us / 1e3);
        }
    }
}
