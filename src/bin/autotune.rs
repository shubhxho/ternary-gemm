//! Autotune the W2A8 kernels on this GPU for BitNet-b1.58-2B decode shapes and record the results.
//!
//! Usage: `cargo run --release --bin autotune -- [--out results/<stem>-autotune] [--batches 1,4,16]
//!        [--top 8] [--cache tune-cache.csv]`
//!
//! Writes `<out>.csv` (machine-readable, also usable as a cache) and `<out>.md` (report with the
//! model's projection next to each measurement). Needs a CUDA GPU; refuses to run without one.

use ternary_gemm::autotune::{kernel_name, Autotuner};
use ternary_gemm::gpu::Gpu;
use ternary_gemm::tuning::{model_config, project_gemv};

/// (N, K): BitNet-b1.58-2B q/o (2560^2), k/v (640 x 2560), FFN up/gate (6912 x 2560), down.
const SHAPES: &[(usize, usize)] = &[(2560, 2560), (640, 2560), (6912, 2560), (2560, 6912)];

fn main() {
    let mut out = String::from("results/autotune");
    let mut batches = vec![1usize, 2, 4, 8, 16];
    let mut top = 8usize;
    let mut cache: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = val(),
            "--batches" => batches = val().split(',').map(|s| s.parse().unwrap()).collect(),
            "--top" => top = val().parse().unwrap(),
            "--cache" => cache = Some(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    let gpu = match Gpu::new(0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("autotune needs a CUDA GPU: {e}");
            std::process::exit(1);
        }
    };
    let mut t = Autotuner::new(&gpu);
    t.top = top;
    if let Some(c) = &cache {
        if let Ok(s) = std::fs::read_to_string(c) {
            eprintln!("loaded {} cached entries from {c}", t.load_csv(&s));
        }
    }
    let i = &gpu.info;
    let mut md = format!(
        "# Autotune: {} (sm_{}{}, {} SMs, PTX {})\n\nProfile: {} (model limits). All times are **measured** medians on this device (CUDA events, quantization included); the `projected` column is the roofline **projection** from public specs (`ternary_gemm::tuning::project_gemv`), not a measurement.\n\n| N x K | batch | best config | kernel regs | best (us) | default (us) | speedup | projected bound (us) | model occupancy | candidates |\n|---|---|---|---|---|---|---|---|---|---|\n",
        i.name, i.compute_capability.0, i.compute_capability.1, i.sm_count, i.ptx_arch, t.spec.board
    );
    for &(n, k) in SHAPES {
        for &b in &batches {
            let e = match t.get(n, k, b) {
                Some(e) => e.clone(),
                None => t.tune(n, k, b).expect("tuning failed"),
            };
            let regs = t.kernel_regs(e.best, b);
            let m = model_config(&t.spec, i.sm_count, n, k, b, e.best, regs);
            let p = project_gemv(&t.spec, n, k, b);
            md += &format!(
                "| {n} x {k} | {b} | {}w x {}r{} (`{}`) | {} | {:.2} | {:.2} | {:.2}x | {:.2} ({}) | {} | {} |\n",
                e.best.warps,
                e.best.rows_per_warp,
                if e.best.fused { " fused" } else { "" },
                kernel_name(e.best, b),
                regs.map_or("?".into(), |r| r.to_string()),
                e.best_us,
                e.default_us,
                e.default_us / e.best_us,
                p.t_bound_us,
                p.bound,
                m.map_or("?".into(), |m| format!(
                    "{} warps/SM ({})",
                    m.occupancy.warps_per_sm, m.occupancy.limiter
                )),
                e.tried
            );
            eprintln!(
                "{n}x{k} b{b}: {:?} {:.2} us (default {:.2})",
                e.best, e.best_us, e.default_us
            );
        }
    }
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(format!("{out}.csv"), t.to_csv()).unwrap();
    std::fs::write(format!("{out}.md"), &md).unwrap();
    if let Some(c) = &cache {
        std::fs::write(c, t.to_csv()).unwrap();
    }
    println!("{md}");
}
