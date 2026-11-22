//! Print the roofline **projections** and modelled launch configs for a GPU profile. No GPU
//! needed; nothing printed here is a measurement.
//!
//! Usage: cargo run --release --bin roofline -- [sm_120|sm_90|sm_89|sm_86|sm_80] [--batches 1,4,16]

use ternary_gemm::tuning::{profile_for, project_gemv, rank_configs, PROFILES};

/// BitNet-b1.58-2B per-layer projections (hidden 2560, kv 640, ffn 6912), 30 layers.
const LAYER: &[(&str, usize, usize)] = &[
    ("q", 2560, 2560),
    ("k", 640, 2560),
    ("v", 640, 2560),
    ("o", 2560, 2560),
    ("gate", 6912, 2560),
    ("up", 6912, 2560),
    ("down", 2560, 6912),
];
const LAYERS: usize = 30;

fn main() {
    let mut arch = "sm_120".to_string();
    let mut batches = vec![1usize, 4, 16];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--batches" => {
                batches = args
                    .next()
                    .expect("--batches needs a value")
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            s if s.starts_with("sm_") => arch = s.to_string(),
            other => panic!("unknown argument {other}"),
        }
    }
    let v: i32 = arch[3..].parse().expect("sm_XY");
    let spec = profile_for((v / 10, v % 10));
    println!(
        "# Roofline projection: {} (sm_{}{})\n\n**These are projections from public specs and stated model assumptions, not measurements.**\n",
        spec.board, spec.compute_capability.0, spec.compute_capability.1
    );
    println!(
        "{} SMs, {:.0} GB/s DRAM, {} MiB L2, {:.0} MHz boost, {} INT32 lanes/SM (model), {} KiB smem/SM, {} warps/SM max\n",
        spec.sm_count,
        spec.dram_gbps,
        spec.l2_bytes >> 20,
        spec.boost_mhz,
        spec.int32_lanes_per_sm,
        spec.smem_per_sm / 1024,
        spec.max_warps_per_sm
    );
    println!("## Per projection (W2A8), BitNet-b1.58-2B shapes\n");
    println!("| proj | N x K | batch | weight MB | t_dram (us) | t_int (us) | bound | modelled best config | occupancy |\n|---|---|---|---|---|---|---|---|---|");
    for &(name, n, k) in LAYER {
        for &b in &batches {
            let p = project_gemv(&spec, n, k, b);
            let best = rank_configs(&spec, spec.sm_count, n, k, b);
            let m = best.first().expect("no valid config");
            println!(
                "| {name} | {n} x {k} | {b} | {:.2} | {:.2} | {:.2} | {} | {}w x {}r{} | {} warps/SM ({}) |",
                (n * k) as f64 / 4.0 / 1e6,
                p.t_dram_us,
                p.t_int_us,
                p.bound,
                m.config.warps,
                m.config.rows_per_warp,
                if m.config.fused { " fused" } else { "" },
                m.occupancy.warps_per_sm,
                m.occupancy.limiter
            );
        }
    }
    println!("\n## Whole model ({LAYERS} layers, projections only)\n");
    println!("| batch | weight bytes (MB) | projected bound per token step (us) | projected tokens/s per sequence |\n|---|---|---|---|");
    for &b in &batches {
        let (mut bytes, mut t) = (0.0, 0.0);
        for &(_, n, k) in LAYER {
            let p = project_gemv(&spec, n, k, b);
            bytes += p.bytes;
            t += p.t_bound_us;
        }
        println!(
            "| {b} | {:.1} | {:.1} | {:.0} |",
            bytes * LAYERS as f64 / 1e6,
            t * LAYERS as f64,
            1e6 / (t * LAYERS as f64)
        );
    }
    let block_bytes: usize = LAYER.iter().map(|&(_, n, k)| n * k / 4).sum();
    println!(
        "\nLooped variant: one shared block = {:.1} MB of packed weights, {} the {} MiB L2, so after the first loop the weight stream can be served from L2 instead of DRAM (L2 bandwidth for this part is not published; measure it).",
        block_bytes as f64 / 1e6,
        if (block_bytes as u64) <= spec.l2_bytes { "fits in" } else { "does not fit in" },
        spec.l2_bytes >> 20
    );
    println!("\nOther profiles: {}", PROFILES.map(|p| p.board).join(", "));
}
