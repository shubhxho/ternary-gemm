//! Looped ternary transformer on the CPU: effective depth vs unique weight bytes, latency per loop,
//! and adaptive halting. CPU-only measurement; run with
//! `cargo run --release --example looped -- [d_model] [tokens] [loops]`.

use half::f16;
use std::time::Instant;
use ternary_gemm::looped::*;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric args: d_model tokens loops"))
        .collect();
    let d = *args.first().unwrap_or(&512);
    let tokens = *args.get(1).unwrap_or(&16);
    let loops = *args.get(2).unwrap_or(&8);
    let cfg = BlockConfig {
        d_model: d,
        n_heads: (d / 64).max(1),
        d_ff: 4 * d,
    };
    let x: Vec<f16> = (0..tokens * d)
        .map(|i| f16::from_f32(((i * 7919) % 2001) as f32 / 1000.0 - 1.0))
        .collect();
    println!(
        "block d_model={d} heads={} d_ff={} tokens={tokens}",
        cfg.n_heads, cfg.d_ff
    );
    println!("(CPU only; numbers are from this machine, not a GPU)\n");

    for inject in [Inject::None, Inject::Add, Inject::Adapter] {
        let m = LoopedTransformer::random(
            cfg,
            LoopConfig {
                loops,
                inject,
                halting: None,
            },
            1,
        )
        .unwrap();
        let mut be = CpuBackend::new(&m.block, m.adapter.as_ref());
        let _ = m.forward(&mut be, &x, tokens).unwrap(); // warm-up
        let reps = 5;
        let t = Instant::now();
        let mut out = None;
        for _ in 0..reps {
            out = Some(m.forward(&mut be, &x, tokens).unwrap());
        }
        let per = t.elapsed().as_secs_f64() / reps as f64;
        let out = out.unwrap();
        println!(
            "{inject:?}: {loops} loops in {:.2} ms ({:.3} ms/loop); unique weights {:.1} KiB vs {:.1} KiB unrolled",
            per * 1e3,
            per * 1e3 / loops as f64,
            m.unique_weight_bytes() as f64 / 1024.0,
            m.unrolled_weight_bytes() as f64 / 1024.0
        );
        let ds: Vec<String> = out.deltas.iter().map(|d| format!("{d:.3}")).collect();
        println!("  relative state change per loop: [{}]", ds.join(", "));
    }

    let halt = LoopConfig {
        loops: 4 * loops,
        inject: Inject::Add,
        halting: Some(Halting {
            tol: 0.05,
            min_loops: 2,
        }),
    };
    let m = LoopedTransformer::random(cfg, halt, 1).unwrap();
    let out = m.forward_cpu(&x, tokens);
    println!(
        "\nadaptive halting (Add, tol 0.05, max {}): ran {} loops, final delta {:.4}",
        halt.loops,
        out.loops_run,
        out.deltas.last().unwrap()
    );
}
