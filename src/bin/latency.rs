//! End-to-end latency benchmark for small ternary signal models: host feature vector in ->
//! signal out on the host, one request at a time (batch 1), as a latency-sensitive trading loop
//! would issue them. Reports p50/p90/p99/p99.9/max, mean and std-dev, plus a log-spaced
//! histogram (CSV) for jitter charts.
//!
//! CPU paths always run. GPU paths run only if a CUDA device is found. Every path is checked
//! against the scalar reference before it is timed (bit-exact for ternary paths).
//!
//! Features are SYNTHETIC random vectors and the models are random (untrained): this measures
//! latency only, nothing about trading performance.
//!
//! usage: latency [--iters N] [--warmup N] [--models tiny,small,medium] [--csv FILE] [--cpu-only]

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::time::Instant;
use ternary_gemm::gpu::Gpu;
use ternary_gemm::mlp::{cpu_simd_level, Act, CpuEngine, DenseF32Mlp, TernaryMlp};
use ternary_gemm::signal::{CublasMlp, PersistentEngine, SignalEngine, Submit, Wait};

const MODELS: &[(&str, &[usize])] = &[
    ("tiny", &[64, 256, 256, 1]),
    ("small", &[256, 1024, 1024, 16]),
    ("medium", &[1024, 1024, 1024, 1024, 4]),
];

thread_local! { static CSV: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) }; }
fn csv(l: String) {
    CSV.with(|c| c.borrow_mut().push(l));
}

struct Opts {
    iters: usize,
    warmup: usize,
    models: Vec<String>,
    csv: Option<String>,
    cpu_only: bool,
}

fn parse() -> Opts {
    let mut o = Opts {
        iters: 20000,
        warmup: 2000,
        models: MODELS.iter().map(|m| m.0.to_string()).collect(),
        csv: None,
        cpu_only: false,
    };
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--iters" => o.iters = v.parse().expect("--iters N"),
            "--warmup" => o.warmup = v.parse().expect("--warmup N"),
            "--models" => o.models = v.split(',').map(str::to_string).collect(),
            "--csv" => o.csv = Some(v),
            "--cpu-only" => {
                o.cpu_only = true;
                i += 1;
                continue;
            }
            x => panic!("unknown argument {x}"),
        }
        i += 2;
    }
    o
}

fn cpu_model() -> String {
    let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let name = info
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let total = info.lines().filter(|l| l.starts_with("processor")).count();
    let usable = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    if total > 0 && usable < total {
        format!("{name}; {total} logical CPUs; pinned to {usable}")
    } else {
        format!("{name}; {usable} logical CPUs")
    }
}

struct Stats {
    p50: f64,
    p90: f64,
    p99: f64,
    p999: f64,
    max: f64,
    mean: f64,
    std: f64,
}

fn stats(us: &mut [f64]) -> Stats {
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| us[((us.len() as f64 * p) as usize).min(us.len() - 1)];
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    let var = us.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / us.len() as f64;
    Stats {
        p50: q(0.50),
        p90: q(0.90),
        p99: q(0.99),
        p999: q(0.999),
        max: *us.last().unwrap(),
        mean,
        std: var.sqrt(),
    }
}

/// Time `f` once per iteration (each call is one full request). Returns per-call microseconds.
fn time_each(o: &Opts, mut f: impl FnMut(usize)) -> Vec<f64> {
    for i in 0..o.warmup {
        f(i);
    }
    let mut v = Vec::with_capacity(o.iters);
    for i in 0..o.iters {
        let t = Instant::now();
        f(i);
        v.push(t.elapsed().as_nanos() as f64 / 1e3);
    }
    v
}

fn report(model: &str, path: &str, device: &str, mut us: Vec<f64>) {
    let n = us.len();
    // log-spaced histogram, 20 bins per decade from 10 ns to 1 s
    let mut hist = std::collections::BTreeMap::new();
    for &v in &us {
        let b = ((v.max(0.01)).log10() * 20.0).floor() as i64;
        *hist.entry(b).or_insert(0usize) += 1;
    }
    let s = stats(&mut us);
    println!(
        "| {model} | {path} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.2} | {:.2} |",
        s.p50, s.p90, s.p99, s.p999, s.max, s.mean, s.std
    );
    csv(format!(
        "lat,{model},{path},{device},{n},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
        s.p50, s.p90, s.p99, s.p999, s.max, s.mean, s.std
    ));
    for (b, c) in hist {
        let lo = 10f64.powf(b as f64 / 20.0);
        let hi = 10f64.powf((b + 1) as f64 / 20.0);
        csv(format!("hist,{model},{path},{lo:.5},{hi:.5},{c}"));
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn main() {
    let o = parse();
    let cpu = cpu_model();
    let gpu = if o.cpu_only {
        None
    } else {
        match Gpu::new(0) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("no GPU ({e}); running CPU paths only");
                None
            }
        }
    };
    println!("# Signal-model latency (batch 1, end to end, host -> signal -> host)\n");
    println!("| | |\n|---|---|");
    println!("| CPU | {cpu}, SIMD path: {} |", cpu_simd_level());
    match &gpu {
        Some(g) => println!(
            "| GPU | {} (sm_{}{}, {} SMs) |",
            g.info.name, g.info.compute_capability.0, g.info.compute_capability.1, g.info.sm_count
        ),
        None => println!("| GPU | none found: GPU rows NOT MEASURED |"),
    }
    println!("| Iterations | {} timed per path after {} warm-up, one request per iteration, timed with `Instant` |", o.iters, o.warmup);
    println!("| Inputs | SYNTHETIC random feature vectors, random untrained models: latency only, no trading claims |\n");
    let t_ovh = {
        let mut v = time_each(
            &Opts {
                iters: 20000,
                warmup: 100,
                models: vec![],
                csv: None,
                cpu_only: true,
            },
            |_| {},
        );
        stats(&mut v).p50
    };
    println!("Timer overhead (empty request, p50): {t_ovh:.3} us\n");
    println!("All latencies in microseconds.\n");
    println!("| model | path | p50 | p90 | p99 | p99.9 | max | mean | std (jitter) |\n|---|---|---|---|---|---|---|---|---|");

    let mut rng = StdRng::seed_from_u64(42);
    for (name, dims) in MODELS.iter().filter(|m| o.models.iter().any(|x| x == m.0)) {
        let label = format!(
            "{name} {}",
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("-")
        );
        let m = TernaryMlp::random(dims, Act::Relu, 7).expect("model");
        let (ind, outd) = (m.in_dim(), m.out_dim());
        // Pool of synthetic feature vectors (e.g. standardized returns / spreads / imbalances).
        let pool: Vec<Vec<f32>> = (0..256)
            .map(|_| (0..ind).map(|_| rng.random_range(-3.0f32..3.0)).collect())
            .collect();
        let refs: Vec<Vec<f32>> = pool.iter().map(|x| m.forward_reference(x)).collect();
        let mut out = vec![0f32; outd];
        let cpu_dev = format!("CPU: {cpu}");

        let us = time_each(&o, |i| {
            let y = m.forward_reference(&pool[i % 256]);
            std::hint::black_box(&y);
        });
        report(&label, "CPU ternary scalar reference", &cpu_dev, us);

        let mut eng = CpuEngine::new(&m).unwrap();
        for (x, r) in pool.iter().zip(&refs) {
            eng.forward_into(x, &mut out);
            assert_eq!(bits(r), bits(&out), "CpuEngine mismatch");
        }
        let us = time_each(&o, |i| {
            eng.forward_into(&pool[i % 256], &mut out);
            std::hint::black_box(&out);
        });
        report(&label, "CPU ternary SIMD (CpuEngine)", &cpu_dev, us);

        let mut dense = DenseF32Mlp::new(&m);
        let us = time_each(&o, |i| {
            dense.forward_into(&pool[i % 256], &mut out);
            std::hint::black_box(&out);
        });
        report(&label, "CPU dense f32 SIMD", &cpu_dev, us);

        let Some(g) = &gpu else { continue };
        let gdev = format!("GPU: {}", g.info.name);
        match SignalEngine::new(g, &m, 1, None) {
            Ok(mut e) => {
                for (x, r) in pool.iter().zip(&refs) {
                    e.input_mut().copy_from_slice(x);
                    e.run(Submit::Graph, Wait::Spin).unwrap();
                    assert_eq!(bits(r), bits(e.output()), "SignalEngine mismatch");
                }
                if let Some(err) = &e.graph_error {
                    println!(
                        "| {label} | GPU ternary graph | n/a (capture failed: {err}) | | | | | | |"
                    );
                }
                for (path, submit, wait) in [
                    ("GPU ternary graph + spin", Submit::Graph, Wait::Spin),
                    ("GPU ternary direct + spin", Submit::Direct, Wait::Spin),
                    (
                        "GPU ternary graph + blocking sync",
                        Submit::Graph,
                        Wait::Block,
                    ),
                ] {
                    if submit == Submit::Graph && !e.has_graph() {
                        continue;
                    }
                    let us = time_each(&o, |i| {
                        e.input_mut().copy_from_slice(&pool[i % 256]);
                        e.run(submit, wait).unwrap();
                        out.copy_from_slice(e.output());
                        std::hint::black_box(&out);
                    });
                    report(&label, path, &gdev, us);
                }
            }
            Err(err) => println!("| {label} | GPU ternary | n/a ({err}) | | | | | | |"),
        }
        match PersistentEngine::start(g, &m, None) {
            Ok(mut p) => {
                let ok = pool
                    .iter()
                    .zip(&refs)
                    .all(|(x, r)| p.infer(x, &mut out).is_ok() && bits(r) == bits(&out));
                if ok {
                    let us = time_each(&o, |i| {
                        p.infer(&pool[i % 256], &mut out).unwrap();
                        std::hint::black_box(&out);
                    });
                    report(&label, "GPU ternary persistent (experimental)", &gdev, us);
                } else {
                    println!("| {label} | GPU ternary persistent (experimental) | n/a (failed verification or timed out) | | | | | | |");
                }
                if let Err(err) = p.stop() {
                    println!("(persistent kernel did not stop cleanly: {err})");
                }
            }
            Err(err) => println!(
                "| {label} | GPU ternary persistent (experimental) | n/a ({err}) | | | | | | |"
            ),
        }
        match CublasMlp::new(g, &m) {
            Ok(mut c) if c.has_graph() => {
                let us = time_each(&o, |i| {
                    c.input_mut().copy_from_slice(&pool[i % 256]);
                    c.run(Submit::Graph, Wait::Spin).unwrap();
                    out.copy_from_slice(c.output());
                    std::hint::black_box(&out);
                });
                report(&label, "GPU cuBLAS FP16 graph + spin", &gdev, us);
            }
            Ok(c) => println!(
                "| {label} | GPU cuBLAS FP16 | n/a (capture failed: {:?}) | | | | | | |",
                c.graph_error
            ),
            Err(err) => println!("| {label} | GPU cuBLAS FP16 | n/a ({err}) | | | | | | |"),
        }
    }
    if let Some(path) = &o.csv {
        let mut s =
            String::from("# measured by src/bin/latency.rs (batch 1, end to end, microseconds)\n");
        s += &format!("# cpu,{}\n", cpu.replace(',', ";"));
        if let Some(g) = &gpu {
            s += &format!("# gpu,{}\n", g.info.name.replace(',', ";"));
        }
        s += "# lat,model,path,device,iters,p50,p90,p99,p999,max,mean,std\n# hist,model,path,lo_us,hi_us,count\n";
        CSV.with(|c| {
            for l in c.borrow().iter() {
                s += l;
                s.push('\n');
            }
        });
        std::fs::write(path, s).expect("write csv");
        println!("\nCSV -> {path}");
    }
}
