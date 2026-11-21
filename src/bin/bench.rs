#![allow(clippy::needless_range_loop)]
//! Benchmarks: ternary kernels vs cuBLAS FP16 and a naive ternary kernel, timed with CUDA events.
//!
//! Prints Markdown. Every GPU number printed is measured on the machine running this binary;
//! sections that are estimates say so explicitly.
//!
//! Usage: cargo run --release --bin bench -- [--quick] [--batches 1,4,16] [--layers 30] [--no-verify] [--csv out.csv]
//! Env:   PEAK_GBPS=448   override the peak DRAM bandwidth used for "% of peak".

use cudarc::driver::CudaSlice;
use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use ternary_gemm::gpu::{time_us, Cublas, Gpu, GpuTernary, Launch, Result, ROWS_PER_WARP};
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference;

/// (N out, K in, label)
const SHAPES: &[(usize, usize, &str)] = &[
    (2560, 2560, "BitNet-2B q/o"),
    (640, 2560, "BitNet-2B k/v"),
    (6912, 2560, "BitNet-2B gate/up"),
    (2560, 6912, "BitNet-2B down"),
    (11008, 4096, "7B-class up"),
    (4096, 11008, "7B-class down"),
];

/// BitNet-b1.58-2B-4T linear layers per transformer block (N, K), from its config.json:
/// hidden 2560, intermediate 6912, 20 heads / 5 KV heads (head_dim 128 -> k/v = 640), 30 layers.
const BITNET_2B_LAYER: &[(usize, usize)] = &[
    (2560, 2560),
    (640, 2560),
    (640, 2560),
    (2560, 2560),
    (6912, 2560),
    (6912, 2560),
    (2560, 6912),
];
const BITNET_2B_LAYERS: usize = 30;

struct Opts {
    batches: Vec<usize>,
    layers: usize,
    reps: usize,
    target_ms: f64,
    verify: bool,
    csv: Option<String>,
}

fn parse() -> Opts {
    let mut o = Opts {
        batches: vec![1, 2, 4, 8, 16],
        layers: BITNET_2B_LAYERS,
        reps: 9,
        target_ms: 40.0,
        verify: true,
        csv: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quick" => {
                o.batches = vec![1, 16];
                o.reps = 5;
                o.target_ms = 15.0;
            }
            "--batches" => {
                i += 1;
                o.batches = args[i]
                    .split(',')
                    .map(|s| s.trim().parse().expect("batch"))
                    .collect();
            }
            "--layers" => {
                i += 1;
                o.layers = args[i].parse().expect("layers");
            }
            "--no-verify" => o.verify = false,
            "--csv" => {
                i += 1;
                o.csv = Some(args[i].clone());
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    o
}

/// Time `f` with an adaptive iteration count (~target_ms per sample), median of `reps` samples.
fn measure(g: &Gpu, o: &Opts, mut f: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for i in 0..3 {
        f(i)?;
    }
    g.synchronize()?;
    let probe = time_us(g, 3, 1, &mut f)?;
    let iters = ((o.target_ms * 1e3 / probe.max(0.5)) as usize).clamp(3, 5000);
    time_us(g, iters, o.reps, &mut f)
}

/// Number of buffer copies to rotate through so the working set is > 3x L2 (defeats L2 reuse).
fn copies(g: &Gpu, bytes: usize) -> usize {
    (3 * g.info.l2_bytes).div_ceil(bytes.max(1)).clamp(1, 64)
}

fn rand_f16(g: &Gpu, len: usize, seed: u32) -> Result<CudaSlice<f16>> {
    let mut b = g.alloc_f16(len)?;
    g.fill_f16(&mut b, seed)?;
    Ok(b)
}

thread_local! {
    static CSV: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// Record one machine-readable result line (written to `--csv <path>` at the end; used for charts).
fn csv(line: String) {
    CSV.with(|c| c.borrow_mut().push(line));
}

struct Row {
    kernel: String,
    us: f64,
    bytes: f64,
    note: String,
}

fn main() {
    let o = parse();
    println!("# ternary-gemm benchmark\n");
    let g = match Gpu::new(0) {
        Ok(g) => g,
        Err(e) => {
            no_gpu(&e.to_string());
            return;
        }
    };
    if let Err(e) = run(&g, &o) {
        eprintln!("benchmark failed: {e}");
        std::process::exit(1);
    }
}

fn no_gpu(why: &str) {
    println!("**No usable CUDA GPU: {why}**\n");
    println!("All GPU latency / bandwidth numbers: **NOT MEASURED**.\n");
    println!("## CPU reference sanity check (measured on this CPU; *not* a GPU number)\n");
    println!("| shape (N x K) | batch | CPU reference time | note |\n|---|---|---|---|");
    let mut rng = StdRng::seed_from_u64(1);
    for &(n, k) in &[(2560usize, 2560usize), (2560, 6912)] {
        let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let p = PackedTernary::from_ternary(&wt, n, k, vec![1.0; n]).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect();
        let a = reference::quantize_acts(&x, 1, k, None, 1e-5);
        let t = Instant::now();
        let y = reference::gemm_i8(&p, &a, None);
        std::hint::black_box(y);
        println!(
            "| {n} x {k} | 1 | {:.1} ms | scalar oracle, unoptimized by design |",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    println!();
}

fn run(g: &Gpu, o: &Opts) -> Result<()> {
    let info = &g.info;
    let peak_env = std::env::var("PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    let peak = peak_env.unwrap_or(info.attr_peak_gbps);
    println!("| | |\n|---|---|");
    println!(
        "| GPU | {} (sm_{}{}, {} SMs) |",
        info.name, info.compute_capability.0, info.compute_capability.1, info.sm_count
    );
    println!(
        "| Memory | {:.1} GiB, L2 {:.0} MiB |",
        info.total_mem as f64 / (1u64 << 30) as f64,
        info.l2_bytes as f64 / (1 << 20) as f64
    );
    println!(
        "| Peak DRAM BW used for % | {:.0} GB/s ({}) |",
        peak,
        if peak_env.is_some() {
            "from PEAK_GBPS".to_string()
        } else {
            "driver attributes: 2 x mem clock x bus width".to_string()
        }
    );
    println!(
        "| Kernels compiled for | {} (NVRTC, driver-JIT to SASS) |",
        info.ptx_arch
    );
    println!(
        "| Fused-kernel smem limit | batch x K <= {} B |",
        info.fused_smem_limit
    );
    println!("| Timing | CUDA events, median of {} samples, weights rotated through >= 3x L2 so they stream from DRAM |\n", o.reps);

    let cublas = Cublas::new(g)?;
    let mut rng = StdRng::seed_from_u64(1234);

    if o.verify {
        verify(g, &mut rng, o)?;
    }

    println!("## Per-shape kernel latency\n");
    println!("`eff GB/s` = compulsory bytes (weights + scales + activations in + outputs) / latency. `vs cuBLAS` = cuBLAS FP16 latency / kernel latency.\n");
    println!("| shape (N x K) | batch | kernel | latency (us) | eff GB/s | % of peak | vs cuBLAS | notes |");
    println!("|---|---|---|---|---|---|---|---|");
    for &(n, k, label) in SHAPES {
        for &batch in &o.batches {
            let rows = bench_shape(g, &cublas, o, n, k, batch)?;
            let base = rows[0].us;
            for r in rows {
                println!(
                    "| {n} x {k} ({label}) | {batch} | {} | {:.2} | {:.0} | {:.0}% | {:.2}x | {} |",
                    r.kernel,
                    r.us,
                    r.bytes / r.us / 1e3,
                    100.0 * r.bytes / r.us / 1e3 / peak,
                    base / r.us,
                    r.note
                );
                csv(format!(
                    "kernel,{n},{k},{label},{batch},{},{:.4},{:.3},{:.3},{:.4}",
                    r.kernel.replace("**", ""),
                    r.us,
                    r.bytes / r.us / 1e3,
                    100.0 * r.bytes / r.us / 1e3 / peak,
                    base / r.us
                ));
            }
        }
    }
    println!();
    layer_stack(g, &cublas, o)?;
    roofline(peak);
    if let Some(path) = &o.csv {
        let mut out = String::from("# measured on this GPU by src/bin/bench.rs\n");
        out += &format!(
            "# gpu,{},sm_{}{},peak_gbps,{peak:.1}\n",
            info.name, info.compute_capability.0, info.compute_capability.1
        );
        out +=
            "# kernel,n,k,label,batch,kernel_name,latency_us,eff_gbps,pct_peak,speedup_vs_cublas\n";
        out += "# layer_stack,batch,path,layers_allocated,step_us,tokens_per_s,speedup_vs_cublas\n";
        CSV.with(|c| {
            for l in c.borrow().iter() {
                out += l;
                out.push('\n');
            }
        });
        std::fs::write(path, out).expect("write csv");
        eprintln!("wrote {path}");
    }
    Ok(())
}

fn verify(g: &Gpu, rng: &mut StdRng, o: &Opts) -> Result<()> {
    println!("## Correctness vs CPU reference (run before timing)\n");
    println!("| shape | batch | int8 (quantize + gemv) | int8 fused | fp16 act. | naive int8 |\n|---|---|---|---|---|---|");
    let mut all_ok = true;
    for &(n, k, _) in SHAPES {
        for &batch in &[1usize, *o.batches.iter().max().unwrap()] {
            let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
            let scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.05)).collect();
            let p = PackedTernary::from_ternary(&wt, n, k, scales).unwrap();
            let xh: Vec<f16> = (0..batch * k)
                .map(|_| f16::from_f32(rng.random_range(-2.0f32..2.0)))
                .collect();
            let w = g.upload_weights(&p)?;
            let x = g.upload_f16(&xh)?;
            let ca = reference::quantize_acts(&xh, batch, k, None, 1e-5);
            let exp = reference::gemm_i8(&p, &ca, None);
            let mut acts = g.alloc_acts(batch, k)?;
            g.quantize(&x, None, 1e-5, &mut acts)?;
            let mut out = g.alloc_f16(batch * n)?;
            let bitexact =
                |v: Vec<f16>| v.iter().zip(&exp).all(|(a, b)| a.to_bits() == b.to_bits());
            g.gemm_i8(&w, &acts, None, &mut out, Launch::default())?;
            let a = bitexact(g.download(&out)?);
            let b = if batch * k <= g.info.fused_smem_limit {
                g.gemm_i8_fused(&w, &x, batch, None, 1e-5, None, &mut out, Launch::default())?;
                Some(bitexact(g.download(&out)?))
            } else {
                None
            };
            g.naive_i8(&w, &acts, None, &mut out)?;
            let d = bitexact(g.download(&out)?);
            let ef = reference::gemm_f16(&p, &xh, batch, None);
            g.gemm_f16(&w, &x, batch, None, &mut out, Launch::default())?;
            let got = g.download(&out)?;
            let max_ref = ef.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let err = got
                .iter()
                .zip(&ef)
                .map(|(a, b)| (a.to_f32() - b).abs())
                .fold(0f32, f32::max);
            let c = err <= 2e-3 * max_ref + 1e-3;
            all_ok &= a && b.unwrap_or(true) && c && d;
            let bx = |v: bool| if v { "bit-exact" } else { "**MISMATCH**" };
            println!(
                "| {n} x {k} | {batch} | {} | {} | {} (max abs err {err:.2e}, max |y| {max_ref:.2}) | {} |",
                bx(a),
                b.map(bx).unwrap_or("n/a (smem)"),
                if c { "within tol" } else { "**OUT OF TOL**" },
                bx(d)
            );
        }
    }
    println!();
    if !all_ok {
        println!("**Correctness check FAILED; timings below are for broken kernels.**\n");
    }
    Ok(())
}

fn bench_shape(
    g: &Gpu,
    cublas: &Cublas,
    o: &Opts,
    n: usize,
    k: usize,
    batch: usize,
) -> Result<Vec<Row>> {
    let tern_bytes = n * k / 4 + n * 4;
    let fp16_bytes = n * k * 2;
    let nt = copies(g, tern_bytes);
    let nf = copies(g, fp16_bytes);
    let tw: Vec<GpuTernary> = (0..nt)
        .map(|i| g.random_weights(n, k, 100 + i as u32))
        .collect::<Result<_>>()?;
    let fw: Vec<_> = (0..nf)
        .map(|i| rand_f16(g, n * k, 200 + i as u32))
        .collect::<Result<_>>()?;
    let x = rand_f16(g, batch * k, 7)?;
    let mut acts = g.alloc_acts(batch, k)?;
    g.quantize(&x, None, 1e-5, &mut acts)?;
    let mut out = g.alloc_f16(batch * n)?;
    let (x16, xq, y) = (
        (batch * k * 2) as f64,
        (batch * k) as f64,
        (batch * n * 2) as f64,
    );
    let mut rows = Vec::new();

    let t = measure(g, o, |i| {
        cublas.gemm(&fw[i % nf], &x, &mut out, n, k, batch)
    })?;
    rows.push(Row {
        kernel: "cuBLAS FP16 (GemmEx)".into(),
        us: t,
        bytes: fp16_bytes as f64 + x16 + y,
        note: "dense fp16 weights".into(),
    });

    let t = measure(g, o, |i| g.naive_i8(&tw[i % nt], &acts, None, &mut out))?;
    rows.push(Row {
        kernel: "naive ternary (int8 act.)".into(),
        us: t,
        bytes: tern_bytes as f64 + xq + y,
        note: "1 thread/output, pre-quantized acts".into(),
    });

    // Sweep rows-per-warp, keep the best (reported in notes).
    let sweep = |f: &mut dyn FnMut(Launch, usize) -> Result<()>| -> Result<(f64, u32)> {
        let mut best = (f64::MAX, 1);
        for r in ROWS_PER_WARP {
            let l = Launch {
                rows_per_warp: Some(r),
                warps: 4,
            };
            let t = measure(g, o, |i| f(l, i))?;
            if t < best.0 {
                best = (t, r);
            }
        }
        Ok(best)
    };

    let (t, r) = sweep(&mut |l, i| g.gemm_i8(&tw[i % nt], &acts, None, &mut out, l))?;
    rows.push(Row {
        kernel: "**ternary W2A8 gemv only**".into(),
        us: t,
        bytes: tern_bytes as f64 + xq + y,
        note: format!("pre-quantized acts, R={r}"),
    });

    let mut acts2 = g.alloc_acts(batch, k)?;
    let (t, r) = sweep(&mut |l, i| {
        g.quantize(&x, None, 1e-5, &mut acts2)?;
        g.gemm_i8(&tw[i % nt], &acts2, None, &mut out, l)
    })?;
    rows.push(Row {
        kernel: "**ternary W2A8 quantize + gemv**".into(),
        us: t,
        bytes: tern_bytes as f64 + x16 + y,
        note: format!("2 launches, fp16 in -> fp16 out, R={r}"),
    });

    if batch * k <= g.info.fused_smem_limit {
        let (t, r) = sweep(&mut |l, i| {
            g.gemm_i8_fused(&tw[i % nt], &x, batch, None, 1e-5, None, &mut out, l)
        })?;
        rows.push(Row {
            kernel: "**ternary W2A8 fused**".into(),
            us: t,
            bytes: tern_bytes as f64 + x16 + y,
            note: format!("1 launch, quant in prologue, R={r}"),
        });
    }

    let (t, r) = sweep(&mut |l, i| g.gemm_f16(&tw[i % nt], &x, batch, None, &mut out, l))?;
    rows.push(Row {
        kernel: "ternary W2A16 (fp16 act.)".into(),
        us: t,
        bytes: tern_bytes as f64 + x16 + y,
        note: format!("R={r}"),
    });
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn tern_step(
    g: &Gpu,
    layers: &[Vec<GpuTernary>],
    x: &CudaSlice<f16>,
    a2560: &mut ternary_gemm::gpu::GpuActs,
    a6912: &mut ternary_gemm::gpu::GpuActs,
    res: &CudaSlice<f16>,
    y: &mut CudaSlice<f16>,
) -> Result<()> {
    let l = Launch::default();
    for ws in layers {
        g.quantize(x, None, 1e-5, a2560)?; // attention input, shared by q, k, v
        for w in &ws[0..3] {
            g.gemm_i8(w, a2560, None, y, l)?;
        }
        g.quantize(x, None, 1e-5, a2560)?; // o-proj input
        g.gemm_i8(&ws[3], a2560, Some(res), y, l)?;
        g.quantize(x, None, 1e-5, a2560)?; // ffn input, shared by gate, up
        g.gemm_i8(&ws[4], a2560, None, y, l)?;
        g.gemm_i8(&ws[5], a2560, None, y, l)?;
        g.quantize(x, None, 1e-5, a6912)?; // down-proj input
        g.gemm_i8(&ws[6], a6912, Some(res), y, l)?;
    }
    Ok(())
}

fn f16_step(
    cublas: &Cublas,
    layers: &[Vec<CudaSlice<f16>>],
    x: &CudaSlice<f16>,
    y: &mut CudaSlice<f16>,
    batch: usize,
) -> Result<()> {
    for ws in layers {
        for (j, (n, k)) in BITNET_2B_LAYER.iter().enumerate() {
            cublas.gemm(&ws[j], x, y, *n, *k, batch)?;
        }
    }
    Ok(())
}

fn layer_stack(g: &Gpu, cublas: &Cublas, o: &Opts) -> Result<()> {
    println!("## BitNet-b1.58-2B linear-layer stack (decode)\n");
    println!("Per block: q, k, v, o, gate, up, down. Ternary path = 4 activation-quantize launches + 7 W2A8 GEMVs");
    println!("(q/k/v share one quantization, residual add fused into o and down). cuBLAS path = 7 FP16 GEMMs.");
    println!(
        "Each path is timed twice: launched op by op from the host, and replayed as one CUDA graph"
    );
    println!("(the whole token step captured once, which removes per-launch CPU/driver overhead).");
    println!("**Linear layers only**: attention, RoPE, norms outside the fused prologue, activation function,");
    println!("embeddings and lm_head are NOT included, so this is an upper bound on end-to-end tokens/s.\n");
    let per_layer_tern: usize = BITNET_2B_LAYER.iter().map(|(n, k)| n * k / 4 + n * 4).sum();
    let per_layer_f16: usize = BITNET_2B_LAYER.iter().map(|(n, k)| n * k * 2).sum();
    let (free, _) = g.ctx.mem_get_info()?;
    let budget = (free as f64 * 0.8) as usize;
    let lt = o.layers.min(budget / per_layer_tern).max(1);
    let lf = o.layers.min(budget / per_layer_f16).max(1);
    println!(
        "| batch | path | layers allocated | time / token step (all {} layers) | linear-only tokens/s | vs cuBLAS (no graph) |",
        o.layers
    );
    println!("|---|---|---|---|---|---|");
    let scaled = |l: usize| {
        if l < o.layers {
            format!("{l} (time scaled x{:.2})", o.layers as f64 / l as f64)
        } else {
            format!("{l}")
        }
    };
    for &batch in &o.batches {
        let x_h = rand_f16(g, batch * 6912, 1)?;
        let mut y_h = g.alloc_f16(batch * 6912)?;
        let mut rows: Vec<(String, usize, Option<f64>, String)> = Vec::new();

        // cuBLAS FP16
        let mut f16_layers = Vec::new();
        for li in 0..lf {
            let ws: Vec<_> = BITNET_2B_LAYER
                .iter()
                .enumerate()
                .map(|(j, (n, k))| rand_f16(g, n * k, (li * 16 + j) as u32))
                .collect::<Result<_>>()?;
            f16_layers.push(ws);
        }
        let t = measure(g, o, |_| {
            f16_step(cublas, &f16_layers, &x_h, &mut y_h, batch)
        })?;
        rows.push(("cuBLAS FP16".into(), lf, Some(t), String::new()));
        match g.capture(|| f16_step(cublas, &f16_layers, &x_h, &mut y_h, batch)) {
            Ok(graph) => {
                let t = measure(g, o, |_| graph.launch())?;
                rows.push((
                    "cuBLAS FP16 + CUDA graph".into(),
                    lf,
                    Some(t),
                    String::new(),
                ));
            }
            Err(e) => rows.push((
                "cuBLAS FP16 + CUDA graph".into(),
                lf,
                None,
                format!("capture failed: {e}"),
            )),
        }
        drop(f16_layers);

        // ternary W2A8
        let mut t_layers = Vec::new();
        for li in 0..lt {
            let ws: Vec<_> = BITNET_2B_LAYER
                .iter()
                .enumerate()
                .map(|(j, (n, k))| g.random_weights(*n, *k, (li * 16 + j) as u32))
                .collect::<Result<_>>()?;
            t_layers.push(ws);
        }
        let mut a2560 = g.alloc_acts(batch, 2560)?;
        let mut a6912 = g.alloc_acts(batch, 6912)?;
        let res = rand_f16(g, batch * 2560, 3)?;
        let t = measure(g, o, |_| {
            tern_step(g, &t_layers, &x_h, &mut a2560, &mut a6912, &res, &mut y_h)
        })?;
        rows.push(("ternary W2A8".into(), lt, Some(t), String::new()));
        match g.capture(|| tern_step(g, &t_layers, &x_h, &mut a2560, &mut a6912, &res, &mut y_h)) {
            Ok(graph) => {
                let t = measure(g, o, |_| graph.launch())?;
                rows.push((
                    "ternary W2A8 + CUDA graph".into(),
                    lt,
                    Some(t),
                    String::new(),
                ));
            }
            Err(e) => rows.push((
                "ternary W2A8 + CUDA graph".into(),
                lt,
                None,
                format!("capture failed: {e}"),
            )),
        }
        drop(t_layers);

        let base = rows[0].2.map(|t| t * o.layers as f64 / lf as f64);
        for (path, l, t, err) in rows {
            match t {
                Some(t) => {
                    let t = t * o.layers as f64 / l as f64;
                    let tok_s = batch as f64 * 1e6 / t;
                    let sp = base.map(|b| b / t).unwrap_or(f64::NAN);
                    println!(
                        "| {batch} | {path} | {} | {t:.0} us | {tok_s:.0} | {sp:.2}x |",
                        scaled(l)
                    );
                    csv(format!(
                        "layer_stack,{batch},{path},{l},{t:.3},{tok_s:.3},{sp:.4}"
                    ));
                }
                None => println!(
                    "| {batch} | {path} | {} | n/a | n/a | n/a ({err}) |",
                    scaled(l)
                ),
            }
        }
    }
    println!();
    Ok(())
}

fn roofline(peak: f64) {
    println!("## Theoretical estimate (NOT measured): DRAM-bandwidth roofline at batch 1\n");
    println!("t_min = bytes / peak BW with peak = {peak:.0} GB/s. Weights dominate: fp16 = 2 B/weight, 2-bit ternary = 0.25 B/weight,");
    println!("so the weight traffic ratio is exactly 8x; activations/outputs/scales make the total ratio slightly lower.\n");
    println!("| shape (N x K) | fp16 bytes | ternary bytes | t_min fp16 (us) | t_min ternary (us) | ideal speedup |\n|---|---|---|---|---|---|");
    for &(n, k, _) in SHAPES {
        let f = (n * k * 2 + k * 2 + n * 2) as f64;
        let t = (n * k / 4 + n * 4 + k * 2 + n * 2) as f64;
        println!(
            "| {n} x {k} | {:.2} MB | {:.2} MB | {:.2} | {:.2} | {:.2}x |",
            f / 1e6,
            t / 1e6,
            f / peak / 1e3,
            t / peak / 1e3,
            f / t
        );
    }
    println!();
}
