//! Nasdaq ITCH 5.0 replay harness: public market data in, a small order-book model's signal out.
//!
//! ```text
//! itch filter  --symbols AAPL,MSFT [--out slice.itch] < day.NASDAQ_ITCH50   (stdin: decompressed)
//! itch replay  --data slice.itch [--model m.tgmlp --engine dense|dense-kmajor|ternary|gpu-ternary]
//!              [--samples-out s.bin] [--preds-out p.f32] [--json out.json] [--pin-note TEXT]
//! itch throughput --data slice.itch --model m.tgmlp --engine dense|dense-kmajor|ternary|gpu-ternary [--batch B]
//! ```
//!
//! `filter` keeps a valid, framed ITCH file containing only system events, the stock-directory
//! entries and every message for the chosen symbols, so the replay runs on real exchange bytes.
//!
//! `replay` loads the slice into memory (file I/O is outside the timed region), then for every
//! message times, with the TSC: parse -> book update -> (for decision points) features ->
//! inference. A decision point is a book change for a tracked symbol during 09:30-16:00 ET with
//! both sides present and uncrossed. It prints and writes p50/p99/p99.9/max per stage and
//! end-to-end, replay throughput, and process CPU time.
//!
//! `throughput` runs inference only, over every decision's feature vector back to back
//! (batch 1 sequentially on the CPU; `--batch B` independent samples per launch on the GPU).
//!
//! Model file (`.tgmlp`, written by `scripts/itch_train.py`), little-endian:
//! `b"TGMLP1\0\0"`, u32 n_dims, u32 dims[n_dims], u32 hidden_act (1 = relu), f32 mean[in],
//! f32 std[in], then per layer f32 w[n*k] (row-major `[n][k]`), f32 b[n]. `dense` runs those
//! weights in f32 with per-row dot products ([`DenseF32Mlp`]), `dense-kmajor` in f32 with
//! transposed weights and 512-bit FMAs ([`DenseF32KMajor`], the stronger dense baseline);
//! `ternary` quantizes them with absmean per row (what QAT trained against) and runs the
//! W2A8 [`CpuEngine`].

use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;
use ternary_gemm::itch::{self, Book, FeatState, Frames, Msg, FEATURES};
use ternary_gemm::mlp::{cpu_simd_level, Act, CpuEngine, DenseF32KMajor, DenseF32Mlp, TernaryMlp};

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1).cloned())
}

fn die(msg: &str) -> ! {
    eprintln!("itch: {msg}");
    std::process::exit(2)
}

// ---------------------------------------------------------------------------------------------
// Timing: TSC on x86_64 (calibrated against Instant), Instant elsewhere.

#[inline(always)]
fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        use std::sync::OnceLock;
        static T0: OnceLock<Instant> = OnceLock::new();
        T0.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Nanoseconds per tick, measured over ~200 ms.
fn calibrate() -> f64 {
    let (t0, c0) = (Instant::now(), ticks());
    while t0.elapsed().as_millis() < 200 {}
    let (dt, dc) = (t0.elapsed().as_nanos() as f64, (ticks() - c0) as f64);
    dt / dc
}

fn cpu_time_s() -> f64 {
    // /proc/self/stat utime + stime (clock ticks, 100 Hz on Linux).
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let rest = s.rsplit(')').next()?.split_whitespace().collect::<Vec<_>>();
            let ut: f64 = rest.get(11)?.parse().ok()?;
            let st: f64 = rest.get(12)?.parse().ok()?;
            Some((ut + st) / 100.0)
        })
        .unwrap_or(f64::NAN)
}

struct Stats {
    n: usize,
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    p999: f64,
    max: f64,
}

fn stats(v: &mut [u32], ns_per_tick: f64) -> Stats {
    if v.is_empty() {
        return Stats {
            n: 0,
            mean: f64::NAN,
            p50: f64::NAN,
            p90: f64::NAN,
            p99: f64::NAN,
            p999: f64::NAN,
            max: f64::NAN,
        };
    }
    v.sort_unstable();
    let q = |p: f64| v[((v.len() as f64 * p) as usize).min(v.len() - 1)] as f64 * ns_per_tick;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 * ns_per_tick;
    Stats {
        n: v.len(),
        mean,
        p50: q(0.5),
        p90: q(0.9),
        p99: q(0.99),
        p999: q(0.999),
        max: *v.last().unwrap() as f64 * ns_per_tick,
    }
}

fn stats_json(name: &str, s: &Stats) -> String {
    format!(
        "\"{name}\": {{\"n\": {}, \"mean_ns\": {:.1}, \"p50_ns\": {:.1}, \"p90_ns\": {:.1}, \"p99_ns\": {:.1}, \"p999_ns\": {:.1}, \"max_ns\": {:.1}}}",
        s.n, s.mean, s.p50, s.p90, s.p99, s.p999, s.max
    )
}

fn print_row(name: &str, s: &Stats) {
    println!(
        "| {name:<34} | {:>10} | {:>8.1} | {:>8.1} | {:>8.1} | {:>9.1} | {:>10.1} | {:>10.1} |",
        s.n, s.mean, s.p50, s.p90, s.p99, s.p999, s.max
    );
}

// ---------------------------------------------------------------------------------------------
// Model file

struct ModelFile {
    dims: Vec<usize>,
    act: Act,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
}

fn load_model(path: &str) -> ModelFile {
    let b = std::fs::read(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    if b.len() < 12 || &b[..8] != b"TGMLP1\0\0" {
        die(&format!("{path}: not a TGMLP1 file"));
    }
    let mut o = 8;
    let u32_ = |o: &mut usize| {
        let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let nd = u32_(&mut o) as usize;
    let dims: Vec<usize> = (0..nd).map(|_| u32_(&mut o) as usize).collect();
    let act = match u32_(&mut o) {
        0 => Act::Identity,
        1 => Act::Relu,
        2 => Act::Relu2,
        a => die(&format!("unknown activation {a}")),
    };
    let f32s = |o: &mut usize, n: usize| -> Vec<f32> {
        let v = b[*o..*o + 4 * n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        *o += 4 * n;
        v
    };
    let mean = f32s(&mut o, dims[0]);
    let std = f32s(&mut o, dims[0]);
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for i in 0..nd - 1 {
        weights.push(f32s(&mut o, dims[i] * dims[i + 1]));
        biases.push(f32s(&mut o, dims[i + 1]));
    }
    if o != b.len() {
        die(&format!("{path}: {} trailing bytes", b.len() - o));
    }
    if dims[0] != FEATURES {
        die(&format!("model input {} != {FEATURES} features", dims[0]));
    }
    let inv_std = std.iter().map(|&s| 1.0 / s.max(1e-6)).collect();
    ModelFile {
        dims,
        act,
        mean,
        inv_std,
        weights,
        biases,
    }
}

enum Engine {
    None,
    Dense(DenseF32Mlp),
    DenseKMajor(DenseF32KMajor),
    Ternary(Box<CpuEngine>),
}

impl Engine {
    #[inline(always)]
    fn run(&mut self, x: &[f32], y: &mut [f32]) {
        match self {
            Engine::None => {}
            Engine::Dense(m) => m.forward_into(x, y),
            Engine::DenseKMajor(m) => m.forward_into(x, y),
            Engine::Ternary(e) => e.forward_into(x, y),
        }
    }
}

fn build_engine(kind: &str, mf: Option<&ModelFile>) -> (Engine, String) {
    match (kind, mf) {
        ("none", _) => (Engine::None, "none (features only)".into()),
        ("dense", Some(m)) => {
            let e = DenseF32Mlp::from_f32(&m.dims, &m.weights, &m.biases, m.act).unwrap_or_else(|e| die(&e));
            (Engine::Dense(e), format!("dense f32 {:?} (avx2+fma dots if available)", m.dims))
        }
        ("dense-kmajor", Some(m)) => {
            let e = DenseF32KMajor::from_f32(&m.dims, &m.weights, &m.biases, m.act).unwrap_or_else(|e| die(&e));
            let simd = e.simd();
            (Engine::DenseKMajor(e), format!("dense f32 {:?} k-major [{simd}]", m.dims))
        }
        ("ternary", Some(m)) => {
            let t = TernaryMlp::from_f32(&m.dims, &m.weights, &m.biases, m.act).unwrap_or_else(|e| die(&e));
            let e = CpuEngine::new(&t).unwrap_or_else(|e| die(&e));
            let simd = e.simd();
            (Engine::Ternary(Box::new(e)), format!("ternary W2A8 {:?} CpuEngine [{simd}]", m.dims))
        }
        ("ternary-bytes", Some(m)) => {
            let t = TernaryMlp::from_f32(&m.dims, &m.weights, &m.biases, m.act).unwrap_or_else(|e| die(&e));
            let e = CpuEngine::new(&t).unwrap_or_else(|e| die(&e)).without_packed();
            let simd = e.simd();
            (
                Engine::Ternary(Box::new(e)),
                format!("ternary weights as 1-byte codes {:?} CpuEngine::without_packed [{simd}] (int8-weight cost proxy)", m.dims),
            )
        }
        (k, None) if k != "none" => die("--engine needs --model"),
        (k, _) => die(&format!("unknown engine {k} (dense|dense-kmajor|ternary|ternary-bytes|none; gpu-ternary needs the `cuda` feature)")),
    }
}

// ---------------------------------------------------------------------------------------------

fn filter(args: &[String]) {
    let syms: Vec<[u8; 8]> = arg(args, "--symbols")
        .unwrap_or_else(|| die("--symbols A,B,C required"))
        .split(',')
        .map(itch::symbol8)
        .collect();
    let out_path = arg(args, "--out").unwrap_or_else(|| "slice.itch".into());
    let max_msgs: u64 = arg(args, "--max-messages")
        .map(|s| s.parse().unwrap())
        .unwrap_or(u64::MAX);
    let mut r = BufReader::with_capacity(1 << 22, std::io::stdin().lock());
    let mut w = BufWriter::with_capacity(1 << 22, std::fs::File::create(&out_path).unwrap());
    let mut keep = vec![false; 65536];
    let (mut n_in, mut n_out) = (0u64, 0u64);
    let mut hdr = [0u8; 2];
    let mut body = [0u8; 65536];
    let mut found = Vec::new();
    let t0 = Instant::now();
    while n_in < max_msgs && r.read_exact(&mut hdr).is_ok() {
        let n = u16::from_be_bytes(hdr) as usize;
        if r.read_exact(&mut body[..n]).is_err() {
            break;
        }
        n_in += 1;
        let b = &body[..n];
        let t = b[0];
        let locate = if n >= 3 {
            u16::from_be_bytes([b[1], b[2]])
        } else {
            0
        };
        let mut k = t == b'S';
        if t == b'R' && n >= 19 {
            let s: [u8; 8] = b[11..19].try_into().unwrap();
            if syms.contains(&s) {
                keep[locate as usize] = true;
                found.push((String::from_utf8_lossy(&s).trim().to_string(), locate));
                k = true;
            }
        } else if keep[locate as usize] && t != b'S' {
            k = true;
        }
        if k {
            w.write_all(&hdr).unwrap();
            w.write_all(b).unwrap();
            n_out += 1;
        }
        if n_in % 50_000_000 == 0 {
            eprintln!(
                "  {n_in} messages read, {n_out} kept, {:.0} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    w.flush().unwrap();
    println!("read {n_in} messages, kept {n_out} -> {out_path}");
    for (s, l) in found {
        println!("  {s} locate={l}");
    }
}

struct Replay {
    buf: Vec<u8>,
    locates: Vec<(u16, String)>,
}

fn load_slice(path: &str) -> Replay {
    let buf = std::fs::read(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    let mut locates = Vec::new();
    for f in Frames::new(&buf) {
        if let Some(Msg::StockDir { locate, stock }) = itch::parse(f) {
            locates.push((locate, String::from_utf8_lossy(&stock).trim().to_string()));
        }
    }
    Replay { buf, locates }
}

/// Run every decision's normalized features through the engine once to collect them (used by
/// `throughput` and the GPU path).
fn collect_features(rp: &Replay, mf: &ModelFile) -> Vec<f32> {
    let track: Vec<u16> = rp.locates.iter().map(|l| l.0).collect();
    let mut book = Book::new(&track);
    let mut st = vec![FeatState::default(); 65536];
    let mut x = [0f32; FEATURES];
    let mut all = Vec::new();
    for f in Frames::new(&rp.buf) {
        let Some(m) = itch::parse(f) else { continue };
        let Some(u) = book.apply(&m) else { continue };
        let sb = book.sym(u.locate).unwrap();
        if itch::features(sb, &mut st[u.locate as usize], &u, &mut x).is_some() {
            itch::normalize(&mut x, &mf.mean, &mf.inv_std);
            all.extend_from_slice(&x);
        }
    }
    all
}

fn replay(args: &[String]) {
    let data = arg(args, "--data").unwrap_or_else(|| die("--data required"));
    let model = arg(args, "--model").map(|p| load_model(&p));
    let engine_kind = arg(args, "--engine").unwrap_or_else(|| {
        if model.is_some() {
            "ternary".into()
        } else {
            "none".into()
        }
    });
    let rp = load_slice(&data);
    let track: Vec<u16> = rp.locates.iter().map(|l| l.0).collect();
    eprintln!("slice: {} bytes, symbols {:?}", rp.buf.len(), rp.locates);

    #[cfg(feature = "cuda")]
    if engine_kind == "gpu-ternary" {
        return gpu::replay(
            &rp,
            model.as_ref().unwrap_or_else(|| die("--model required")),
            args,
        );
    }
    let (mut engine, engine_desc) = build_engine(&engine_kind, model.as_ref());
    let out_dim = model.as_ref().map(|m| *m.dims.last().unwrap()).unwrap_or(1);
    let ns = calibrate();

    // Timer overhead: two back-to-back reads.
    let mut ovh: Vec<u32> = (0..100_000)
        .map(|_| {
            let a = ticks();
            (ticks() - a) as u32
        })
        .collect();
    let ovh_s = stats(&mut ovh, ns);

    let mut samples_out =
        arg(args, "--samples-out").map(|p| BufWriter::new(std::fs::File::create(p).unwrap()));
    let mut preds_out =
        arg(args, "--preds-out").map(|p| BufWriter::new(std::fs::File::create(p).unwrap()));

    // Two passes: pass 0 warms caches, branch predictors and the hash map's allocation; pass 1 is
    // timed and recorded (the book is rebuilt from scratch in both).
    let n_msgs = Frames::new(&rp.buf).count();
    let mut t_parse: Vec<u32> = Vec::with_capacity(n_msgs);
    let mut t_book: Vec<u32> = Vec::with_capacity(n_msgs);
    let mut t_feat: Vec<u32> = Vec::new();
    let mut t_inf: Vec<u32> = Vec::new();
    let mut t_e2e: Vec<u32> = Vec::new();
    let mut t_dec_parse_book: Vec<u32> = Vec::new();
    let mut y = vec![0f32; out_dim];
    let mut x = [0f32; FEATURES];
    let (mut wall_s, mut cpu_s, mut n_dec, mut unknown) = (0.0, 0.0, 0usize, 0u64);
    let mut checked = 0usize;
    let ref_model = match (&engine, model.as_ref()) {
        (Engine::Ternary(_), Some(m)) => {
            Some(TernaryMlp::from_f32(&m.dims, &m.weights, &m.biases, m.act).unwrap())
        }
        _ => None,
    };
    for pass in 0..2 {
        let record = pass == 1;
        let mut book = Book::new(&track);
        let mut st = vec![FeatState::default(); 65536];
        t_parse.clear();
        t_book.clear();
        n_dec = 0;
        let (w0, c0) = (Instant::now(), cpu_time_s());
        for f in Frames::new(&rp.buf) {
            let a = ticks();
            let Some(m) = itch::parse(f) else { continue };
            let b = ticks();
            let upd = book.apply(&m);
            let c = ticks();
            if record {
                t_parse.push((b - a) as u32);
                t_book.push((c - b) as u32);
            }
            let Some(u) = upd else { continue };
            let sb = book.sym(u.locate).unwrap();
            let Some(d) = itch::features(sb, &mut st[u.locate as usize], &u, &mut x) else {
                continue;
            };
            if let Some(mf) = model.as_ref() {
                itch::normalize(&mut x, &mf.mean, &mf.inv_std);
            }
            let e = ticks();
            engine.run(&x, &mut y);
            let g = ticks();
            n_dec += 1;
            if record {
                t_feat.push((e - c) as u32);
                t_inf.push((g - e) as u32);
                t_e2e.push((g - a) as u32);
                t_dec_parse_book.push((c - a) as u32);
                if let Some(w) = samples_out.as_mut() {
                    w.write_all(&d.ts.to_le_bytes()).unwrap();
                    w.write_all(&d.mid2.to_le_bytes()).unwrap();
                    w.write_all(&d.locate.to_le_bytes()).unwrap();
                    w.write_all(&[0u8; 2]).unwrap();
                    for v in &x {
                        w.write_all(&v.to_le_bytes()).unwrap();
                    }
                }
                if let Some(w) = preds_out.as_mut() {
                    w.write_all(&y[0].to_le_bytes()).unwrap();
                }
                if let Some(rm) = ref_model.as_ref() {
                    if checked < 2000 {
                        let r = rm.forward_reference(&x);
                        assert_eq!(
                            r[0].to_bits(),
                            y[0].to_bits(),
                            "ternary engine != scalar reference"
                        );
                        checked += 1;
                    }
                }
            }
        }
        if record {
            wall_s = w0.elapsed().as_secs_f64();
            cpu_s = cpu_time_s() - c0;
            unknown = book.unknown_refs;
        }
    }
    if let Some(w) = samples_out.as_mut() {
        w.flush().unwrap();
    }
    if let Some(w) = preds_out.as_mut() {
        w.flush().unwrap();
    }

    // Untimed pass for clean replay throughput (no per-stage timers, no output writes).
    let w0 = Instant::now();
    {
        let mut book = Book::new(&track);
        let mut st = vec![FeatState::default(); 65536];
        for f in Frames::new(&rp.buf) {
            let Some(m) = itch::parse(f) else { continue };
            let Some(u) = book.apply(&m) else { continue };
            let sb = book.sym(u.locate).unwrap();
            if itch::features(sb, &mut st[u.locate as usize], &u, &mut x).is_some() {
                if let Some(mf) = model.as_ref() {
                    itch::normalize(&mut x, &mf.mean, &mf.inv_std);
                }
                engine.run(&x, &mut y);
                std::hint::black_box(&y);
            }
        }
    }
    let clean_s = w0.elapsed().as_secs_f64();

    let sp = stats(&mut t_parse, ns);
    let sb = stats(&mut t_book, ns);
    let sdpb = stats(&mut t_dec_parse_book, ns);
    let sf = stats(&mut t_feat, ns);
    let si = stats(&mut t_inf, ns);
    let se = stats(&mut t_e2e, ns);
    println!("engine: {engine_desc}");
    println!("cpu simd level: {}", cpu_simd_level());
    println!("messages: {n_msgs}, decisions (RTH book updates, two-sided): {n_dec}, unknown order refs: {unknown}");
    println!(
        "tsc: {:.4} ns/tick; timer overhead p50 {:.1} ns (included in each stage below)",
        ns, ovh_s.p50
    );
    if checked > 0 {
        println!("ternary engine bit-exact vs scalar reference on {checked} replayed decisions");
    }
    println!();
    println!("| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |");
    println!("|---|---|---|---|---|---|---|---|");
    print_row("parse (all messages)", &sp);
    print_row("book update (all messages)", &sb);
    print_row("parse + book (decision msgs)", &sdpb);
    print_row("features + normalize", &sf);
    print_row("inference", &si);
    print_row("end-to-end msg -> decision", &se);
    println!();
    let mps = n_msgs as f64 / clean_s;
    println!(
        "replay throughput (untimed pass): {:.2} M msgs/s, {:.2} M decisions/s, {:.3} s wall",
        mps / 1e6,
        n_dec as f64 / clean_s / 1e6,
        clean_s
    );
    println!(
        "timed pass: {:.3} s wall, {:.3} s CPU ({:.0}% of one core)",
        wall_s,
        cpu_s,
        100.0 * cpu_s / wall_s
    );
    let inf_share = si.mean / se.mean;
    println!(
        "inference share of mean end-to-end decision latency: {:.1}%",
        100.0 * inf_share
    );

    if let Some(p) = arg(args, "--json") {
        let syms: Vec<String> = rp
            .locates
            .iter()
            .map(|(l, s)| format!("{{\"locate\": {l}, \"symbol\": \"{s}\"}}"))
            .collect();
        let j = format!(
            "{{\n\"engine\": \"{engine_kind}\", \"engine_desc\": \"{engine_desc}\", \"simd\": \"{}\",\n\"data\": \"{data}\", \"symbols\": [{}],\n\"messages\": {n_msgs}, \"decisions\": {n_dec}, \"unknown_refs\": {unknown}, \"ns_per_tick\": {ns}, \"timer_overhead_p50_ns\": {:.1},\n\"replay_throughput_msgs_per_s\": {mps:.0}, \"replay_clean_wall_s\": {clean_s:.4}, \"timed_wall_s\": {wall_s:.4}, \"timed_cpu_s\": {cpu_s:.4},\n\"bitexact_checked\": {checked},\n{},\n{},\n{},\n{},\n{},\n{}\n}}\n",
            cpu_simd_level(),
            syms.join(", "),
            ovh_s.p50,
            stats_json("parse", &sp),
            stats_json("book", &sb),
            stats_json("parse_book_decision", &sdpb),
            stats_json("features", &sf),
            stats_json("inference", &si),
            stats_json("end_to_end", &se),
        );
        std::fs::write(&p, j).unwrap();
        eprintln!("wrote {p}");
    }
}

fn throughput(args: &[String]) {
    let data = arg(args, "--data").unwrap_or_else(|| die("--data required"));
    let mf = load_model(&arg(args, "--model").unwrap_or_else(|| die("--model required")));
    let kind = arg(args, "--engine").unwrap_or_else(|| "ternary".into());
    let rp = load_slice(&data);
    let feats = collect_features(&rp, &mf);
    let n = feats.len() / FEATURES;
    #[cfg(feature = "cuda")]
    if kind == "gpu-ternary" {
        return gpu::throughput(&feats, &mf, args);
    }
    let (mut engine, desc) = build_engine(&kind, Some(&mf));
    let mut y = vec![0f32; *mf.dims.last().unwrap()];
    for i in 0..n.min(10_000) {
        engine.run(&feats[i * FEATURES..(i + 1) * FEATURES], &mut y);
    }
    let reps: usize = arg(args, "--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        for i in 0..n {
            engine.run(&feats[i * FEATURES..(i + 1) * FEATURES], &mut y);
            std::hint::black_box(&y);
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!("engine: {desc}");
    println!(
        "inference-only, batch 1 back to back over {n} replayed feature vectors: {:.1} ns/sample, {:.2} M samples/s (best of {reps})",
        best / n as f64 * 1e9,
        n as f64 / best / 1e6
    );
}

#[cfg(feature = "cuda")]
mod gpu {
    //! GPU path (compile-checked on a machine without a GPU; run `scripts/itch_gpu.sh` on one).
    use super::*;
    use ternary_gemm::gpu::Gpu;
    use ternary_gemm::signal::{SignalEngine, Submit, Wait};

    fn model(mf: &ModelFile) -> TernaryMlp {
        TernaryMlp::from_f32(&mf.dims, &mf.weights, &mf.biases, mf.act).unwrap_or_else(|e| die(&e))
    }

    /// Same replay loop as the CPU one, with inference = host features -> GPU (CUDA graph,
    /// pinned buffers, spin wait) -> host signal, batch 1.
    pub fn replay(rp: &Replay, mf: &ModelFile, args: &[String]) {
        let gpu = Gpu::new(0).unwrap_or_else(|e| die(&format!("no CUDA device: {e}")));
        let m = model(mf);
        let mut e = SignalEngine::new(&gpu, &m, 1, None).unwrap_or_else(|e| die(&e.to_string()));
        if !e.has_graph() {
            eprintln!(
                "warning: CUDA graph capture failed ({:?}); direct launches",
                e.graph_error
            );
        }
        let ns = calibrate();
        let track: Vec<u16> = rp.locates.iter().map(|l| l.0).collect();
        let mut x = [0f32; FEATURES];
        let mut y = [0f32; 1];
        let (mut t_inf, mut t_e2e, mut t_pb) = (Vec::new(), Vec::new(), Vec::new());
        let mut checked = 0;
        let limit: usize = arg(args, "--max-decisions")
            .map(|s| s.parse().unwrap())
            .unwrap_or(usize::MAX);
        for pass in 0..2 {
            let mut book = Book::new(&track);
            let mut st = vec![FeatState::default(); 65536];
            let mut n = 0usize;
            for f in Frames::new(&rp.buf) {
                let a = ticks();
                let Some(msg) = itch::parse(f) else { continue };
                let Some(u) = book.apply(&msg) else { continue };
                let c = ticks();
                let sb = book.sym(u.locate).unwrap();
                if itch::features(sb, &mut st[u.locate as usize], &u, &mut x).is_none() {
                    continue;
                }
                itch::normalize(&mut x, &mf.mean, &mf.inv_std);
                let e0 = ticks();
                e.infer(&x, &mut y).unwrap();
                let g = ticks();
                n += 1;
                if pass == 1 {
                    t_inf.push((g - e0) as u32);
                    t_e2e.push((g - a) as u32);
                    t_pb.push((c - a) as u32);
                    if checked < 2000 {
                        assert_eq!(
                            m.forward_reference(&x)[0].to_bits(),
                            y[0].to_bits(),
                            "GPU != reference"
                        );
                        checked += 1;
                    }
                }
                if n >= limit {
                    break;
                }
            }
        }
        println!(
            "engine: GPU ternary SignalEngine batch 1 (graph + pinned + spin), {}",
            gpu.info.name
        );
        println!("GPU output bit-exact vs scalar reference on {checked} decisions");
        println!("| stage (ns) | n | mean | p50 | p90 | p99 | p99.9 | max |");
        println!("|---|---|---|---|---|---|---|---|");
        print_row("parse + book (decision msgs)", &stats(&mut t_pb, ns));
        print_row("inference (GPU round trip)", &stats(&mut t_inf, ns));
        print_row("end-to-end msg -> decision", &stats(&mut t_e2e, ns));
    }

    /// Inference only: `--batch B` decisions per launch (graph + spin), back to back.
    pub fn throughput(feats: &[f32], mf: &ModelFile, args: &[String]) {
        let gpu = Gpu::new(0).unwrap_or_else(|e| die(&format!("no CUDA device: {e}")));
        let m = model(mf);
        let batch: usize = arg(args, "--batch")
            .map(|s| s.parse().unwrap())
            .unwrap_or(1);
        let mut e =
            SignalEngine::new(&gpu, &m, batch, None).unwrap_or_else(|e| die(&e.to_string()));
        let n = feats.len() / FEATURES / batch * batch;
        let chunk = batch * FEATURES;
        let run = |e: &mut SignalEngine| {
            for c in feats[..n * FEATURES].chunks_exact(chunk) {
                e.input_mut().copy_from_slice(c);
                e.run(Submit::Graph, Wait::Spin).unwrap();
                std::hint::black_box(e.output());
            }
        };
        run(&mut e);
        let t0 = Instant::now();
        run(&mut e);
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "GPU ternary SignalEngine batch {batch} on {}: {:.1} ns/sample, {:.2} M samples/s, {:.2} us/launch over {n} samples",
            gpu.info.name,
            dt / n as f64 * 1e9,
            n as f64 / dt / 1e6,
            dt / (n / batch) as f64 * 1e6
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("filter") => filter(&args),
        Some("replay") => replay(&args),
        Some("throughput") => throughput(&args),
        _ => die(
            "usage: itch filter|replay|throughput ... (see the module docs / docs/ITCH_REPLAY.md)",
        ),
    }
}
