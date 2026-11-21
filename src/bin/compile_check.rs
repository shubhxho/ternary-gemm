//! Compile the kernels with NVRTC for several architectures (no GPU needed) and, if `ptxas` is
//! available, assemble each PTX to SASS and print register / spill / shared-memory usage.
//!
//! Usage: cargo run --release --bin compile_check [-- compute_80 compute_120 ...]

use std::process::Command;

fn main() {
    let mut archs: Vec<String> = std::env::args().skip(1).collect();
    if archs.is_empty() {
        archs = [
            "compute_80",
            "compute_86",
            "compute_89",
            "compute_90",
            "compute_120",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
    }
    let ptxas = std::env::var("PTXAS").unwrap_or_else(|_| "ptxas".into());
    let have_ptxas = Command::new(&ptxas)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let out_dir = std::path::Path::new("target/ptx");
    std::fs::create_dir_all(out_dir).unwrap();
    let mut failed = false;
    for arch in &archs {
        let t = std::time::Instant::now();
        match ternary_gemm::gpu::compile(arch) {
            Ok(ptx) => {
                let src = ptx.to_src();
                let path = out_dir.join(format!("ternary_{arch}.ptx"));
                std::fs::write(&path, &src).unwrap();
                let n_entries = src.matches(".entry").count();
                println!(
                    "NVRTC {arch}: OK in {:.1}s, {} kernels, PTX -> {}",
                    t.elapsed().as_secs_f64(),
                    n_entries,
                    path.display()
                );
                if have_ptxas {
                    let sm = arch.replace("compute_", "sm_");
                    let cubin = out_dir.join(format!("ternary_{sm}.cubin"));
                    let o = Command::new(&ptxas)
                        .args(["-v", "-O3", &format!("--gpu-name={sm}"), "-o"])
                        .arg(&cubin)
                        .arg(&path)
                        .output()
                        .expect("run ptxas");
                    let log = String::from_utf8_lossy(&o.stderr);
                    if !o.status.success() {
                        println!("  ptxas {sm}: FAILED\n{log}");
                        failed = true;
                        continue;
                    }
                    // Summarize: one line per kernel with registers and spills.
                    let mut cur = String::new();
                    let mut spills = 0;
                    for line in log.lines() {
                        if let Some(i) = line.find("Compiling entry function '") {
                            let rest = &line[i + 26..];
                            cur = rest[..rest.find('\'').unwrap_or(rest.len())].to_string();
                        } else if line.contains("bytes spill stores") {
                            let spill = line.trim();
                            if !spill.contains(" 0 bytes spill stores, 0 bytes spill loads") {
                                spills += 1;
                                println!("  {sm} {cur:<28} SPILL: {spill}");
                            }
                        } else if line.contains("Used") && line.contains("registers") {
                            let used = line.split("Used").nth(1).unwrap_or("").trim().to_string();
                            println!("  {sm} {cur:<28} Used {used}");
                        }
                    }
                    println!("  {sm}: ptxas OK, {spills} kernels with register spills");
                }
            }
            Err(e) => {
                println!("NVRTC {arch}: FAILED\n{e}");
                failed = true;
            }
        }
    }
    if !have_ptxas {
        println!("(ptxas not found; set PTXAS=/path/to/ptxas to also check SASS register usage)");
    }
    if failed {
        std::process::exit(1);
    }
}
