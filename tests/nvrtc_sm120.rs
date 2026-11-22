//! Blackwell (sm_120) compile checks: NVRTC must compile every kernel the autotuner can pick for
//! `compute_120`, and if `PTXAS` points at a ptxas binary, the SASS register counts must match the
//! tables the occupancy model uses (`tuning::SM120_I8_REGS` / `SM120_I8_FUSED_REGS`).
//! Skipped when libnvrtc is missing unless `TERNARY_REQUIRE_NVRTC=1` (CI's nvrtc job sets it).

use std::process::Command;
use ternary_gemm::autotune::kernel_name;
use ternary_gemm::gpu::{
    compile, compile_ext, compile_tensor_cores, BATCH_TILES, ROWS_PER_WARP, TENSOR_CORE_KERNELS,
};
use ternary_gemm::tuning::{KernelConfig, SM120_I8_FUSED_REGS, SM120_I8_REGS};

fn ptx(arch: &str) -> Option<String> {
    match compile(arch) {
        Ok(p) => Some(p.to_src()),
        Err(e) => {
            if std::env::var("TERNARY_REQUIRE_NVRTC").as_deref() == Ok("1") {
                panic!("TERNARY_REQUIRE_NVRTC=1 but NVRTC failed: {e}");
            }
            eprintln!("SKIPPED (no NVRTC): {e}");
            None
        }
    }
}

fn all_configs() -> Vec<(KernelConfig, usize)> {
    let mut v = Vec::new();
    for fused in [false, true] {
        for &r in &ROWS_PER_WARP {
            for &b in &BATCH_TILES {
                v.push((KernelConfig::dp4a(4, r, fused), b));
            }
        }
    }
    v
}

#[test]
fn compute_120_has_every_tunable_kernel() {
    for arch in ["compute_120", "compute_120a"] {
        let Some(src) = ptx(arch) else { return };
        for (cfg, b) in all_configs() {
            let name = kernel_name(cfg, b);
            assert!(
                src.contains(&format!(".entry {name}(")),
                "{arch}: missing {name}"
            );
        }
    }
}

#[test]
fn sm120_register_table_matches_ptxas() {
    let Ok(ptxas) = std::env::var("PTXAS") else {
        eprintln!("SKIPPED (set PTXAS to check sm_120 register counts)");
        return;
    };
    let Some(src) = ptx("compute_120") else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("tgemm-sm120-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("k.ptx");
    std::fs::write(&p, src).unwrap();
    let o = Command::new(ptxas)
        .args(["-v", "-O3", "--gpu-name=sm_120", "-o"])
        .arg(dir.join("k.cubin"))
        .arg(&p)
        .output()
        .expect("run ptxas");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let log = String::from_utf8_lossy(&o.stderr);
    let mut regs = std::collections::HashMap::new();
    let mut cur = String::new();
    for line in log.lines() {
        if let Some(i) = line.find("Compiling entry function '") {
            let rest = &line[i + 26..];
            cur = rest[..rest.find('\'').unwrap()].to_string();
        } else if let Some(i) = line.find("Used ") {
            let n: u32 = line[i + 5..]
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap();
            regs.insert(cur.clone(), n);
        }
    }
    for (cfg, b) in all_configs() {
        let bi = BATCH_TILES.iter().position(|&t| t == b).unwrap();
        let ri = ROWS_PER_WARP
            .iter()
            .position(|&r| r == cfg.rows_per_warp)
            .unwrap();
        let table = if cfg.fused {
            SM120_I8_FUSED_REGS
        } else {
            SM120_I8_REGS
        };
        let name = kernel_name(cfg, b);
        assert_eq!(regs.get(&name), Some(&table[bi][ri]), "{name}");
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn compute_80_plus_has_tensor_core_kernels() {
    for arch in ["compute_80", "compute_89", "compute_120"] {
        match compile_tensor_cores(arch) {
            Ok(p) => {
                let src = p.to_src();
                for name in TENSOR_CORE_KERNELS {
                    assert!(
                        src.contains(&format!(".entry {name}(")),
                        "{arch}: missing {name}"
                    );
                    assert!(
                        src.contains("mma.sync.aligned.m16n8k16"),
                        "{arch}: missing mma.sync"
                    );
                }
            }
            Err(e) => {
                if std::env::var("TERNARY_REQUIRE_NVRTC").as_deref() == Ok("1") {
                    panic!("TERNARY_REQUIRE_NVRTC=1 but tensor-core NVRTC failed for {arch}: {e}");
                }
                eprintln!("SKIPPED (no NVRTC): {e}");
                return;
            }
        }
    }
}

/// The extra GEMV kernels (cp.async, split-K, LUT, warp-specialized) compile for every sm_80+
/// target, use the PTX features they are named for, and (with `PTXAS`) assemble without spills.
#[test]
fn ext_kernels_compile_for_sm80_through_sm120() {
    for arch in [
        "compute_80",
        "compute_86",
        "compute_89",
        "compute_90",
        "compute_120",
        "compute_120a",
    ] {
        let src = match compile_ext(arch) {
            Ok(p) => p.to_src(),
            Err(e) => {
                if std::env::var("TERNARY_REQUIRE_NVRTC").as_deref() == Ok("1") {
                    panic!("TERNARY_REQUIRE_NVRTC=1 but ext NVRTC failed for {arch}: {e}");
                }
                eprintln!("SKIPPED (no NVRTC): {e}");
                return;
            }
        };
        for name in ternary_gemm::cuda_ext::EXT_KERNELS {
            assert!(
                src.contains(&format!(".entry {name}(")),
                "{arch}: missing {name}"
            );
        }
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.wait_group",
            "bar.arrive",
            "dp4a",
        ] {
            assert!(src.contains(needle), "{arch}: PTX lacks {needle}");
        }
        let Ok(ptxas) = std::env::var("PTXAS") else {
            continue;
        };
        let sm = arch.replace("compute_", "sm_");
        let dir = std::env::temp_dir().join(format!("tgemm-ext-{}-{arch}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.ptx");
        std::fs::write(&p, &src).unwrap();
        let o = Command::new(&ptxas)
            .args(["-v", "-O3", &format!("--gpu-name={sm}"), "-o"])
            .arg(dir.join("x.cubin"))
            .arg(&p)
            .output()
            .expect("run ptxas");
        let log = String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success(), "{arch}: {log}");
        for line in log.lines().filter(|l| l.contains("bytes spill stores")) {
            assert!(
                line.contains(" 0 bytes spill stores, 0 bytes spill loads"),
                "{arch}: {line}"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
