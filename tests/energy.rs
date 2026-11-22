//! RAPL parsing against a fake powercap tree (the CI/VM hosts have no real counters), counter
//! wrap-around, report arithmetic, and graceful absence of counters.

use std::fs;
use std::path::PathBuf;
use ternary_gemm::energy::{EnergyMeter, EnergyReport, Rapl};

fn fake_tree(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("tgemm-rapl-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    for (zone, name, uj, max) in [
        (
            "intel-rapl:0",
            "package-0",
            1_000_000u64,
            262_143_328_850u64,
        ),
        ("intel-rapl:1", "package-1", 5_000_000, 262_143_328_850),
        ("intel-rapl:0:0", "core", 42, 262_143_328_850), // subzone: must be ignored
    ] {
        let d = root.join(zone);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("name"), format!("{name}\n")).unwrap();
        fs::write(d.join("energy_uj"), format!("{uj}\n")).unwrap();
        fs::write(d.join("max_energy_range_uj"), format!("{max}\n")).unwrap();
    }
    fs::create_dir_all(root.join("dtpm")).unwrap(); // unrelated zone
    root
}

#[test]
fn reads_top_level_packages_only() {
    let root = fake_tree("read");
    let r = Rapl::open_at(&root).expect("fake tree has RAPL zones");
    let names: Vec<&str> = r.domains.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["package-0", "package-1"]);
    assert_eq!(r.read_uj(), vec![1_000_000, 5_000_000]);
    fs::write(root.join("intel-rapl:0/energy_uj"), "3500000\n").unwrap();
    fs::write(root.join("intel-rapl:1/energy_uj"), "5250000\n").unwrap();
    let j = r.joules_between(&[1_000_000, 5_000_000], &r.read_uj());
    assert!((j - 2.75).abs() < 1e-12, "{j}");
    fs::remove_dir_all(root).ok();
}

#[test]
fn handles_counter_wraparound() {
    let root = fake_tree("wrap");
    let r = Rapl::open_at(&root).unwrap();
    let max = 262_143_328_850u64;
    // package-0 wrapped: from max - 1 J to 0.5 J => 1.5 J; package-1 +1 J.
    let j = r.joules_between(&[max - 1_000_000, 0], &[500_000, 1_000_000]);
    assert!((j - 2.5).abs() < 1e-9, "{j}");
    fs::remove_dir_all(root).ok();
}

#[test]
fn missing_powercap_is_none() {
    assert!(Rapl::open_at(std::path::Path::new("/nonexistent/powercap")).is_none());
}

#[test]
fn report_metrics() {
    let r = EnergyReport {
        seconds: 2.0,
        iterations: 1000,
        cpu_joules: Some(30.0),
        gpu_joules: Some(10.0),
    };
    assert_eq!(r.total_joules(), Some(40.0));
    assert_eq!(r.avg_watts(), Some(20.0));
    assert_eq!(r.joules_per_iter(), Some(0.04));
    assert_eq!(r.iters_per_joule(), Some(25.0));
    let none = EnergyReport {
        cpu_joules: None,
        gpu_joules: None,
        ..r
    };
    assert_eq!(none.total_joules(), None);
    assert_eq!(none.iters_per_joule(), None);
}

#[test]
fn meter_runs_the_workload_even_without_counters() {
    let m = EnergyMeter::open();
    let mut n = 0u64;
    let r = m.measure(100, || n += 1);
    assert_eq!(n, 100);
    assert_eq!(r.iterations, 100);
    assert!(r.seconds >= 0.0);
    if m.rapl.is_none() {
        assert!(r.cpu_joules.is_none());
    }
    eprintln!("counters on this host: {}", m.describe());
}
