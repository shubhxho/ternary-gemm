//! Machine-readable benchmark results: environment capture and a small JSON writer for the
//! schema in `docs/results-schema.json` (validated and compared by `scripts/compare_results.py`).
//!
//! Every record produced here is a measurement taken by the calling binary on the machine named
//! in the environment block; nothing is filled in or estimated.

use std::fmt::Write as _;
use std::process::Command;

/// Schema version written to every document.
pub const SCHEMA_VERSION: u32 = 1;

/// Where and with what a run happened.
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    pub crate_version: String,
    pub git_sha: Option<String>,
    pub rustc: Option<String>,
    pub os: String,
    pub kernel: Option<String>,
    pub cpu: String,
    pub logical_cpus: usize,
    pub cpu_governor: Option<String>,
    pub gpu: Option<String>,
    pub unix_time: u64,
}

fn cmd(program: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(program).args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_trim(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

impl Environment {
    /// Capture the current machine. `gpu` is filled by callers that opened a device.
    pub fn capture() -> Self {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpu = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| std::env::consts::ARCH.to_string());
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: cmd("git", &["rev-parse", "HEAD"]),
            rustc: cmd("rustc", &["--version"]),
            os: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            kernel: read_trim("/proc/sys/kernel/osrelease"),
            cpu,
            logical_cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0),
            cpu_governor: read_trim("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            gpu: None,
            unix_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

/// One latency distribution (microseconds).
#[derive(Debug, Clone, PartialEq)]
pub struct LatencyRecord {
    pub model: String,
    pub path: String,
    pub device: String,
    pub iters: usize,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    pub std_us: f64,
    /// Measured joules per inference, when energy counters were available.
    pub joules_per_inference: Option<f64>,
}

/// JSON string literal with escaping.
pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn opt_str(v: &Option<String>) -> String {
    v.as_deref().map_or("null".into(), json_str)
}

/// JSON number; non-finite values (never expected) become `null` so the document stays valid.
fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".into()
    }
}

/// Serialize a latency results document (schema version [`SCHEMA_VERSION`]).
pub fn latency_json(tool: &str, env: &Environment, records: &[LatencyRecord]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "  \"schema_version\": {SCHEMA_VERSION},");
    let _ = writeln!(s, "  \"kind\": \"latency\",");
    let _ = writeln!(s, "  \"tool\": {},", json_str(tool));
    let _ = writeln!(s, "  \"environment\": {{");
    let _ = writeln!(
        s,
        "    \"crate_version\": {},",
        json_str(&env.crate_version)
    );
    let _ = writeln!(s, "    \"git_sha\": {},", opt_str(&env.git_sha));
    let _ = writeln!(s, "    \"rustc\": {},", opt_str(&env.rustc));
    let _ = writeln!(s, "    \"os\": {},", json_str(&env.os));
    let _ = writeln!(s, "    \"kernel\": {},", opt_str(&env.kernel));
    let _ = writeln!(s, "    \"cpu\": {},", json_str(&env.cpu));
    let _ = writeln!(s, "    \"logical_cpus\": {},", env.logical_cpus);
    let _ = writeln!(s, "    \"cpu_governor\": {},", opt_str(&env.cpu_governor));
    let _ = writeln!(s, "    \"gpu\": {},", opt_str(&env.gpu));
    let _ = writeln!(s, "    \"unix_time\": {}", env.unix_time);
    let _ = writeln!(s, "  }},");
    let _ = writeln!(s, "  \"results\": [");
    for (i, r) in records.iter().enumerate() {
        let _ = write!(
            s,
            "    {{\"model\": {}, \"path\": {}, \"device\": {}, \"iters\": {}, \"p50_us\": {}, \"p90_us\": {}, \"p99_us\": {}, \"p999_us\": {}, \"max_us\": {}, \"mean_us\": {}, \"std_us\": {}, \"joules_per_inference\": {}}}",
            json_str(&r.model),
            json_str(&r.path),
            json_str(&r.device),
            r.iters,
            num(r.p50_us),
            num(r.p90_us),
            num(r.p99_us),
            num(r.p999_us),
            num(r.max_us),
            num(r.mean_us),
            num(r.std_us),
            r.joules_per_inference.map_or("null".into(), num)
        );
        let _ = writeln!(s, "{}", if i + 1 < records.len() { "," } else { "" });
    }
    let _ = writeln!(s, "  ]");
    let _ = writeln!(s, "}}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_json_strings() {
        assert_eq!(json_str("a\"b\\c\n\u{1}"), "\"a\\\"b\\\\c\\n\\u0001\"");
    }

    #[test]
    fn document_shape() {
        let mut env = Environment::capture();
        env.git_sha = None;
        let r = LatencyRecord {
            model: "tiny".into(),
            path: "CPU \"x\"".into(),
            device: "CPU".into(),
            iters: 10,
            p50_us: 1.5,
            p90_us: 2.0,
            p99_us: 3.0,
            p999_us: 4.0,
            max_us: 5.0,
            mean_us: 1.7,
            std_us: 0.2,
            joules_per_inference: None,
        };
        let j = latency_json("latency", &env, &[r.clone(), r]);
        assert!(j.contains("\"schema_version\": 1"));
        assert!(j.contains("\"git_sha\": null"));
        assert!(j.contains("\"path\": \"CPU \\\"x\\\"\""));
        assert_eq!(j.matches("\"model\"").count(), 2);
        assert_eq!(num(f64::NAN), "null");
    }
}
