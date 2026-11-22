//! Energy measurement: Intel/AMD RAPL counters on the CPU (Linux powercap sysfs) and, with the
//! `nvml` feature, the NVIDIA driver's total-energy counter on the GPU.
//!
//! Both are cumulative hardware counters read before and after a workload, so the result is
//! **measured** energy (package / board level, including everything else running at the time).
//! When no counter is available (VMs, containers without `/sys/class/powercap`, no driver) the
//! meters return `None` and callers must report energy as unavailable rather than estimate it.

use std::path::{Path, PathBuf};
use std::time::Instant;

/// One RAPL domain (e.g. `package-0`, `dram`).
#[derive(Debug, Clone)]
pub struct RaplDomain {
    pub name: String,
    energy_path: PathBuf,
    /// Counter wrap value (`max_energy_range_uj`).
    max_uj: u64,
}

/// Reads every top-level RAPL package domain under a powercap root.
#[derive(Debug, Clone)]
pub struct Rapl {
    pub domains: Vec<RaplDomain>,
}

fn read_u64(p: &Path) -> Option<u64> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

impl Rapl {
    /// Open `/sys/class/powercap` (needs read access to `energy_uj`, often root-only since
    /// Linux 5.10; see `docs/ENERGY.md`).
    pub fn open() -> Option<Self> {
        Self::open_at(Path::new("/sys/class/powercap"))
    }

    /// Open a powercap tree at `root` (tests point this at a fake tree).
    pub fn open_at(root: &Path) -> Option<Self> {
        let mut domains = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(root).ok()?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let fname = e.file_name().to_string_lossy().into_owned();
            // Top-level zones only (intel-rapl:0, intel-rapl:1, amd-rapl...), not subzones (:0:0).
            let is_zone = (fname.starts_with("intel-rapl:") || fname.starts_with("amd-rapl:"))
                && fname.matches(':').count() == 1;
            if !is_zone {
                continue;
            }
            let dir = e.path();
            let energy_path = dir.join("energy_uj");
            if read_u64(&energy_path).is_none() {
                continue;
            }
            let name = std::fs::read_to_string(dir.join("name"))
                .map(|s| s.trim().to_string())
                .unwrap_or(fname);
            let max_uj = read_u64(&dir.join("max_energy_range_uj")).unwrap_or(u64::MAX);
            domains.push(RaplDomain {
                name,
                energy_path,
                max_uj,
            });
        }
        (!domains.is_empty()).then_some(Self { domains })
    }

    /// Current counters in microjoules, one per domain.
    pub fn read_uj(&self) -> Vec<u64> {
        self.domains
            .iter()
            .map(|d| read_u64(&d.energy_path).unwrap_or(0))
            .collect()
    }

    /// Joules between two readings, summed over domains, handling counter wrap-around.
    pub fn joules_between(&self, before: &[u64], after: &[u64]) -> f64 {
        self.domains
            .iter()
            .zip(before.iter().zip(after))
            .map(|(d, (&b, &a))| {
                let delta = if a >= b {
                    a - b
                } else {
                    d.max_uj.saturating_sub(b) + a
                };
                delta as f64 * 1e-6
            })
            .sum()
    }
}

/// NVIDIA GPU energy via NVML's `nvmlDeviceGetTotalEnergyConsumption` (millijoules since driver
/// load, Volta and newer), loaded at runtime with `dlopen` so there is no link-time dependency.
#[cfg(feature = "nvml")]
pub struct Nvml {
    _lib: libloading::Library,
    device: *mut std::ffi::c_void,
    total_energy: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> i32,
}

#[cfg(feature = "nvml")]
impl Nvml {
    /// Load `libnvidia-ml.so.1` and open GPU `index`; `None` if the library, driver or counter is
    /// unavailable.
    pub fn open(index: u32) -> Option<Self> {
        type Init = unsafe extern "C" fn() -> i32;
        type Handle = unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32;
        type Energy = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> i32;
        // SAFETY: loading the vendor library; symbols are looked up with their documented C
        // signatures (NVML API reference) and only called while `_lib` is alive.
        unsafe {
            let lib = ["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"]
                .iter()
                .find_map(|n| libloading::Library::new(*n).ok())?;
            let init: libloading::Symbol<Init> = lib.get(b"nvmlInit_v2\0").ok()?;
            if init() != 0 {
                return None;
            }
            let handle: libloading::Symbol<Handle> =
                lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
            let mut device = std::ptr::null_mut();
            if handle(index, &mut device) != 0 {
                return None;
            }
            let total_energy: Energy = *lib
                .get::<Energy>(b"nvmlDeviceGetTotalEnergyConsumption\0")
                .ok()?;
            let mut probe = 0u64;
            if total_energy(device, &mut probe) != 0 {
                return None;
            }
            Some(Self {
                _lib: lib,
                device,
                total_energy,
            })
        }
    }

    /// Total energy counter in millijoules.
    pub fn read_mj(&self) -> Option<u64> {
        let mut v = 0u64;
        // SAFETY: `device` came from NVML for this library instance, which is still loaded.
        (unsafe { (self.total_energy)(self.device, &mut v) } == 0).then_some(v)
    }
}

/// Measured energy for a workload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyReport {
    pub seconds: f64,
    pub iterations: u64,
    /// CPU package energy (RAPL), if readable.
    pub cpu_joules: Option<f64>,
    /// GPU board energy (NVML), if available.
    pub gpu_joules: Option<f64>,
}

impl EnergyReport {
    /// CPU + GPU joules (whichever are available), `None` if neither is.
    pub fn total_joules(&self) -> Option<f64> {
        match (self.cpu_joules, self.gpu_joules) {
            (None, None) => None,
            (c, g) => Some(c.unwrap_or(0.0) + g.unwrap_or(0.0)),
        }
    }
    pub fn avg_watts(&self) -> Option<f64> {
        Some(self.total_joules()? / self.seconds)
    }
    pub fn joules_per_iter(&self) -> Option<f64> {
        Some(self.total_joules()? / self.iterations as f64)
    }
    /// Inferences per joule (the per-watt figure of merit: iterations/s per W).
    pub fn iters_per_joule(&self) -> Option<f64> {
        Some(self.iterations as f64 / self.total_joules()?)
    }
}

/// Holds whatever counters this machine exposes.
pub struct EnergyMeter {
    pub rapl: Option<Rapl>,
    #[cfg(feature = "nvml")]
    pub nvml: Option<Nvml>,
}

impl EnergyMeter {
    /// Open RAPL and (with feature `nvml`) GPU 0.
    pub fn open() -> Self {
        Self {
            rapl: Rapl::open(),
            #[cfg(feature = "nvml")]
            nvml: Nvml::open(0),
        }
    }

    /// Which counters are live, e.g. `"rapl(package-0) nvml"` or `"none"`.
    pub fn describe(&self) -> String {
        let mut v = Vec::new();
        if let Some(r) = &self.rapl {
            let names: Vec<&str> = r.domains.iter().map(|d| d.name.as_str()).collect();
            v.push(format!("rapl({})", names.join(",")));
        }
        #[cfg(feature = "nvml")]
        if self.nvml.is_some() {
            v.push("nvml".into());
        }
        if v.is_empty() {
            "none".into()
        } else {
            v.join(" ")
        }
    }

    /// Run `f` `iterations` times and report time and measured energy.
    pub fn measure(&self, iterations: u64, mut f: impl FnMut()) -> EnergyReport {
        let rapl0 = self.rapl.as_ref().map(|r| r.read_uj());
        #[cfg(feature = "nvml")]
        let gpu0 = self.nvml.as_ref().and_then(|n| n.read_mj());
        let t = Instant::now();
        for _ in 0..iterations {
            f();
        }
        let seconds = t.elapsed().as_secs_f64();
        let cpu_joules = self
            .rapl
            .as_ref()
            .zip(rapl0)
            .map(|(r, b)| r.joules_between(&b, &r.read_uj()));
        #[cfg(feature = "nvml")]
        let gpu_joules = self
            .nvml
            .as_ref()
            .and_then(|n| Some((n.read_mj()? as f64 - gpu0? as f64) * 1e-3));
        #[cfg(not(feature = "nvml"))]
        let gpu_joules = None;
        EnergyReport {
            seconds,
            iterations,
            cpu_joules,
            gpu_joules,
        }
    }
}
