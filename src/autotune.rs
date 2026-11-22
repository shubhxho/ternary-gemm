//! Runtime autotuner for the W2A8 kernels (feature `cuda`).
//!
//! For each `(N, K, batch)` it takes the candidates the analytic model in [`crate::tuning`] ranks
//! highest for this GPU's architecture profile (using the device's real SM count and the
//! driver-reported register usage of each kernel), always adds the library default, times every
//! candidate with CUDA events on the device, and keeps the fastest. Results can be saved to and
//! loaded from a CSV cache keyed by device name, so tuning runs once per machine.
//!
//! Timings compare "quantize + gemm" (split) against the fused single-launch kernel on equal
//! terms: both include activation quantization.

use crate::gpu::{time_us, Gpu, Launch, Result, BATCH_TILES};
use crate::tuning::{candidates, profile_for, ArchSpec, KernelConfig};
use std::collections::HashMap;

/// The tuned choice for one problem shape. All times are measured on the device.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneEntry {
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub best: KernelConfig,
    /// Median time of the best config (us, includes quantization).
    pub best_us: f64,
    /// Median time of the untuned default (4 warps, heuristic rows per warp, split) (us).
    pub default_us: f64,
    /// Number of candidates timed.
    pub tried: usize,
}

impl TuneEntry {
    /// `Launch` to pass to [`Gpu::gemm_i8`] / [`Gpu::gemm_i8_fused`].
    pub fn launch(&self) -> Launch {
        Launch {
            rows_per_warp: Some(self.best.rows_per_warp),
            warps: self.best.warps,
        }
    }
}

/// Times candidate configurations on the device and caches the winners.
pub struct Autotuner<'g> {
    gpu: &'g Gpu,
    pub spec: ArchSpec,
    /// How many model-ranked candidates to time (plus the defaults).
    pub top: usize,
    pub iters: usize,
    pub reps: usize,
    cache: HashMap<(usize, usize, usize), TuneEntry>,
}

fn tile(batch: usize) -> usize {
    BATCH_TILES
        .iter()
        .copied()
        .find(|&t| t >= batch)
        .unwrap_or(16)
}

/// Kernel name for a config at a batch size.
pub fn kernel_name(cfg: KernelConfig, batch: usize) -> String {
    format!(
        "ternary_i8_{}b{}_r{}",
        if cfg.fused { "fused_" } else { "" },
        tile(batch),
        cfg.rows_per_warp
    )
}

impl<'g> Autotuner<'g> {
    pub fn new(gpu: &'g Gpu) -> Self {
        Self {
            gpu,
            spec: profile_for(gpu.info.compute_capability),
            top: 8,
            iters: 50,
            reps: 7,
            cache: HashMap::new(),
        }
    }

    /// Cached result for a shape, if tuned or loaded.
    pub fn get(&self, n: usize, k: usize, batch: usize) -> Option<&TuneEntry> {
        self.cache.get(&(n, k, batch))
    }

    /// Model-ranked candidates for this device (the search space `tune` will time).
    pub fn search_space(&self, n: usize, k: usize, batch: usize) -> Vec<KernelConfig> {
        let mut spec = self.spec;
        spec.smem_per_block_optin = spec
            .smem_per_block_optin
            .min(self.gpu.info.fused_smem_limit as u32 + 1024);
        candidates(&spec, self.gpu.info.sm_count, n, k, batch, self.top)
    }

    /// Time every candidate for `(n, k, batch)` and cache the fastest.
    pub fn tune(&mut self, n: usize, k: usize, batch: usize) -> Result<TuneEntry> {
        let g = self.gpu;
        let w = g.random_weights(n, k, 0x7e57)?;
        let mut x = g.alloc_f16(batch * k)?;
        g.fill_f16(&mut x, 0x5eed)?;
        let mut acts = g.alloc_acts(batch, k)?;
        let mut out = g.alloc_f16(batch * n)?;
        let mut time = |cfg: KernelConfig, l: Launch| -> Result<f64> {
            if cfg.fused {
                time_us(g, self.iters, self.reps, |_| {
                    g.gemm_i8_fused(&w, &x, batch, None, 1e-5, None, &mut out, l)
                })
            } else {
                time_us(g, self.iters, self.reps, |_| {
                    g.quantize(&x, None, 1e-5, &mut acts)?;
                    g.gemm_i8(&w, &acts, None, &mut out, l)
                })
            }
        };
        let default_cfg = KernelConfig {
            warps: 4,
            rows_per_warp: g.pick_rows_per_warp(n, 4),
            fused: false,
        };
        let default_us = time(default_cfg, Launch::default())?;
        let mut best = (default_cfg, default_us);
        let space = self.search_space(n, k, batch);
        for &cfg in &space {
            let l = Launch {
                rows_per_warp: Some(cfg.rows_per_warp),
                warps: cfg.warps,
            };
            // A config the device rejects (e.g. smem) is skipped, not fatal.
            if let Ok(us) = time(cfg, l) {
                if us < best.1 {
                    best = (cfg, us);
                }
            }
        }
        let e = TuneEntry {
            n,
            k,
            batch,
            best: best.0,
            best_us: best.1,
            default_us,
            tried: space.len() + 1,
        };
        self.cache.insert((n, k, batch), e.clone());
        Ok(e)
    }

    /// Driver-reported registers of the kernel a config would launch.
    pub fn kernel_regs(&self, cfg: KernelConfig, batch: usize) -> Option<u32> {
        self.gpu.kernel_regs(&kernel_name(cfg, batch))
    }

    /// CSV cache: `device,n,k,batch,warps,rows_per_warp,fused,best_us,default_us,tried`.
    pub fn to_csv(&self) -> String {
        let mut rows: Vec<&TuneEntry> = self.cache.values().collect();
        rows.sort_by_key(|e| (e.n, e.k, e.batch));
        let dev = self.gpu.info.name.replace(',', " ");
        let mut s =
            String::from("device,n,k,batch,warps,rows_per_warp,fused,best_us,default_us,tried\n");
        for e in rows {
            s += &format!(
                "{dev},{},{},{},{},{},{},{:.3},{:.3},{}\n",
                e.n,
                e.k,
                e.batch,
                e.best.warps,
                e.best.rows_per_warp,
                e.best.fused as u8,
                e.best_us,
                e.default_us,
                e.tried
            );
        }
        s
    }

    /// Load entries for this device from a CSV written by [`Self::to_csv`]; returns how many
    /// were loaded (rows for other devices are ignored).
    pub fn load_csv(&mut self, csv: &str) -> usize {
        let dev = self.gpu.info.name.replace(',', " ");
        let mut loaded = 0;
        for line in csv.lines().skip(1) {
            if let Some(e) = parse_row(line, &dev) {
                self.cache.insert((e.n, e.k, e.batch), e);
                loaded += 1;
            }
        }
        loaded
    }
}

fn parse_row(line: &str, dev: &str) -> Option<TuneEntry> {
    let f: Vec<&str> = line.split(',').collect();
    if f.len() != 10 || f[0] != dev {
        return None;
    }
    let u = |i: usize| f[i].parse::<usize>().ok();
    Some(TuneEntry {
        n: u(1)?,
        k: u(2)?,
        batch: u(3)?,
        best: KernelConfig {
            warps: f[4].parse().ok()?,
            rows_per_warp: f[5].parse().ok()?,
            fused: f[6] == "1",
        },
        best_us: f[7].parse().ok()?,
        default_us: f[8].parse().ok()?,
        tried: u(9)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_match_the_instantiations() {
        let c = |warps, rows_per_warp, fused| KernelConfig {
            warps,
            rows_per_warp,
            fused,
        };
        assert_eq!(kernel_name(c(4, 2, false), 3), "ternary_i8_b4_r2");
        assert_eq!(kernel_name(c(8, 4, true), 16), "ternary_i8_fused_b16_r4");
        assert_eq!(kernel_name(c(1, 1, false), 1), "ternary_i8_b1_r1");
    }

    #[test]
    fn csv_rows_roundtrip_and_filter_by_device() {
        let line = "NVIDIA GeForce RTX 5090,2560,2560,1,4,2,1,3.500,4.250,11";
        let e = parse_row(line, "NVIDIA GeForce RTX 5090").unwrap();
        assert_eq!((e.n, e.k, e.batch), (2560, 2560, 1));
        assert_eq!(
            e.best,
            KernelConfig {
                warps: 4,
                rows_per_warp: 2,
                fused: true
            }
        );
        assert_eq!(e.launch().rows_per_warp, Some(2));
        assert!(parse_row(line, "Other GPU").is_none());
        assert!(parse_row("garbage", "NVIDIA GeForce RTX 5090").is_none());
    }
}
