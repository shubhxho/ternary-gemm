//! Architecture tuning profiles and an analytic launch-config model (no GPU needed).
//!
//! Per-architecture limits come from public NVIDIA documentation (CUDA Programming Guide,
//! "Compute Capabilities" tables; the RTX Blackwell architecture whitepaper for the RTX 5090; the
//! Ampere / Ada / Hopper whitepapers for the others). Sources are listed in `docs/RTX5090.md`.
//! Kernel register counts are the `ptxas` output of `cargo run --bin compile_check` for sm_120
//! (see `docs/compile_check_output.txt`); on a real device the autotuner reads them from the
//! driver instead.
//!
//! Everything in this module is a **model**: it ranks launch configurations and produces
//! **projections**. Measurements come only from the autotuner / benchmarks on real hardware.

/// Static per-architecture limits (plus the flagship board's SM count / bandwidth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchSpec {
    /// Marketing name of the reference board used for the bandwidth / SM figures.
    pub board: &'static str,
    pub compute_capability: (i32, i32),
    /// SMs on the reference board (the autotuner uses the device's real count).
    pub sm_count: u32,
    pub max_warps_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub regs_per_sm: u32,
    /// Shared memory per SM available to blocks (bytes).
    pub smem_per_sm: u32,
    /// Max dynamic + static shared memory per block after the opt-in (bytes).
    pub smem_per_block_optin: u32,
    /// L2 cache on the reference board (bytes).
    pub l2_bytes: u64,
    /// Peak DRAM bandwidth of the reference board (GB/s, 1e9 B/s).
    pub dram_gbps: f64,
    /// Boost clock of the reference board (MHz).
    pub boost_mhz: f64,
    /// INT32 lanes per SM (`dp4a` issue model, see [`project_gemv`]).
    pub int32_lanes_per_sm: u32,
}

const KIB: u32 = 1024;
const MIB: u64 = 1024 * 1024;

/// RTX 5090 (GB202, sm_120): 170 SMs, 1792 GB/s GDDR7, 96 MB L2, 2407 MHz boost.
pub const SM120_RTX5090: ArchSpec = ArchSpec {
    board: "GeForce RTX 5090",
    compute_capability: (12, 0),
    sm_count: 170,
    max_warps_per_sm: 48,
    // Programming Guide table lists 24 for 12.x; the Blackwell Tuning Guide says 32. Use the
    // conservative value.
    max_blocks_per_sm: 24,
    regs_per_sm: 64 * KIB,
    smem_per_sm: 100 * KIB,
    smem_per_block_optin: 99 * KIB,
    l2_bytes: 96 * MIB,
    dram_gbps: 1792.0,
    boost_mhz: 2407.0,
    int32_lanes_per_sm: 64,
};

/// RTX 4090 (AD102, sm_89).
pub const SM89_RTX4090: ArchSpec = ArchSpec {
    board: "GeForce RTX 4090",
    compute_capability: (8, 9),
    sm_count: 128,
    max_warps_per_sm: 48,
    max_blocks_per_sm: 24,
    regs_per_sm: 64 * KIB,
    smem_per_sm: 100 * KIB,
    smem_per_block_optin: 99 * KIB,
    l2_bytes: 72 * MIB,
    dram_gbps: 1008.0,
    boost_mhz: 2520.0,
    int32_lanes_per_sm: 64,
};

/// RTX 3090 (GA102, sm_86).
pub const SM86_RTX3090: ArchSpec = ArchSpec {
    board: "GeForce RTX 3090",
    compute_capability: (8, 6),
    sm_count: 82,
    max_warps_per_sm: 48,
    max_blocks_per_sm: 16,
    regs_per_sm: 64 * KIB,
    smem_per_sm: 100 * KIB,
    smem_per_block_optin: 99 * KIB,
    l2_bytes: 6 * MIB,
    dram_gbps: 936.0,
    boost_mhz: 1695.0,
    int32_lanes_per_sm: 64,
};

/// A100 SXM4 40 GB (GA100, sm_80).
pub const SM80_A100: ArchSpec = ArchSpec {
    board: "A100 SXM4 40GB",
    compute_capability: (8, 0),
    sm_count: 108,
    max_warps_per_sm: 64,
    max_blocks_per_sm: 32,
    regs_per_sm: 64 * KIB,
    smem_per_sm: 164 * KIB,
    smem_per_block_optin: 163 * KIB,
    l2_bytes: 40 * MIB,
    dram_gbps: 1555.0,
    boost_mhz: 1410.0,
    int32_lanes_per_sm: 64,
};

/// H100 SXM5 80 GB (GH100, sm_90).
pub const SM90_H100: ArchSpec = ArchSpec {
    board: "H100 SXM5 80GB",
    compute_capability: (9, 0),
    sm_count: 132,
    max_warps_per_sm: 64,
    max_blocks_per_sm: 32,
    regs_per_sm: 64 * KIB,
    smem_per_sm: 228 * KIB,
    smem_per_block_optin: 227 * KIB,
    l2_bytes: 50 * MIB,
    dram_gbps: 3350.0,
    boost_mhz: 1980.0,
    int32_lanes_per_sm: 64,
};

/// All built-in profiles, newest first.
pub const PROFILES: [ArchSpec; 5] = [
    SM120_RTX5090,
    SM90_H100,
    SM89_RTX4090,
    SM86_RTX3090,
    SM80_A100,
];

/// The profile for a compute capability: exact match, else the newest older one with the same
/// major version, else the newest older one overall (limits only grow with newer parts), else
/// the oldest profile.
pub fn profile_for(cc: (i32, i32)) -> ArchSpec {
    if let Some(p) = PROFILES.iter().find(|p| p.compute_capability == cc) {
        return *p;
    }
    PROFILES
        .iter()
        .filter(|p| p.compute_capability.0 == cc.0 && p.compute_capability <= cc)
        .chain(PROFILES.iter().filter(|p| p.compute_capability <= cc))
        .copied()
        .next()
        .unwrap_or(SM80_A100)
}

/// Occupancy of one kernel configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy {
    pub blocks_per_sm: u32,
    pub warps_per_sm: u32,
    /// Which resource bounds it: "warps", "blocks", "registers" or "smem".
    pub limiter: &'static str,
}

/// CUDA-occupancy-calculator style estimate (registers allocated per warp in units of 256).
pub fn occupancy(
    spec: &ArchSpec,
    warps_per_block: u32,
    regs_per_thread: u32,
    smem: u32,
) -> Occupancy {
    let wpb = warps_per_block.max(1);
    let regs_per_warp = (regs_per_thread.max(1) * 32).div_ceil(256) * 256;
    let by_warps = spec.max_warps_per_sm / wpb;
    let by_blocks = spec.max_blocks_per_sm;
    let by_regs = (spec.regs_per_sm / regs_per_warp) / wpb;
    // 1 KiB per block is reserved by the runtime on sm_80+.
    let by_smem = spec.smem_per_sm / (smem + KIB);
    let (blocks, limiter) = [
        (by_warps, "warps"),
        (by_blocks, "blocks"),
        (by_regs, "registers"),
        (by_smem, "smem"),
    ]
    .into_iter()
    .min_by_key(|(b, _)| *b)
    .unwrap();
    Occupancy {
        blocks_per_sm: blocks,
        warps_per_sm: blocks * wpb,
        limiter,
    }
}

/// `ptxas` register counts of the W2A8 kernels for sm_120, `[batch tile][rows per warp]` with
/// batch tiles `[1, 2, 4, 8, 16]` and rows per warp `[1, 2, 4]` (from `compile_check`).
pub const SM120_I8_REGS: [[u32; 3]; 5] = [
    [38, 40, 48],
    [44, 66, 98],
    [48, 66, 124],
    [52, 80, 128],
    [62, 108, 204],
];

/// Fused-kernel (`ternary_i8_fused_*`) register counts for sm_120, same indexing.
pub const SM120_I8_FUSED_REGS: [[u32; 3]; 5] = [
    [40, 40, 48],
    [48, 64, 97],
    [40, 64, 122],
    [48, 76, 128],
    [58, 122, 164],
];

/// Bytes each warp keeps in flight per main-loop step: one 16-byte `uint4` per lane per row.
pub const BYTES_IN_FLIGHT_PER_WARP_ROW: u32 = 512;

/// A launch configuration of the W2A8 kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelConfig {
    pub warps: u32,
    pub rows_per_warp: u32,
    /// Fused quantize-in-prologue kernel (single launch) instead of quantize + gemm.
    pub fused: bool,
}

/// One modelled configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Modelled {
    pub config: KernelConfig,
    pub occupancy: Occupancy,
    /// Grid blocks / (SMs * resident blocks per SM).
    pub waves: f64,
    /// Bytes in flight per SM / bytes needed to cover DRAM latency (Little's law), capped at 1.
    pub latency_cover: f64,
    /// Modelled kernel time (us): a ranking device, **not** a prediction to quote.
    pub t_model_us: f64,
    /// `1 / t_model_us`, normalised so higher is better; used only to rank / prune.
    pub score: f64,
}

/// Assumed DRAM latency for the Little's-law term (ns). A model parameter, not a measurement.
pub const ASSUMED_DRAM_LATENCY_NS: f64 = 600.0;
/// Assumed L2 : DRAM bandwidth ratio for re-reading activations (L2 bandwidth of consumer parts
/// is not published). A model parameter.
pub const ASSUMED_L2_DRAM_RATIO: f64 = 2.5;
/// Assumed cost of one extra dependent kernel launch on a stream (us). A model parameter.
pub const ASSUMED_LAUNCH_US: f64 = 2.0;

fn tile_index(batch: usize) -> Option<usize> {
    [1usize, 2, 4, 8, 16].iter().position(|&t| t >= batch)
}

/// Model one configuration for an `N x K` GEMM at `batch` on `spec` with `sm_count` SMs.
pub fn model_config(
    spec: &ArchSpec,
    sm_count: u32,
    n: usize,
    k: usize,
    batch: usize,
    cfg: KernelConfig,
    regs: Option<u32>,
) -> Option<Modelled> {
    let bt = tile_index(batch)?;
    let ri = [1u32, 2, 4].iter().position(|&r| r == cfg.rows_per_warp)?;
    if !(1..=8).contains(&cfg.warps) || batch == 0 {
        return None;
    }
    let smem = if cfg.fused {
        let need = (batch * k) as u32 + KIB;
        if need > spec.smem_per_block_optin {
            return None;
        }
        need
    } else {
        0
    };
    let table = if cfg.fused {
        &SM120_I8_FUSED_REGS
    } else {
        &SM120_I8_REGS
    };
    let regs = regs.unwrap_or(table[bt][ri]);
    let occ = occupancy(spec, cfg.warps, regs, smem);
    if occ.blocks_per_sm == 0 {
        return None;
    }
    let rows_per_block = (cfg.warps * cfg.rows_per_warp) as usize;
    let blocks = n.div_ceil(rows_per_block) as f64;
    let resident = (sm_count * occ.blocks_per_sm) as f64;
    let waves = blocks / resident;
    // Little's law: bytes in flight per SM needed = (BW per SM) * latency.
    let need = spec.dram_gbps * 1e9 / spec.sm_count as f64 * ASSUMED_DRAM_LATENCY_NS * 1e-9;
    let active_blocks_per_sm = (blocks / sm_count as f64).min(occ.blocks_per_sm as f64);
    let inflight = active_blocks_per_sm
        * (cfg.warps * cfg.rows_per_warp * BYTES_IN_FLIGHT_PER_WARP_ROW) as f64;
    let latency_cover = (inflight / need).min(1.0);
    // SM coverage / tail effect: with fewer blocks than SMs some SMs idle; with more than one
    // full wave the last partial wave idles part of the GPU.
    let tail = if waves > 1.0 {
        waves / waves.ceil()
    } else {
        (blocks / sm_count as f64).min(1.0)
    };
    let bw = spec.dram_gbps * 1e9 * (sm_count as f64 / spec.sm_count as f64);
    let weight_bytes = (n * k) as f64 / 4.0 + 4.0 * n as f64;
    // The fused kernel re-reads and re-quantizes the fp16 activations in every block (from L2);
    // the split path reads int8 activations per block but pays one extra launch.
    let (act_bytes_per_block, launches) = if cfg.fused {
        ((batch * k * 2) as f64, 1.0)
    } else {
        ((batch * k) as f64, 2.0)
    };
    let act_bytes = blocks * act_bytes_per_block;
    let t_model_us = (weight_bytes / bw + act_bytes / (bw * ASSUMED_L2_DRAM_RATIO)) * 1e6
        / (latency_cover * tail).max(1e-6)
        + (launches - 1.0) * ASSUMED_LAUNCH_US;
    let score = 1.0 / t_model_us;
    Some(Modelled {
        config: cfg,
        occupancy: occ,
        waves,
        latency_cover,
        t_model_us,
        score,
    })
}

/// Every valid configuration, best modelled score first.
pub fn rank_configs(
    spec: &ArchSpec,
    sm_count: u32,
    n: usize,
    k: usize,
    batch: usize,
) -> Vec<Modelled> {
    let mut out = Vec::new();
    for fused in [false, true] {
        for warps in [1u32, 2, 4, 8] {
            for rows_per_warp in [1u32, 2, 4] {
                let cfg = KernelConfig {
                    warps,
                    rows_per_warp,
                    fused,
                };
                if let Some(m) = model_config(spec, sm_count, n, k, batch, cfg, None) {
                    out.push(m);
                }
            }
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

/// The autotuner's search space: the `top` best modelled configs plus the library default
/// (4 warps, heuristic rows) so a tuned result is never worse than the untuned one.
pub fn candidates(
    spec: &ArchSpec,
    sm_count: u32,
    n: usize,
    k: usize,
    batch: usize,
    top: usize,
) -> Vec<KernelConfig> {
    let mut c: Vec<KernelConfig> = rank_configs(spec, sm_count, n, k, batch)
        .into_iter()
        .take(top)
        .map(|m| m.config)
        .collect();
    for rows_per_warp in [1, 2, 4] {
        let d = KernelConfig {
            warps: 4,
            rows_per_warp,
            fused: false,
        };
        if !c.contains(&d) {
            c.push(d);
        }
    }
    c
}

/// Roofline projection for one W2A8 GEMV/GEMM. **A projection from public specs, not a
/// measurement.**
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    /// Packed weight + scale bytes read once.
    pub bytes: f64,
    /// Multiply-accumulates.
    pub macs: f64,
    /// Time if DRAM bandwidth were the only limit (us).
    pub t_dram_us: f64,
    /// Time if `dp4a` issue were the only limit (us); see [`project_gemv`].
    pub t_int_us: f64,
    /// `max(t_dram, t_int)` (us).
    pub t_bound_us: f64,
    pub bound: &'static str,
    /// Whether the weights fit in L2 (then DRAM is not the limit after the first touch).
    pub fits_l2: bool,
}

/// Roofline for `out[batch][n] = W[n][k] . q[batch][k]` with 2-bit weights.
///
/// Compute roof model (an assumption, stated so it can be checked): one `dp4a` = 4 MACs, issued
/// at `int32_lanes_per_sm` per SM per clock, and the kernel spends about one extra integer op per
/// `dp4a` decoding the 2-bit codes, so effective MAC rate = `sm * lanes * clock * 4 / 2`.
pub fn project_gemv(spec: &ArchSpec, n: usize, k: usize, batch: usize) -> Projection {
    let bytes = (n * k) as f64 / 4.0 + 4.0 * n as f64 + (batch * k) as f64;
    let macs = (n * k * batch) as f64;
    let t_dram_us = bytes / (spec.dram_gbps * 1e9) * 1e6;
    let mac_rate =
        spec.sm_count as f64 * spec.int32_lanes_per_sm as f64 * spec.boost_mhz * 1e6 * 2.0;
    let t_int_us = macs / mac_rate * 1e6;
    let (t_bound_us, bound) = if t_dram_us >= t_int_us {
        (t_dram_us, "dram")
    } else {
        (t_int_us, "int")
    };
    Projection {
        bytes,
        macs,
        t_dram_us,
        t_int_us,
        t_bound_us,
        bound,
        fits_l2: (n * k) as u64 / 4 <= spec.l2_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_matches_hand_calculation_on_sm120() {
        // 4 warps, 48 regs -> 1536 regs/warp -> 42 warps by registers -> 10 blocks; warps allow 12.
        let o = occupancy(&SM120_RTX5090, 4, 48, 0);
        assert_eq!((o.blocks_per_sm, o.limiter), (10, "registers"));
        // 204 regs (b16 r4): 6656 regs/warp -> 9 warps -> 2 blocks of 4 warps.
        let o = occupancy(&SM120_RTX5090, 4, 204, 0);
        assert_eq!(o.blocks_per_sm, 2);
        // 32 regs, 1 warp: block limit binds.
        let o = occupancy(&SM120_RTX5090, 1, 32, 0);
        assert_eq!((o.blocks_per_sm, o.limiter), (24, "blocks"));
        // 40 KiB smem: 100 / 41 -> 2 blocks.
        assert_eq!(occupancy(&SM120_RTX5090, 2, 32, 40 * 1024).limiter, "smem");
    }

    #[test]
    fn profile_lookup() {
        assert_eq!(profile_for((12, 0)).board, "GeForce RTX 5090");
        assert_eq!(profile_for((12, 1)).compute_capability, (12, 0));
        assert_eq!(profile_for((8, 7)).compute_capability, (8, 6));
        assert_eq!(profile_for((7, 5)).compute_capability, (8, 0));
    }

    #[test]
    fn ranking_is_valid_and_includes_default() {
        for &(n, k) in &[
            (2560usize, 2560usize),
            (6912, 2560),
            (640, 2560),
            (2560, 6912),
        ] {
            for batch in [1, 4, 16] {
                let r = rank_configs(&SM120_RTX5090, 170, n, k, batch);
                assert!(!r.is_empty());
                assert!(r.windows(2).all(|w| w[0].score >= w[1].score));
                for m in &r {
                    if m.config.fused {
                        assert!(batch * k + 1024 <= 99 * 1024);
                    }
                }
                let c = candidates(&SM120_RTX5090, 170, n, k, batch, 6);
                assert!(c.iter().any(|c| c.warps == 4 && !c.fused));
            }
        }
    }

    #[test]
    fn projection_is_dram_bound_for_decode() {
        // BitNet-2B FFN up (6912 x 2560) at batch 1 on the 5090: 4.4 MB -> ~2.5 us at 1792 GB/s.
        let p = project_gemv(&SM120_RTX5090, 6912, 2560, 1);
        assert_eq!(p.bound, "dram");
        assert!((p.t_dram_us - p.bytes / 1792e3).abs() < 1e-9);
        assert!(p.fits_l2);
    }
}
