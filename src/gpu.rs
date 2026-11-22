//! CUDA backend: NVRTC-compiled kernels driven through `cudarc` with a safe API.
//!
//! Every public method validates shapes and buffer lengths before launching, so the API is safe:
//! out-of-bounds launches are rejected with [`Error::Shape`] instead of reaching the GPU.

use crate::pack::{PackedTernary, K_ALIGN};
use cudarc::cublas::{sys as cublas_sys, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;

/// The CUDA source of the dp4a / fused kernels (compiled at runtime with NVRTC).
pub const KERNEL_SOURCE: &str = include_str!("../kernels/ternary.cu");
/// Tensor-core kernels (`ternary_tc_i8`, `ternary_tc_f16`). Compiled only for sm_80+.
pub const TENSOR_CORE_SOURCE: &str = include_str!("../kernels/tensor_core.cu");

/// Entry points defined in [`TENSOR_CORE_SOURCE`].
pub const TENSOR_CORE_KERNELS: [&str; 2] = ["ternary_tc_i8", "ternary_tc_f16"];

/// Batch-size template instantiations available on the device.
pub const BATCH_TILES: [usize; 5] = [1, 2, 4, 8, 16];
/// Rows-per-warp instantiations.
pub const ROWS_PER_WARP: [u32; 3] = [1, 2, 4];
/// Conservative fused-kernel shared-memory budget (bytes) used when the opt-in attribute can't be
/// set: the 48 KiB default per-block limit minus static shared memory. The real limit for a device
/// is [`DeviceInfo::fused_smem_limit`], which opts in to the larger per-block carve-out
/// (~99 KiB on sm_86/89/120, ~163 KiB on sm_80, ~227 KiB on sm_90).
pub const FUSED_SMEM_LIMIT: usize = 47 * 1024;
/// Static shared memory reserved by the fused kernel (scales, sums, reduction scratch), rounded up.
const FUSED_STATIC_SMEM: usize = 1024;

#[derive(Debug)]
pub enum Error {
    NoCuda(String),
    Driver(cudarc::driver::DriverError),
    Compile(String),
    Cublas(cudarc::cublas::result::CublasError),
    Shape(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoCuda(s) => write!(f, "CUDA unavailable: {s}"),
            Error::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Error::Compile(s) => write!(f, "NVRTC compile error: {s}"),
            Error::Cublas(e) => write!(f, "cuBLAS error: {e}"),
            Error::Shape(s) => write!(f, "shape error: {s}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error::Driver(e)
    }
}
impl From<cudarc::cublas::result::CublasError> for Error {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        Error::Cublas(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;

fn shape_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Shape(msg.into()))
}

/// NVRTC options used for every compile. No fast-math: the int8 epilogue relies on IEEE div/mul.
pub fn nvrtc_options(arch: &str) -> CompileOptions {
    CompileOptions {
        options: vec![
            format!("--gpu-architecture={arch}"),
            "-std=c++17".into(),
            "-lineinfo".into(),
        ],
        name: Some("ternary.cu".into()),
        ..Default::default()
    }
}

fn nvrtc_present() -> Result<()> {
    // SAFETY: only checks whether the shared library can be dlopen'ed.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(Error::NoCuda(
            "libnvrtc not found (install the CUDA toolkit or `pip install nvidia-cuda-nvrtc-cu12`)"
                .into(),
        ));
    }
    Ok(())
}

fn compile_src(src: &str, arch: &str, name: &str) -> Result<Ptx> {
    let mut opts = nvrtc_options(arch);
    opts.name = Some(name.into());
    compile_ptx_with_opts(src, opts).map_err(|e| match e {
        cudarc::nvrtc::CompileError::CompileError { log, .. } => {
            Error::Compile(log.to_string_lossy().into_owned())
        }
        other => Error::Compile(format!("{other:?}")),
    })
}

/// Compile the dp4a / fused kernels to PTX for a virtual architecture such as `"compute_120"`.
/// Works without a GPU (only libnvrtc is needed).
pub fn compile(arch: &str) -> Result<Ptx> {
    nvrtc_present()?;
    compile_src(KERNEL_SOURCE, arch, "ternary.cu")
}

/// Compile the extra batch-1 GEMV kernels ([`crate::cuda_ext`]: cp.async pipeline, split-K, LUT,
/// warp-specialized) to PTX. `arch` must be sm_80+ / compute_80+ (`cp.async`).
pub fn compile_ext(arch: &str) -> Result<Ptx> {
    if !crate::tensor_core::arch_has_mma(arch) {
        return Err(Error::Compile(format!(
            "extra GEMV kernels require sm_80+ (cp.async); got {arch}"
        )));
    }
    nvrtc_present()?;
    compile_src(crate::cuda_ext::EXT_SOURCE, arch, "ternary_ext.cu")
}

/// Compile the tensor-core kernels to PTX. `arch` must be sm_80+ / compute_80+ (`mma.sync`).
pub fn compile_tensor_cores(arch: &str) -> Result<Ptx> {
    if !crate::tensor_core::arch_has_mma(arch) {
        return Err(Error::Compile(format!(
            "tensor-core path requires sm_80+ (mma.sync); got {arch}"
        )));
    }
    nvrtc_present()?;
    compile_src(TENSOR_CORE_SOURCE, arch, "tensor_core.cu")
}

/// Static facts about the device, used by the benchmarks.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub sm_count: u32,
    pub l2_bytes: usize,
    pub total_mem: usize,
    /// `2 * memory_clock * bus_width / 8` from driver attributes (GB/s). May be wrong for GDDR7/PAM3
    /// boards; benchmarks let you override it with `PEAK_GBPS`.
    pub attr_peak_gbps: f64,
    /// The NVRTC target the kernels were compiled for.
    pub ptx_arch: String,
    /// Max `batch * K` bytes the single-launch fused kernel accepts on this device.
    pub fused_smem_limit: usize,
}

/// ternary weights resident on the device.
pub struct GpuTernary {
    pub n: usize,
    pub k: usize,
    pub permuted: CudaSlice<u32>,
    pub natural: CudaSlice<u32>,
    pub row_scale: CudaSlice<f32>,
}

/// Quantized activations resident on the device.
pub struct GpuActs {
    pub batch: usize,
    pub k: usize,
    pub q: CudaSlice<i8>,
    pub s: CudaSlice<f32>,
    pub sum: CudaSlice<i32>,
}

/// Selectable GEMM/GEMV backend.
/// Which extra batch-1 kernel [`Gpu::gemv_ext`] launches (see `kernels/ternary_ext.cu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtKernel {
    /// 4-stage cp.async weight pipeline per warp; `rows_per_warp` in {1, 2, 4}.
    CpAsync { rows_per_warp: u32 },
    /// Split-K with int32 atomics + finalize; `splits = 0` picks [`crate::cuda_ext::splitk_factor`].
    SplitK { splits: u32 },
    /// T-MAC style table lookup (natural layout).
    Lut,
    /// 1 producer warp (cp.async) + 4 consumer warps, named-barrier ring.
    WarpSpecialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// Existing `dp4a` kernels (sm_61+). Permuted 2-bit layout, optional fused quantize.
    #[default]
    Dp4a,
    /// Tensor-core path (sm_80+): unpack 2-bit natural-layout weights into int8/fp16
    /// fragments and issue `mma.sync.aligned.m16n8k16`. Not fused; call [`Gpu::quantize`]
    /// first on the W2A8 path. `rows_per_warp` is ignored (the MMA M tile is 16).
    TensorCore,
}

/// Kernel launch tuning.
#[derive(Debug, Clone, Copy)]
pub struct Launch {
    /// Output rows per warp (1, 2 or 4). `None` = heuristic. Ignored for [`Backend::TensorCore`].
    pub rows_per_warp: Option<u32>,
    /// Warps per block (1..=8).
    pub warps: u32,
    /// Which kernel family to launch. Default: [`Backend::Dp4a`].
    pub backend: Backend,
}
impl Default for Launch {
    fn default() -> Self {
        Self {
            rows_per_warp: None,
            warps: 4,
            backend: Backend::Dp4a,
        }
    }
}

impl Launch {
    /// A tensor-core launch with `warps` warps per block (each warp owns 16 output rows).
    pub fn tensor_core(warps: u32) -> Self {
        Self {
            rows_per_warp: None,
            warps,
            backend: Backend::TensorCore,
        }
    }
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _tc_module: Option<Arc<CudaModule>>,
    _ext_module: Option<Arc<CudaModule>>,
    funcs: HashMap<String, CudaFunction>,
    pub info: DeviceInfo,
}

fn batch_tile(batch: usize) -> Result<usize> {
    BATCH_TILES
        .iter()
        .copied()
        .find(|&t| t >= batch)
        .filter(|_| batch > 0)
        .ok_or_else(|| {
            Error::Shape(format!(
                "batch={batch} must be in 1..=16 per launch (the kernels' largest batch tile); \
                 split larger batches into chunks of 16 (as looped::GpuBackend does) or use \
                 cpu::CpuTernary, which takes any batch"
            ))
        })
}

impl Gpu {
    /// Open device `ordinal`, compile the kernels for it with NVRTC and load them.
    pub fn new(ordinal: usize) -> Result<Self> {
        // SAFETY: only checks whether the shared library can be dlopen'ed.
        if !unsafe { sys::is_culib_present() } {
            return Err(Error::NoCuda("libcuda not found (no NVIDIA driver)".into()));
        }
        if CudaContext::device_count().unwrap_or(0) <= ordinal as i32 {
            return Err(Error::NoCuda(format!("no CUDA device #{ordinal}")));
        }
        let ctx = CudaContext::new(ordinal)?;
        // Single stream: skip cudarc's per-buffer event tracking (it would add work to every launch).
        // SAFETY: all work is issued on one stream, so no cross-stream synchronization is needed.
        unsafe { ctx.disable_event_tracking() };
        // A real (non-legacy) stream, so it can be captured into a CUDA graph, created at the
        // device's highest priority so latency-critical work is scheduled ahead of other streams.
        let (_least, greatest) = cudarc::driver::result::stream::get_priority_range()?;
        let stream = ctx.new_stream_with_priority(greatest)?;
        let (maj, min) = ctx.compute_capability()?;
        if (maj, min) < (6, 1) {
            return Err(Error::NoCuda(format!(
                "compute capability {maj}.{min} < 6.1 (dp4a required)"
            )));
        }
        // Try the exact arch first; older NVRTCs may not know it, so fall back to older PTX targets
        // (the driver JIT-compiles PTX forward to newer GPUs).
        let mut candidates = vec![format!("compute_{maj}{min}")];
        for a in [
            "compute_120",
            "compute_100",
            "compute_90",
            "compute_89",
            "compute_86",
            "compute_80",
            "compute_75",
            "compute_61",
        ] {
            let v: i32 = a[8..].parse().unwrap();
            if v <= maj * 10 + min && !candidates.iter().any(|c| c == a) {
                candidates.push(a.into());
            }
        }
        let mut last_err = None;
        let mut loaded = None;
        for arch in &candidates {
            match compile(arch) {
                Ok(ptx) => match ctx.load_module(ptx) {
                    Ok(m) => {
                        loaded = Some((m, arch.clone()));
                        break;
                    }
                    Err(e) => last_err = Some(Error::Driver(e)),
                },
                Err(e) => last_err = Some(e),
            }
        }
        let (module, ptx_arch) = match loaded {
            Some(x) => x,
            None => return Err(last_err.unwrap()),
        };
        let mut names: Vec<String> = vec![
            "quantize_i8".into(),
            "naive_i8".into(),
            "naive_f16".into(),
            "fill_ternary_codes".into(),
            "fill_f16".into(),
            "fill_f32".into(),
            "ternary_mlp".into(),
            "ternary_mlp_persistent".into(),
            "f32_to_f16".into(),
            "bias_act_f16".into(),
        ];
        for bt in BATCH_TILES {
            for r in ROWS_PER_WARP {
                for fam in ["ternary_i8", "ternary_i8_fused", "ternary_f16"] {
                    names.push(format!("{fam}_b{bt}_r{r}"));
                }
            }
        }
        let mut funcs = HashMap::new();
        for n in names {
            let f = module.load_function(&n)?;
            funcs.insert(n, f);
        }
        let mut tc_module = None;
        if (maj, min) >= (8, 0) {
            let tc_ptx = compile_tensor_cores(&ptx_arch)?;
            let tc = ctx.load_module(tc_ptx)?;
            for n in TENSOR_CORE_KERNELS {
                let f = tc.load_function(n)?;
                funcs.insert(n.to_string(), f);
            }
            tc_module = Some(tc);
        }
        let mut ext_module = None;
        if (maj, min) >= (8, 0) {
            // Optional: an NVRTC too old for these kernels must not take the main path down.
            if let Ok(ptx) = compile_ext(&ptx_arch) {
                let ext = ctx.load_module(ptx)?;
                for n in crate::cuda_ext::EXT_KERNELS {
                    funcs.insert(n.to_string(), ext.load_function(n)?);
                }
                ext_module = Some(ext);
            }
        }
        let attr = |a| ctx.attribute(a).map(|v| v as i64);
        // Opt the fused kernels in to the large dynamic shared-memory carve-out so the single-launch
        // path covers bigger batch * K. Falls back to the 48 KiB default if the driver refuses.
        let optin =
            attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
                .unwrap_or(0) as usize;
        let mut fused_smem_limit = FUSED_SMEM_LIMIT;
        if optin > FUSED_SMEM_LIMIT + FUSED_STATIC_SMEM {
            let want = optin - FUSED_STATIC_SMEM;
            let ok = funcs
                .iter()
                .filter(|(n, _)| n.starts_with("ternary_i8_fused_"))
                .all(|(_, f)| {
                    f.set_attribute(
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        want as i32,
                    )
                    .is_ok()
                });
            if ok {
                fused_smem_limit = want;
            }
        }
        let mem_khz = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?;
        let bus_bits = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let info = DeviceInfo {
            name: ctx.name()?,
            compute_capability: (maj, min),
            sm_count: attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
                as u32,
            l2_bytes: attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as usize,
            total_mem: ctx.total_mem()?,
            attr_peak_gbps: 2.0 * mem_khz as f64 * 1e3 * (bus_bits as f64 / 8.0) / 1e9,
            ptx_arch,
            fused_smem_limit,
        };
        Ok(Self {
            ctx,
            stream,
            _module: module,
            _tc_module: tc_module,
            _ext_module: ext_module,
            funcs,
            info,
        })
    }

    pub(crate) fn func(&self, name: &str) -> &CudaFunction {
        &self.funcs[name]
    }

    /// Registers per thread the driver reports for a loaded kernel (e.g. `"ternary_i8_b16_r4"`).
    pub fn kernel_regs(&self, name: &str) -> Option<u32> {
        self.funcs.get(name)?.num_regs().ok().map(|r| r as u32)
    }

    /// True when this device compiled and loaded the sm_80+ `mma.sync` kernels.
    pub fn supports_tensor_cores(&self) -> bool {
        self._tc_module.is_some()
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    pub fn upload_f16(&self, x: &[f16]) -> Result<CudaSlice<f16>> {
        Ok(self.stream.clone_htod(x)?)
    }
    pub fn download<T: cudarc::driver::DeviceRepr + Default + Clone>(
        &self,
        x: &CudaSlice<T>,
    ) -> Result<Vec<T>> {
        Ok(self.stream.clone_dtoh(x)?)
    }
    pub fn alloc_f16(&self, len: usize) -> Result<CudaSlice<f16>> {
        Ok(self.stream.alloc_zeros::<f16>(len)?)
    }

    pub fn upload_weights(&self, w: &PackedTernary) -> Result<GpuTernary> {
        Ok(GpuTernary {
            n: w.n,
            k: w.k,
            permuted: self.stream.clone_htod(&w.permuted)?,
            natural: self.stream.clone_htod(&w.natural)?,
            row_scale: self.stream.clone_htod(&w.row_scale)?,
        })
    }

    /// Allocate device ternary weights of shape `[n][k]` filled with pseudo-random codes (benchmarks).
    pub fn random_weights(&self, n: usize, k: usize, seed: u32) -> Result<GpuTernary> {
        if !k.is_multiple_of(K_ALIGN) {
            return shape_err(format!("K={k} must be a multiple of {K_ALIGN}"));
        }
        let words = n * k / 16;
        let mut permuted = self.stream.alloc_zeros::<u32>(words)?;
        let mut row_scale = self.stream.alloc_zeros::<f32>(n)?;
        self.fill_codes(&mut permuted, seed)?;
        self.fill_f32(&mut row_scale, seed ^ 0x51)?;
        // The natural layout is only used by the naive kernel; same random bits are fine for timing.
        let natural = self.stream.clone_dtod(&permuted)?;
        Ok(GpuTernary {
            n,
            k,
            permuted,
            natural,
            row_scale,
        })
    }

    pub fn alloc_acts(&self, batch: usize, k: usize) -> Result<GpuActs> {
        Ok(GpuActs {
            batch,
            k,
            q: self.stream.alloc_zeros::<i8>(batch * k)?,
            s: self.stream.alloc_zeros::<f32>(batch)?,
            sum: self.stream.alloc_zeros::<i32>(batch)?,
        })
    }

    fn grid_fill(&self, n: usize) -> LaunchConfig {
        let blocks = n.div_ceil(256).clamp(1, self.info.sm_count as usize * 32) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    pub fn fill_codes(&self, p: &mut CudaSlice<u32>, seed: u32) -> Result<()> {
        let n = p.len() as u64;
        let cfg = self.grid_fill(p.len());
        let mut b = self.stream.launch_builder(self.func("fill_ternary_codes"));
        b.arg(p).arg(&n).arg(&seed);
        // SAFETY: kernel writes exactly `n` elements of `p`.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }
    pub fn fill_f16(&self, p: &mut CudaSlice<f16>, seed: u32) -> Result<()> {
        let n = p.len() as u64;
        let cfg = self.grid_fill(p.len());
        let mut b = self.stream.launch_builder(self.func("fill_f16"));
        b.arg(p).arg(&n).arg(&seed);
        // SAFETY: kernel writes exactly `n` elements of `p`.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }
    pub fn fill_f32(&self, p: &mut CudaSlice<f32>, seed: u32) -> Result<()> {
        let n = p.len() as u64;
        let cfg = self.grid_fill(p.len());
        let mut b = self.stream.launch_builder(self.func("fill_f32"));
        b.arg(p).arg(&n).arg(&seed);
        // SAFETY: kernel writes exactly `n` elements of `p`.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Heuristic: the most rows per warp that still gives >= 2 waves of blocks.
    pub fn pick_rows_per_warp(&self, n: usize, warps: u32) -> u32 {
        for r in [4u32, 2, 1] {
            let blocks = n.div_ceil((warps * r) as usize);
            if blocks >= 2 * self.info.sm_count as usize * 2 {
                return r;
            }
        }
        1
    }

    fn row_launch(&self, n: usize, l: Launch, smem: u32) -> Result<(LaunchConfig, u32)> {
        if !(1..=8).contains(&l.warps) {
            return shape_err("warps must be 1..=8");
        }
        let r = l
            .rows_per_warp
            .unwrap_or_else(|| self.pick_rows_per_warp(n, l.warps));
        if !ROWS_PER_WARP.contains(&r) {
            return shape_err("rows_per_warp must be 1, 2 or 4");
        }
        let rows_per_block = (l.warps * r) as usize;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(rows_per_block) as u32, 1, 1),
            block_dim: (32 * l.warps, 1, 1),
            shared_mem_bytes: smem,
        };
        Ok((cfg, r))
    }

    fn check_common(
        w: &GpuTernary,
        batch: usize,
        residual: Option<&CudaSlice<f16>>,
        out: &CudaSlice<f16>,
    ) -> Result<()> {
        if !w.k.is_multiple_of(K_ALIGN) {
            return shape_err(format!(
                "K={} must be a multiple of {K_ALIGN} (one 16-byte load = 64 weights per lane); \
                 zero-pad K (zero weights and zero activations leave the result unchanged)",
                w.k
            ));
        }
        if w.permuted.len() != w.n * w.k / 16
            || w.natural.len() != w.n * w.k / 16
            || w.row_scale.len() != w.n
        {
            return shape_err("inconsistent GpuTernary");
        }
        if out.len() < batch * w.n {
            return shape_err(format!(
                "out has {} elements, need {}",
                out.len(),
                batch * w.n
            ));
        }
        if let Some(r) = residual {
            if r.len() < batch * w.n {
                return shape_err("residual too small");
            }
        }
        if w.n * w.k >= (1usize << 31) || batch * w.n >= (1usize << 31) {
            return shape_err("matrix too large for 32-bit indexing");
        }
        Ok(())
    }

    /// Quantize `[batch][k]` fp16 activations to int8 (optionally RMSNorm'ed by `gamma`).
    pub fn quantize(
        &self,
        x: &CudaSlice<f16>,
        gamma: Option<&CudaSlice<f16>>,
        eps: f32,
        acts: &mut GpuActs,
    ) -> Result<()> {
        let (batch, k) = (acts.batch, acts.k);
        if k % K_ALIGN != 0
            || x.len() < batch * k
            || acts.q.len() < batch * k
            || acts.s.len() < batch
            || acts.sum.len() < batch
        {
            return shape_err("quantize: bad buffer sizes");
        }
        if gamma.is_some_and(|g| g.len() < k) {
            return shape_err("gamma too small");
        }
        let null = 0u64;
        let ki = k as i32;
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(self.func("quantize_i8"));
        b.arg(x);
        match gamma {
            Some(g) => b.arg(g),
            None => b.arg(&null),
        };
        b.arg(&eps)
            .arg(&ki)
            .arg(&mut acts.q)
            .arg(&mut acts.s)
            .arg(&mut acts.sum);
        // SAFETY: sizes validated above; kernel touches batch*k inputs/outputs.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// W2A8 GEMV/GEMM on pre-quantized activations: `out[b][n] = W·q[b] / s[b] * scale[n] (+ residual)`.
    pub fn gemm_i8(
        &self,
        w: &GpuTernary,
        a: &GpuActs,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
        l: Launch,
    ) -> Result<()> {
        Self::check_common(w, a.batch, residual, out)?;
        if a.k != w.k || a.q.len() < a.batch * a.k || a.s.len() < a.batch || a.sum.len() < a.batch {
            return shape_err("activations do not match weights");
        }
        if l.backend == Backend::TensorCore {
            return self.launch_tc_i8(w, a, residual, out, l);
        }
        let bt = batch_tile(a.batch)?;
        let (cfg, r) = self.row_launch(w.n, l, 0)?;
        let f = self.func(&format!("ternary_i8_b{bt}_r{r}"));
        let null = 0u64;
        let (n, k, batch) = (w.n as i32, w.k as i32, a.batch as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&w.permuted)
            .arg(&w.row_scale)
            .arg(&a.q)
            .arg(&a.s)
            .arg(&a.sum);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&batch);
        // SAFETY: shapes validated above.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Single-launch W2A8: activation quantization (+ optional RMSNorm) fused into the prologue.
    /// Requires `batch * K <= self.info.fused_smem_limit` (device dependent, >= [`FUSED_SMEM_LIMIT`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_i8_fused(
        &self,
        w: &GpuTernary,
        x: &CudaSlice<f16>,
        batch: usize,
        gamma: Option<&CudaSlice<f16>>,
        eps: f32,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
        l: Launch,
    ) -> Result<()> {
        Self::check_common(w, batch, residual, out)?;
        if x.len() < batch * w.k || gamma.is_some_and(|g| g.len() < w.k) {
            return shape_err("x / gamma too small");
        }
        if l.backend == Backend::TensorCore {
            return shape_err(
                "fused tensor-core kernel is not implemented; use quantize + gemm_i8 with Backend::TensorCore",
            );
        }
        let smem = batch * w.k;
        if smem > self.info.fused_smem_limit {
            return shape_err(format!(
                "fused kernel needs batch*K={smem} B of shared memory > {} on this device; use quantize + gemm_i8",
                self.info.fused_smem_limit
            ));
        }
        let bt = batch_tile(batch)?;
        let (cfg, r) = self.row_launch(w.n, l, smem as u32)?;
        let f = self.func(&format!("ternary_i8_fused_b{bt}_r{r}"));
        let null = 0u64;
        let (n, k, bi) = (w.n as i32, w.k as i32, batch as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&w.permuted).arg(&w.row_scale).arg(x);
        match gamma {
            Some(g) => b.arg(g),
            None => b.arg(&null),
        };
        b.arg(&eps);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&bi);
        // SAFETY: shapes validated above.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// W2A16: fp16 activations, fp32 accumulation, `out = W·x * scale[n] (+ residual)`.
    pub fn gemm_f16(
        &self,
        w: &GpuTernary,
        x: &CudaSlice<f16>,
        batch: usize,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
        l: Launch,
    ) -> Result<()> {
        Self::check_common(w, batch, residual, out)?;
        if x.len() < batch * w.k {
            return shape_err("x too small");
        }
        if l.backend == Backend::TensorCore {
            return self.launch_tc_f16(w, x, batch, residual, out, l);
        }
        let bt = batch_tile(batch)?;
        let (cfg, r) = self.row_launch(w.n, l, 0)?;
        let f = self.func(&format!("ternary_f16_b{bt}_r{r}"));
        let null = 0u64;
        let (n, k, bi) = (w.n as i32, w.k as i32, batch as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&w.permuted).arg(&w.row_scale).arg(x);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&bi);
        // SAFETY: shapes validated above.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn tc_launch(&self, n: usize, l: Launch) -> Result<LaunchConfig> {
        if l.backend != Backend::TensorCore {
            return shape_err("tc_launch requires Backend::TensorCore");
        }
        if !(1..=8).contains(&l.warps) {
            return shape_err("warps must be 1..=8");
        }
        if !self.supports_tensor_cores() {
            return Err(Error::NoCuda(
                "tensor-core path requires sm_80+ (mma.sync) and compiled ternary_tc_* kernels"
                    .into(),
            ));
        }
        let rows_per_block = l.warps as usize * 16;
        Ok(LaunchConfig {
            grid_dim: (n.div_ceil(rows_per_block) as u32, 1, 1),
            block_dim: (32 * l.warps, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    /// Batch-1 W2A8 GEMV with one of the extra kernels ([`ExtKernel`]), fused activation `act`
    /// applied before the residual. Bit-identical to [`crate::cuda_ext::reference_gemv`].
    /// Needs sm_80+. `scratch` (split-K partials) must hold `w.n` i32 when `kind` is `SplitK`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_ext(
        &self,
        kind: ExtKernel,
        w: &GpuTernary,
        a: &GpuActs,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
        act: crate::mlp::Act,
        scratch: Option<&mut CudaSlice<i32>>,
    ) -> Result<()> {
        use crate::cuda_ext as x;
        Self::check_common(w, 1, residual, out)?;
        if a.batch != 1 || a.k != w.k {
            return shape_err("gemv_ext takes batch-1 activations matching K");
        }
        if self._ext_module.is_none() {
            return Err(Error::Compile("extra GEMV kernels need sm_80+".into()));
        }
        let (n, k, act) = (w.n as i32, w.k as i32, act as i32);
        let null = 0u64;
        macro_rules! res {
            ($b:expr) => {
                match residual {
                    Some(r) => $b.arg(r),
                    None => $b.arg(&null),
                }
            };
        }
        match kind {
            ExtKernel::CpAsync { rows_per_warp } => {
                let r = rows_per_warp as usize;
                if ![1, 2, 4].contains(&r) {
                    return shape_err("cp.async rows_per_warp must be 1, 2 or 4");
                }
                let smem = x::cpasync_smem_bytes(w.k, r);
                if smem > 48 * 1024 {
                    return shape_err(format!("K={} needs {smem} B of smem (> 48 KiB)", w.k));
                }
                let rows_per_block = x::CP_WARPS * r;
                let cfg = LaunchConfig {
                    grid_dim: (w.n.div_ceil(rows_per_block) as u32, 1, 1),
                    block_dim: ((x::CP_WARPS * 32) as u32, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                let mut b = self
                    .stream
                    .launch_builder(self.func(&format!("tgx_gemv_cpasync_r{r}")));
                b.arg(&w.permuted)
                    .arg(&w.row_scale)
                    .arg(&a.q)
                    .arg(&a.s)
                    .arg(&a.sum);
                res!(b);
                b.arg(out).arg(&n).arg(&k).arg(&act);
                // SAFETY: shapes validated above; smem sized by cpasync_smem_bytes.
                unsafe { b.launch(cfg) }?;
            }
            ExtKernel::SplitK { splits } => {
                let part = scratch.ok_or_else(|| Error::Shape("split-K needs scratch".into()))?;
                if part.len() < w.n {
                    return shape_err("split-K scratch must hold n i32");
                }
                let splits = if splits == 0 {
                    x::splitk_factor(self.info.sm_count as usize, w.n, w.k)
                } else {
                    splits as usize
                };
                self.stream.memset_zeros(part)?;
                let cfg = LaunchConfig {
                    grid_dim: (w.n.div_ceil(16) as u32, splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(self.func("tgx_gemv_splitk_r2"));
                b.arg(&w.permuted).arg(&a.q).arg(&mut *part).arg(&n).arg(&k);
                // SAFETY: shapes validated above.
                unsafe { b.launch(cfg) }?;
                let cfg = LaunchConfig {
                    grid_dim: (w.n.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(self.func("tgx_splitk_finalize"));
                b.arg(&*part).arg(&w.row_scale).arg(&a.s).arg(&a.sum);
                res!(b);
                b.arg(out).arg(&n).arg(&act);
                // SAFETY: shapes validated above.
                unsafe { b.launch(cfg) }?;
            }
            ExtKernel::Lut => {
                let cfg = LaunchConfig {
                    grid_dim: (w.n.div_ceil(x::LUT_ROWS) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(self.func("tgx_gemv_lut"));
                b.arg(&w.natural)
                    .arg(&w.row_scale)
                    .arg(&a.q)
                    .arg(&a.s)
                    .arg(&a.sum);
                res!(b);
                b.arg(out).arg(&n).arg(&k).arg(&act);
                // SAFETY: shapes validated above.
                unsafe { b.launch(cfg) }?;
            }
            ExtKernel::WarpSpecialized => {
                let smem = x::warpspec_smem_bytes(w.k);
                if smem > 48 * 1024 {
                    return shape_err(format!("K={} needs {smem} B of smem (> 48 KiB)", w.k));
                }
                let cfg = LaunchConfig {
                    grid_dim: (w.n.div_ceil(x::WS_ROWS) as u32, 1, 1),
                    block_dim: (x::WS_THREADS as u32, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                let mut b = self.stream.launch_builder(self.func("tgx_gemv_warpspec"));
                b.arg(&w.permuted)
                    .arg(&w.row_scale)
                    .arg(&a.q)
                    .arg(&a.s)
                    .arg(&a.sum);
                res!(b);
                b.arg(out).arg(&n).arg(&k).arg(&act);
                // SAFETY: shapes validated above; smem sized by warpspec_smem_bytes.
                unsafe { b.launch(cfg) }?;
            }
        }
        Ok(())
    }

    fn launch_tc_i8(
        &self,
        w: &GpuTernary,
        a: &GpuActs,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
        l: Launch,
    ) -> Result<()> {
        let cfg = self.tc_launch(w.n, l)?;
        let null = 0u64;
        let (n, k, batch) = (w.n as i32, w.k as i32, a.batch as i32);
        let mut b = self.stream.launch_builder(self.func("ternary_tc_i8"));
        b.arg(&w.natural).arg(&w.row_scale).arg(&a.q).arg(&a.s);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&batch);
        // SAFETY: shapes validated by the caller; natural layout length checked in check_common.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn launch_tc_f16(
        &self,
        w: &GpuTernary,
        x: &CudaSlice<f16>,
        batch: usize,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
        l: Launch,
    ) -> Result<()> {
        let cfg = self.tc_launch(w.n, l)?;
        let null = 0u64;
        let (n, k, bi) = (w.n as i32, w.k as i32, batch as i32);
        let mut b = self.stream.launch_builder(self.func("ternary_tc_f16"));
        b.arg(&w.natural).arg(&w.row_scale).arg(x);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&bi);
        // SAFETY: shapes validated by the caller.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Naive baseline (one thread per output, natural layout, scalar decode). Same output bits as
    /// [`Gpu::gemm_i8`].
    pub fn naive_i8(
        &self,
        w: &GpuTernary,
        a: &GpuActs,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
    ) -> Result<()> {
        Self::check_common(w, a.batch, residual, out)?;
        if a.k != w.k || a.q.len() < a.batch * a.k || a.s.len() < a.batch {
            return shape_err("activations do not match weights");
        }
        let total = (w.n * a.batch) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let null = 0u64;
        let (n, k, bi) = (w.n as i32, w.k as i32, a.batch as i32);
        let mut b = self.stream.launch_builder(self.func("naive_i8"));
        b.arg(&w.natural).arg(&w.row_scale).arg(&a.q).arg(&a.s);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&bi);
        // SAFETY: shapes validated above.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Naive W2A16 baseline.
    pub fn naive_f16(
        &self,
        w: &GpuTernary,
        x: &CudaSlice<f16>,
        batch: usize,
        residual: Option<&CudaSlice<f16>>,
        out: &mut CudaSlice<f16>,
    ) -> Result<()> {
        Self::check_common(w, batch, residual, out)?;
        if x.len() < batch * w.k {
            return shape_err("x too small");
        }
        let total = (w.n * batch) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let null = 0u64;
        let (n, k, bi) = (w.n as i32, w.k as i32, batch as i32);
        let mut b = self.stream.launch_builder(self.func("naive_f16"));
        b.arg(&w.natural).arg(&w.row_scale).arg(x);
        match residual {
            Some(res) => b.arg(res),
            None => b.arg(&null),
        };
        b.arg(out).arg(&n).arg(&k).arg(&bi);
        // SAFETY: shapes validated above.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }
}

/// A captured CUDA graph: a fixed sequence of launches replayed with one driver call.
pub struct Graph {
    graph: cudarc::driver::CudaGraph,
}
impl Graph {
    /// Replay the captured work on the [`Gpu`]'s stream.
    pub fn launch(&self) -> Result<()> {
        Ok(self.graph.launch()?)
    }
}

impl Gpu {
    /// Capture every launch `f` issues on this GPU's stream into a [`Graph`] (nothing runs during
    /// capture). `f` must only launch work: no allocation, copies to host or synchronization.
    /// Buffer contents can change between replays; buffer *addresses* are baked into the graph.
    pub fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<Graph> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let body = f();
        // Always end the capture, even if `f` failed, so the stream is usable again.
        let ended = self.stream.end_capture(sys::CUgraphInstantiate_flags(0));
        body?;
        match ended? {
            Some(graph) => {
                graph.upload()?;
                Ok(Graph { graph })
            }
            None => shape_err("captured an empty graph"),
        }
    }
}

/// cuBLAS FP16 baseline (`cublasGemmEx`, fp16 in/out, fp32 compute) on the same stream.
pub struct Cublas {
    blas: CudaBlas,
}
impl Cublas {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        // SAFETY: only checks whether the shared library can be dlopen'ed.
        if !unsafe { cublas_sys::is_culib_present() } {
            return Err(Error::NoCuda("libcublas not found".into()));
        }
        Ok(Self {
            blas: CudaBlas::new(gpu.stream.clone())?,
        })
    }

    /// `out[b][n] = sum_k w[n][k] * x[b][k]` with `w` row-major `[n][k]` fp16.
    pub fn gemm(
        &self,
        w: &CudaSlice<f16>,
        x: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<()> {
        if w.len() < n * k || x.len() < batch * k || out.len() < batch * n {
            return shape_err("cublas: bad sizes");
        }
        // Column-major view: C[n x batch] = op(A)[n x k] * B[k x batch], A = w stored as k x n (lda = k).
        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: batch as i32,
            k: k as i32,
            alpha: f16::ONE,
            lda: k as i32,
            ldb: k as i32,
            beta: f16::ZERO,
            ldc: n as i32,
        };
        // SAFETY: sizes validated above.
        unsafe { self.blas.gemm(cfg, w, x, out) }?;
        Ok(())
    }
}

/// Simple CUDA-event timer: median microseconds per call over `reps` batches of `iters` calls.
pub fn time_us(
    gpu: &Gpu,
    iters: usize,
    reps: usize,
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    let start = gpu
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = gpu
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(reps);
    let mut call = 0usize;
    for _ in 0..reps {
        start.record(&gpu.stream)?;
        for _ in 0..iters {
            f(call)?;
            call += 1;
        }
        end.record(&gpu.stream)?;
        end.synchronize()?;
        samples.push(start.elapsed_ms(&end)? as f64 * 1e3 / iters as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}
