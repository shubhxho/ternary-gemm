//! Latency-first GPU inference for small ternary models ([`crate::mlp::TernaryMlp`]):
//! a feature vector goes in, a signal comes out, and the hot path never allocates.
//!
//! * [`SignalEngine`]: preallocated device buffers + pinned host staging buffers. One request is
//!   `H2D copy -> one fused whole-MLP kernel -> D2H copy`, captured once as a CUDA graph and
//!   replayed with a single driver call on the [`Gpu`]'s highest-priority stream. Completion is
//!   detected by spinning on `cuStreamQuery` (or by a blocking sync, see [`Wait`]).
//! * [`PersistentEngine`] (**experimental**): one resident block spins on a doorbell in mapped
//!   pinned host memory and reads/writes the feature and signal vectors zero-copy, so a request
//!   costs no launch and no memcpy call at all. It permanently occupies one SM while running.
//! * [`CublasMlp`]: the same model with dense fp16 weights on cuBLAS, captured the same way, as the
//!   baseline.
//!
//! All ternary paths are bit-identical to [`crate::mlp::TernaryMlp::forward_reference`].

use crate::gpu::{Cublas, Error, Gpu, Graph, Result};
use crate::mlp::{TernaryMlp, MLP_MAX_DIM};
use cudarc::driver::{
    result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use half::f16;
use std::ffi::c_void;
use std::sync::atomic::{fence, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Page-locked host buffer (optionally mapped into the device address space).
pub struct PinnedBuf<T: Copy + Default> {
    ptr: *mut T,
    len: usize,
    dev: u64,
    ctx: Arc<CudaContext>,
}
// SAFETY: plain host memory owned by this value.
unsafe impl<T: Copy + Default + Send> Send for PinnedBuf<T> {}

impl<T: Copy + Default> PinnedBuf<T> {
    /// `mapped = true` also maps it for zero-copy device access ([`PinnedBuf::device_ptr`]).
    pub fn new(ctx: &Arc<CudaContext>, len: usize, mapped: bool) -> Result<Self> {
        ctx.bind_to_thread()?;
        let bytes = (len.max(1) * std::mem::size_of::<T>()).next_multiple_of(64);
        let flags = if mapped {
            sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE
        } else {
            0
        };
        // SAFETY: fresh allocation; zero-initialized below before any read.
        let ptr = unsafe { result::malloc_host(bytes, flags)? } as *mut T;
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        let mut dev = 0u64;
        if mapped {
            // SAFETY: ptr was allocated with DEVICEMAP above.
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, ptr as *mut c_void, 0).result()? };
        }
        Ok(Self {
            ptr,
            len,
            dev,
            ctx: ctx.clone(),
        })
    }
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: valid for len elements for the life of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: valid for len elements, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
    /// Device address of a mapped buffer (0 if not mapped).
    pub fn device_ptr(&self) -> u64 {
        self.dev
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl<T: Copy + Default> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: allocated by malloc_host in new().
        let _ = unsafe { result::free_host(self.ptr as *mut c_void) };
    }
}

/// How the host waits for a request to finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wait {
    /// Busy-poll `cuStreamQuery` (lowest latency, burns a CPU core).
    Spin,
    /// `cuStreamSynchronize` (behaviour depends on the context's scheduling flags).
    Block,
}

/// How work is submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submit {
    /// Replay the captured CUDA graph (one driver call).
    Graph,
    /// Issue copy, kernel, copy one by one.
    Direct,
}

fn spin_wait(stream: &CudaStream) -> Result<()> {
    loop {
        // SAFETY: valid stream handle.
        match unsafe { sys::cuStreamQuery(stream.cu_stream()) } {
            sys::CUresult::CUDA_SUCCESS => return Ok(()),
            sys::CUresult::CUDA_ERROR_NOT_READY => std::hint::spin_loop(),
            e => return Err(Error::Driver(e.result().unwrap_err())),
        }
    }
}

fn wait(stream: &CudaStream, w: Wait) -> Result<()> {
    match w {
        Wait::Spin => spin_wait(stream),
        Wait::Block => Ok(stream.synchronize()?),
    }
}

fn dptr<T>(s: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    let (p, _g) = s.device_ptr(stream);
    p
}

unsafe fn htod_async(dst: u64, src: *const c_void, bytes: usize, s: &CudaStream) -> Result<()> {
    Ok(sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, s.cu_stream()).result()?)
}
unsafe fn dtoh_async(dst: *mut c_void, src: u64, bytes: usize, s: &CudaStream) -> Result<()> {
    Ok(sys::cuMemcpyDtoHAsync_v2(dst, src, bytes, s.cu_stream()).result()?)
}

/// A [`TernaryMlp`] resident on the device in the kernel's k-major layout.
pub struct GpuMlp {
    pub layers: i32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub max_dim: usize,
    desc: CudaSlice<i32>,
    w: CudaSlice<u32>,
    sb: CudaSlice<f32>,
}

impl GpuMlp {
    pub fn upload(gpu: &Gpu, m: &TernaryMlp) -> Result<Self> {
        m.validate().map_err(Error::Shape)?;
        let mut desc = Vec::new();
        let mut w = Vec::new();
        let mut sb = Vec::new();
        for l in &m.layers {
            desc.extend_from_slice(&[l.n as i32, l.k as i32, l.act as i32, 0]);
            w.extend(l.kmajor_words());
            sb.extend_from_slice(&l.scale);
            sb.extend_from_slice(&l.bias);
        }
        let s = &gpu.stream;
        let g = Self {
            layers: m.layers.len() as i32,
            in_dim: m.in_dim(),
            out_dim: m.out_dim(),
            max_dim: m.max_dim(),
            desc: s.clone_htod(&desc)?,
            w: s.clone_htod(&w)?,
            sb: s.clone_htod(&sb)?,
        };
        s.synchronize()?;
        Ok(g)
    }
    fn smem_bytes(&self) -> u32 {
        (8 * self.max_dim + 4 * self.max_dim.div_ceil(4)) as u32
    }
}

/// Threads per block for the fused MLP kernel: enough to cover the widest layer, 128..=1024.
pub fn default_threads(m: &TernaryMlp) -> u32 {
    (m.layers
        .iter()
        .map(|l| l.n)
        .max()
        .unwrap_or(1)
        .next_multiple_of(32) as u32)
        .clamp(128, 512)
}

/// Graph-replayed, allocation-free GPU inference of a [`TernaryMlp`].
pub struct SignalEngine<'g> {
    gpu: &'g Gpu,
    pub model: GpuMlp,
    pub batch: usize,
    pub threads: u32,
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
    h_in: PinnedBuf<f32>,
    h_out: PinnedBuf<f32>,
    graph: Option<Graph>,
    /// Why graph capture failed, if it did (then [`Submit::Graph`] falls back to direct).
    pub graph_error: Option<String>,
}

impl<'g> SignalEngine<'g> {
    /// `batch` independent samples per request (1 for a single signal); `threads` = `None` uses
    /// [`default_threads`].
    pub fn new(gpu: &'g Gpu, m: &TernaryMlp, batch: usize, threads: Option<u32>) -> Result<Self> {
        if batch == 0 || batch > 65535 {
            return Err(Error::Shape(format!("batch={batch} must be in 1..=65535")));
        }
        if m.max_dim() > MLP_MAX_DIM {
            return Err(Error::Shape(format!(
                "max width {} > {MLP_MAX_DIM}",
                m.max_dim()
            )));
        }
        let threads = threads.unwrap_or_else(|| default_threads(m));
        if threads == 0 || !threads.is_multiple_of(32) || threads > 1024 {
            return Err(Error::Shape(format!(
                "threads={threads} must be a multiple of 32 in 32..=1024"
            )));
        }
        let model = GpuMlp::upload(gpu, m)?;
        let x = gpu.stream.alloc_zeros::<f32>(batch * model.in_dim)?;
        let y = gpu.stream.alloc_zeros::<f32>(batch * model.out_dim)?;
        let h_in = PinnedBuf::new(&gpu.ctx, batch * model.in_dim, false)?;
        let h_out = PinnedBuf::new(&gpu.ctx, batch * model.out_dim, false)?;
        gpu.stream.synchronize()?;
        let mut e = Self {
            gpu,
            model,
            batch,
            threads,
            x,
            y,
            h_in,
            h_out,
            graph: None,
            graph_error: None,
        };
        let captured = {
            let er = &mut e;
            let gpu = er.gpu;
            gpu.capture(|| er.enqueue())
        };
        match captured {
            Ok(g) => e.graph = Some(g),
            Err(err) => e.graph_error = Some(err.to_string()),
        }
        Ok(e)
    }

    /// Pinned staging buffer for the next request's features, `[batch][in_dim]`.
    pub fn input_mut(&mut self) -> &mut [f32] {
        self.h_in.as_mut_slice()
    }
    /// Pinned buffer with the last request's signals, `[batch][out_dim]`.
    pub fn output(&self) -> &[f32] {
        self.h_out.as_slice()
    }
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    fn enqueue(&mut self) -> Result<()> {
        let s = &self.gpu.stream;
        let (xd, yd) = (dptr(&self.x, s), dptr(&self.y, s));
        let in_b = self.h_in.len() * 4;
        let out_b = self.h_out.len() * 4;
        // SAFETY: sizes match the allocations made in new().
        unsafe { htod_async(xd, self.h_in.as_ptr() as *const c_void, in_b, s)? };
        let m = &self.model;
        let (ind, outd, maxd) = (m.in_dim as i32, m.out_dim as i32, m.max_dim as i32);
        let f = self.gpu.func("ternary_mlp");
        let mut b = s.launch_builder(f);
        b.arg(&m.desc)
            .arg(&m.layers)
            .arg(&m.w)
            .arg(&m.sb)
            .arg(&self.x)
            .arg(&mut self.y);
        b.arg(&ind).arg(&outd).arg(&maxd);
        let cfg = LaunchConfig {
            grid_dim: (self.batch as u32, 1, 1),
            block_dim: (self.threads, 1, 1),
            shared_mem_bytes: m.smem_bytes(),
        };
        // SAFETY: argument list matches the kernel signature; all buffers sized in new().
        unsafe { b.launch(cfg)? };
        unsafe { dtoh_async(self.h_out.as_mut_ptr() as *mut c_void, yd, out_b, s)? };
        Ok(())
    }

    /// Run one request on whatever is in [`Self::input_mut`]; results land in [`Self::output`].
    pub fn run(&mut self, submit: Submit, w: Wait) -> Result<()> {
        match (&self.graph, submit) {
            (Some(g), Submit::Graph) => g.launch()?,
            _ => self.enqueue()?,
        }
        wait(&self.gpu.stream, w)
    }

    /// Copy `x` in, run (graph + spin), copy the signal out. No allocation.
    pub fn infer(&mut self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.input_mut().copy_from_slice(x);
        self.run(Submit::Graph, Wait::Spin)?;
        out.copy_from_slice(self.output());
        Ok(())
    }
}

const STOP: u32 = 0xffff_ffff;

/// **Experimental** persistent-kernel engine (batch 1). See the module docs.
///
/// While it runs, one block is resident on the GPU, spinning. Do not call
/// `CudaContext::synchronize()` (it would wait forever); drop or [`PersistentEngine::stop`] it first.
pub struct PersistentEngine<'g> {
    gpu: &'g Gpu,
    _model: GpuMlp,
    stream: Arc<CudaStream>,
    ctrl: PinnedBuf<u32>,
    x: PinnedBuf<f32>,
    y: PinnedBuf<f32>,
    seq: u32,
    running: bool,
    /// Give up waiting for the device after this long.
    pub timeout: Duration,
}

impl<'g> PersistentEngine<'g> {
    pub fn start(gpu: &'g Gpu, m: &TernaryMlp, threads: Option<u32>) -> Result<Self> {
        let threads = threads.unwrap_or_else(|| default_threads(m));
        if threads == 0 || !threads.is_multiple_of(32) || threads > 1024 {
            return Err(Error::Shape(format!(
                "threads={threads} must be a multiple of 32 in 32..=1024"
            )));
        }
        let model = GpuMlp::upload(gpu, m)?;
        let (_least, greatest) = result::stream::get_priority_range()?;
        let stream = gpu.ctx.new_stream_with_priority(greatest)?;
        let ctrl = PinnedBuf::<u32>::new(&gpu.ctx, 16, true)?;
        let x = PinnedBuf::<f32>::new(&gpu.ctx, model.in_dim, true)?;
        let y = PinnedBuf::<f32>::new(&gpu.ctx, model.out_dim, true)?;
        let (ind, outd, maxd) = (
            model.in_dim as i32,
            model.out_dim as i32,
            model.max_dim as i32,
        );
        let (cp, xp, yp) = (ctrl.device_ptr(), x.device_ptr(), y.device_ptr());
        {
            let f = gpu.func("ternary_mlp_persistent");
            let mut b = stream.launch_builder(f);
            b.arg(&model.desc)
                .arg(&model.layers)
                .arg(&model.w)
                .arg(&model.sb);
            b.arg(&cp).arg(&xp).arg(&yp).arg(&ind).arg(&outd).arg(&maxd);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: model.smem_bytes(),
            };
            // SAFETY: argument list matches the kernel; mapped buffers outlive the kernel (stop()).
            unsafe { b.launch(cfg)? };
        }
        Ok(Self {
            gpu,
            _model: model,
            stream,
            ctrl,
            x,
            y,
            seq: 0,
            running: true,
            timeout: Duration::from_secs(2),
        })
    }

    /// One request: write features, ring the doorbell, spin until the device reports completion.
    pub fn infer(&mut self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if !self.running {
            return Err(Error::Shape("persistent engine is stopped".into()));
        }
        let xs = self.x.as_mut_ptr();
        for (i, &v) in x.iter().enumerate().take(self.x.len()) {
            // SAFETY: in bounds; volatile so the stores are not elided or reordered by the compiler.
            unsafe { xs.add(i).write_volatile(v) };
        }
        self.seq = if self.seq >= STOP - 1 {
            1
        } else {
            self.seq + 1
        };
        fence(Ordering::SeqCst);
        let c = self.ctrl.as_mut_ptr();
        // SAFETY: ctrl has 16 elements.
        unsafe { c.write_volatile(self.seq) };
        let t0 = Instant::now();
        let mut spins = 0u32;
        while unsafe { c.add(1).read_volatile() } != self.seq {
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(65536) && t0.elapsed() > self.timeout {
                return Err(Error::Shape(
                    "persistent kernel did not answer before the timeout".into(),
                ));
            }
        }
        fence(Ordering::SeqCst);
        let ys = self.y.as_ptr();
        for (i, o) in out.iter_mut().enumerate().take(self.y.len()) {
            // SAFETY: in bounds.
            *o = unsafe { ys.add(i).read_volatile() };
        }
        Ok(())
    }

    /// Tell the kernel to exit and wait (up to `timeout`) for it.
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }
        fence(Ordering::SeqCst);
        // SAFETY: ctrl has 16 elements.
        unsafe { self.ctrl.as_mut_ptr().write_volatile(STOP) };
        let t0 = Instant::now();
        loop {
            // SAFETY: valid stream handle.
            match unsafe { sys::cuStreamQuery(self.stream.cu_stream()) } {
                sys::CUresult::CUDA_SUCCESS => break,
                sys::CUresult::CUDA_ERROR_NOT_READY if t0.elapsed() < self.timeout => {
                    std::thread::yield_now()
                }
                sys::CUresult::CUDA_ERROR_NOT_READY => {
                    return Err(Error::Shape(
                        "persistent kernel did not stop before the timeout".into(),
                    ))
                }
                e => return Err(Error::Driver(e.result().unwrap_err())),
            }
        }
        self.running = false;
        let _ = self.gpu;
        Ok(())
    }
}
impl Drop for PersistentEngine<'_> {
    fn drop(&mut self) {
        if self.stop().is_err() {
            // The kernel may still be reading the mapped buffers: leak them rather than free memory
            // a running kernel can touch.
            std::mem::forget(std::mem::replace(
                &mut self.ctrl,
                PinnedBuf::new(&self.gpu.ctx, 1, false).unwrap(),
            ));
            std::mem::forget(std::mem::replace(
                &mut self.x,
                PinnedBuf::new(&self.gpu.ctx, 1, false).unwrap(),
            ));
            std::mem::forget(std::mem::replace(
                &mut self.y,
                PinnedBuf::new(&self.gpu.ctx, 1, false).unwrap(),
            ));
        }
    }
}

/// cuBLAS FP16 baseline for the same model: dense fp16 weights (dequantized ternary * scale), one
/// `cublasGemmEx` + one bias/activation kernel per layer, captured as a CUDA graph with the copies.
/// One cuBLAS layer: (n, k, fp16 weights, f32 bias, activation id).
type CublasLayer = (usize, usize, CudaSlice<f16>, CudaSlice<f32>, i32);

pub struct CublasMlp<'g> {
    gpu: &'g Gpu,
    blas: Cublas,
    layers: Vec<CublasLayer>,
    x: CudaSlice<f32>,
    h: [CudaSlice<f16>; 2],
    y: CudaSlice<f32>,
    h_in: PinnedBuf<f32>,
    h_out: PinnedBuf<f32>,
    graph: Option<Graph>,
    pub graph_error: Option<String>,
}

impl<'g> CublasMlp<'g> {
    pub fn new(gpu: &'g Gpu, m: &TernaryMlp) -> Result<Self> {
        m.validate().map_err(Error::Shape)?;
        let s = &gpu.stream;
        let mut layers = Vec::new();
        for l in &m.layers {
            let w: Vec<f16> = (0..l.n * l.k)
                .map(|i| f16::from_f32(l.w[i] as f32 * l.scale[i / l.k]))
                .collect();
            layers.push((
                l.n,
                l.k,
                s.clone_htod(&w)?,
                s.clone_htod(&l.bias)?,
                l.act as i32,
            ));
        }
        let d = m.max_dim();
        let mut e = Self {
            gpu,
            blas: Cublas::new(gpu)?,
            layers,
            x: s.alloc_zeros(m.in_dim())?,
            h: [s.alloc_zeros(d)?, s.alloc_zeros(d)?],
            y: s.alloc_zeros(m.out_dim())?,
            h_in: PinnedBuf::new(&gpu.ctx, m.in_dim(), false)?,
            h_out: PinnedBuf::new(&gpu.ctx, m.out_dim(), false)?,
            graph: None,
            graph_error: None,
        };
        s.synchronize()?;
        // Warm up once outside capture (cuBLAS may initialize lazily).
        e.enqueue()?;
        s.synchronize()?;
        let captured = {
            let er = &mut e;
            let gpu = er.gpu;
            gpu.capture(|| er.enqueue())
        };
        match captured {
            Ok(g) => e.graph = Some(g),
            Err(err) => e.graph_error = Some(err.to_string()),
        }
        Ok(e)
    }

    pub fn input_mut(&mut self) -> &mut [f32] {
        self.h_in.as_mut_slice()
    }
    pub fn output(&self) -> &[f32] {
        self.h_out.as_slice()
    }
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    fn enqueue(&mut self) -> Result<()> {
        let s = &self.gpu.stream;
        let (xd, yd) = (dptr(&self.x, s), dptr(&self.y, s));
        unsafe {
            htod_async(
                xd,
                self.h_in.as_ptr() as *const c_void,
                self.h_in.len() * 4,
                s,
            )?
        };
        let n_in = self.h_in.len() as i32;
        let grid = |n: usize| LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let [h0, _] = &mut self.h;
            let mut b = s.launch_builder(self.gpu.func("f32_to_f16"));
            b.arg(&self.x).arg(h0).arg(&n_in);
            // SAFETY: sizes validated in new().
            unsafe { b.launch(grid(n_in as usize))? };
        }
        let nl = self.layers.len();
        for (i, (n, k, w, bias, act)) in self.layers.iter().enumerate() {
            let [h0, h1] = &mut self.h;
            let (src, dst) = if i % 2 == 0 { (&*h0, h1) } else { (&*h1, h0) };
            self.blas.gemm(w, src, dst, *n, *k, 1)?;
            let out = if i + 1 == nl { yd } else { 0u64 };
            let n32 = *n as i32;
            let mut b = s.launch_builder(self.gpu.func("bias_act_f16"));
            b.arg(dst).arg(bias).arg(&n32).arg(act).arg(&out);
            // SAFETY: sizes validated in new().
            unsafe { b.launch(grid(*n))? };
        }
        unsafe {
            dtoh_async(
                self.h_out.as_mut_ptr() as *mut c_void,
                yd,
                self.h_out.len() * 4,
                s,
            )?
        };
        Ok(())
    }

    pub fn run(&mut self, submit: Submit, w: Wait) -> Result<()> {
        match (&self.graph, submit) {
            (Some(g), Submit::Graph) => g.launch()?,
            _ => self.enqueue()?,
        }
        wait(&self.gpu.stream, w)
    }
}
