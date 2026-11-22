//! Python bindings (CPU paths only; GPU work stays in Rust for now).
//!
//! Exposes packing, the bit-exact CPU W2A8 GEMM, the signal-model CPU engine and the looped
//! transformer. Arrays cross the boundary as flat Python lists (row-major), keeping the binding
//! free of a NumPy dependency; `ternary_gemm.torch_ops` wraps them for PyTorch tensors.

use half::f16;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ternary_gemm::cpu::{quantize_acts, CpuTernary};
use ternary_gemm::looped::{BlockConfig, Halting, Inject, LoopConfig, LoopedTransformer};
use ternary_gemm::mlp::{Act, CpuEngine, TernaryMlp};
use ternary_gemm::PackedTernary;

fn verr<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A packed ternary weight matrix `[n][k]` (2 bits per weight) with per-row scales.
#[pyclass(name = "PackedTernary", module = "ternary_gemm")]
#[derive(Clone)]
struct PyPacked {
    inner: PackedTernary,
}

#[pymethods]
impl PyPacked {
    /// Pack ternary weights (values in {-1, 0, 1}, row-major `[n][k]`, `k % 64 == 0`).
    #[new]
    fn new(weights: Vec<i8>, n: usize, k: usize, row_scale: Vec<f32>) -> PyResult<Self> {
        if weights.len() != n * k {
            return Err(verr(format!(
                "expected {} weights, got {}",
                n * k,
                weights.len()
            )));
        }
        Ok(Self {
            inner: PackedTernary::from_ternary(&weights, n, k, row_scale).map_err(verr)?,
        })
    }

    /// BitNet absmean ternarization of f32 weights, then pack.
    #[staticmethod]
    #[pyo3(signature = (weights, n, k, per_row=false))]
    fn from_float(weights: Vec<f32>, n: usize, k: usize, per_row: bool) -> PyResult<Self> {
        if weights.len() != n * k {
            return Err(verr(format!(
                "expected {} weights, got {}",
                n * k,
                weights.len()
            )));
        }
        let (t, s) = ternary_gemm::quantize_absmean(&weights, n, k, per_row);
        Self::new(t, n, k, s)
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }
    #[getter]
    fn row_scale(&self) -> Vec<f32> {
        self.inner.row_scale.clone()
    }
    /// The ternary weights, row-major.
    fn unpack(&self) -> Vec<i8> {
        self.inner.unpack()
    }
    /// Fraction of zero weights.
    fn sparsity(&self) -> f64 {
        self.inner.sparsity()
    }
    /// Packed weight bytes (2 bits per weight; scales not included).
    fn nbytes(&self) -> usize {
        self.inner.weight_bytes()
    }
    fn __repr__(&self) -> String {
        format!("PackedTernary(n={}, k={})", self.inner.n, self.inner.k)
    }
}

/// CPU W2A8 GEMM, bit-identical to the crate's reference and GPU kernels.
#[pyclass(name = "CpuGemm", module = "ternary_gemm")]
struct PyCpuGemm {
    inner: CpuTernary,
}

#[pymethods]
impl PyCpuGemm {
    #[new]
    #[pyo3(signature = (packed, threads=1))]
    fn new(packed: &PyPacked, threads: usize) -> Self {
        Self {
            inner: CpuTernary::new(&packed.inner).with_threads(threads),
        }
    }

    /// `y[b][n] = W . quant(x[b])`: x is `[batch][k]` floats (rounded to fp16 like the kernels),
    /// optional RMSNorm `gamma` fused into quantization, optional `residual` `[batch][n]`.
    /// Returns `[batch][n]` floats (fp16 values).
    #[pyo3(signature = (x, batch, gamma=None, residual=None, eps=1e-5))]
    fn __call__(
        &self,
        py: Python<'_>,
        x: Vec<f32>,
        batch: usize,
        gamma: Option<Vec<f32>>,
        residual: Option<Vec<f32>>,
        eps: f32,
    ) -> PyResult<Vec<f32>> {
        let (n, k) = (self.inner.n, self.inner.k);
        if x.len() != batch * k {
            return Err(verr(format!("x must have batch*k = {} values", batch * k)));
        }
        if gamma.as_ref().is_some_and(|g| g.len() != k) {
            return Err(verr(format!("gamma must have k = {k} values")));
        }
        if residual.as_ref().is_some_and(|r| r.len() != batch * n) {
            return Err(verr(format!(
                "residual must have batch*n = {} values",
                batch * n
            )));
        }
        let to16 = |v: &[f32]| v.iter().map(|&a| f16::from_f32(a)).collect::<Vec<_>>();
        let xh = to16(&x);
        let gh = gamma.as_deref().map(to16);
        let rh = residual.as_deref().map(to16);
        let y = py.detach(|| {
            let a = quantize_acts(&xh, batch, k, gh.as_deref(), eps);
            self.inner.gemm_i8(&a, rh.as_deref())
        });
        Ok(y.iter().map(|v| v.to_f32()).collect())
    }
}

/// A small ternary MLP (signal model) on the allocation-free SIMD CPU engine.
#[pyclass(name = "SignalModel", module = "ternary_gemm")]
struct PySignal {
    engine: CpuEngine,
    out: Vec<f32>,
}

#[pymethods]
impl PySignal {
    /// Random model with layer widths `dims` (e.g. `[64, 256, 256, 1]`), ReLU^2 hidden layers.
    #[staticmethod]
    #[pyo3(signature = (dims, seed=0))]
    fn random(dims: Vec<usize>, seed: u64) -> PyResult<Self> {
        let m = TernaryMlp::random(&dims, Act::Relu2, seed).map_err(verr)?;
        let out = vec![0.0; m.out_dim()];
        Ok(Self {
            engine: CpuEngine::new(&m).map_err(verr)?,
            out,
        })
    }

    /// One forward pass (batch 1).
    fn __call__(&mut self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.engine.model().in_dim() {
            return Err(verr(format!(
                "expected {} inputs",
                self.engine.model().in_dim()
            )));
        }
        self.engine.forward_into(&x, &mut self.out);
        Ok(self.out.clone())
    }

    /// The SIMD path in use (`"avx512-vnni"`, `"avx2"`, ...).
    #[getter]
    fn simd(&self) -> &'static str {
        self.engine.simd()
    }
}

/// Weight-shared looped ternary transformer (CPU backend).
#[pyclass(name = "LoopedTransformer", module = "ternary_gemm")]
struct PyLooped {
    inner: LoopedTransformer,
}

#[pymethods]
impl PyLooped {
    /// Random model. `inject` is "none", "add" or "adapter"; `halt_tol` enables adaptive halting.
    #[staticmethod]
    #[pyo3(signature = (d_model, n_heads, d_ff, loops, inject="add", halt_tol=None, min_loops=1, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn random(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        loops: usize,
        inject: &str,
        halt_tol: Option<f32>,
        min_loops: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let inject = match inject {
            "none" => Inject::None,
            "add" => Inject::Add,
            "adapter" => Inject::Adapter,
            o => return Err(verr(format!("inject must be none|add|adapter, got {o}"))),
        };
        let cfg = LoopConfig {
            loops,
            inject,
            halting: halt_tol.map(|tol| Halting { tol, min_loops }),
        };
        let b = BlockConfig {
            d_model,
            n_heads,
            d_ff,
        };
        Ok(Self {
            inner: LoopedTransformer::random(b, cfg, seed).map_err(verr)?,
        })
    }

    /// Forward `[tokens][d_model]` floats; returns `(hidden, loops_run)`.
    fn __call__(&self, py: Python<'_>, x: Vec<f32>, tokens: usize) -> PyResult<(Vec<f32>, usize)> {
        let d = self.inner.block.cfg.d_model;
        if x.len() != tokens * d {
            return Err(verr(format!(
                "x must have tokens*d_model = {} values",
                tokens * d
            )));
        }
        let xh: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
        let out = py.detach(|| self.inner.forward_cpu(&xh, tokens));
        Ok((
            out.hidden.iter().map(|v| v.to_f32()).collect(),
            out.loops_run,
        ))
    }

    #[getter]
    fn unique_weight_bytes(&self) -> usize {
        self.inner.unique_weight_bytes()
    }
}

#[pymodule]
fn _ternary_gemm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPacked>()?;
    m.add_class::<PyCpuGemm>()?;
    m.add_class::<PySignal>()?;
    m.add_class::<PyLooped>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
