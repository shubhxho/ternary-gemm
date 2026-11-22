# ternary-gemm (Python)

CPU bindings for the `ternary-gemm` Rust crate: 2-bit ternary packing, the bit-exact W2A8 GEMM,
the SIMD signal-model engine and the looped transformer. GPU paths stay in Rust for now.

```bash
pip install ./bindings/python            # builds the wheel with maturin (needs Rust)
pip install "./bindings/python[test]" && (cd bindings/python/tests && pytest -q)
```

```python
import ternary_gemm as tg

p = tg.PackedTernary.from_float(weights, n=2560, k=2560)  # absmean ternarization + 2-bit pack
y = tg.CpuGemm(p, threads=4)(x, batch=4)                    # flat lists, row-major

m = tg.SignalModel.random([64, 256, 256, 1], seed=0)
signal = m(features)                                        # allocation-free SIMD path

lt = tg.LoopedTransformer.random(512, 8, 2048, loops=8, inject="add", halt_tol=0.05)
hidden, loops_run = lt(x_tokens, tokens=16)
```

PyTorch (optional, >= 2.4): `ternary_gemm.torch_ops` registers
`torch.ops.ternary_gemm.gemm_i8(x, handle)` with a fake (meta) kernel for shape inference.
It round-trips through Python lists: an integration stub, not a fast path.

```python
from ternary_gemm import torch_ops
h = torch_ops.register_weights(p)
y = torch.ops.ternary_gemm.gemm_i8(torch.randn(4, 2560), h)
```

Arrays are flat Python lists to avoid a NumPy build dependency; values are rounded to fp16 on
the way in, exactly as the kernels see them. The GEMM is bit-identical to the crate's
reference (tested against an independent pure-Python oracle).
