"""Tests for the Python bindings (run after `pip install ./bindings/python`)."""

import math
import random

import pytest
import ternary_gemm as tg


def _ternary(n, k, seed=0):
    rng = random.Random(seed)
    return [rng.choice((-1, 0, 1)) for _ in range(n * k)]


def test_pack_roundtrip_and_metadata():
    w = _ternary(8, 128)
    p = tg.PackedTernary(w, 8, 128, [0.5] * 8)
    assert (p.n, p.k) == (8, 128)
    assert p.unpack() == w
    assert 0.0 <= p.sparsity() <= 1.0
    assert p.nbytes() == 8 * 128 // 4
    assert "n=8" in repr(p)


def test_pack_rejects_bad_input():
    with pytest.raises(ValueError):
        tg.PackedTernary([0] * 8 * 100, 8, 100, [1.0] * 8)  # k % 64 != 0
    with pytest.raises(ValueError):
        tg.PackedTernary([2] + [0] * (64 - 1), 1, 64, [1.0])  # not ternary
    with pytest.raises(ValueError):
        tg.PackedTernary([0] * 10, 8, 128, [1.0] * 8)  # wrong length


def test_from_float_absmean():
    rng = random.Random(1)
    w = [rng.gauss(0, 1) for _ in range(4 * 64)]
    p = tg.PackedTernary.from_float(w, 4, 64)
    assert set(p.unpack()) <= {-1, 0, 1}
    s = p.row_scale
    assert all(math.isclose(v, s[0]) for v in s)  # per-tensor scale by default


def _oracle(w, n, k, x, batch, scale):
    """Independent re-implementation of the W2A8 math (absmax int8, exact int dot, f32 epilogue)."""
    import struct

    def f16(v):
        return struct.unpack("e", struct.pack("e", v))[0]

    def f32(v):
        return struct.unpack("f", struct.pack("f", v))[0]

    out = []
    for b in range(batch):
        xb = [f16(v) for v in x[b * k : (b + 1) * k]]
        m = max(max(abs(v) for v in xb), 1e-5)
        s = f32(127.0 / f32(m))
        q = [max(-128, min(127, round(f32(v * s)))) for v in xb]
        for r in range(n):
            acc = sum(w[r * k + c] * q[c] for c in range(k))
            out.append(f16(f32(f32(acc / s) * scale[r])))
    return out


def test_cpu_gemm_matches_python_oracle():
    n, k, batch = 5, 128, 3
    w = _ternary(n, k, 2)
    scale = [0.01 * (r + 1) for r in range(n)]
    p = tg.PackedTernary(w, n, k, scale)
    rng = random.Random(3)
    x = [rng.uniform(-2, 2) for _ in range(batch * k)]
    y = tg.CpuGemm(p, threads=2)(x, batch)
    want = _oracle(w, n, k, x, batch, scale)
    assert len(y) == batch * n
    # Python's round() is round-half-even like the kernels; allow 1 fp16 ulp for f32 op-order nuance.
    for a, b in zip(y, want, strict=True):
        assert abs(a - b) <= max(abs(b), 1e-3) * 2**-10


def test_cpu_gemm_shape_errors():
    p = tg.PackedTernary(_ternary(2, 64), 2, 64, [1.0, 1.0])
    g = tg.CpuGemm(p)
    with pytest.raises(ValueError):
        g([0.0] * 63, 1)
    with pytest.raises(ValueError):
        g([0.0] * 64, 1, residual=[0.0])


def test_signal_model_deterministic():
    m = tg.SignalModel.random([64, 256, 256, 1], seed=7)
    x = [0.01 * i for i in range(64)]
    a, b = m(x), m(x)
    assert a == b and len(a) == 1
    assert isinstance(m.simd, str)
    with pytest.raises(ValueError):
        m([0.0] * 3)


def test_looped_transformer():
    m = tg.LoopedTransformer.random(128, 4, 256, loops=3, inject="adapter", seed=1)
    x = [0.001 * (i % 97) for i in range(4 * 128)]
    h, loops = m(x, 4)
    assert len(h) == 4 * 128 and loops == 3
    assert all(math.isfinite(v) for v in h)
    assert m.unique_weight_bytes > 0
    halting = tg.LoopedTransformer.random(128, 4, 256, loops=8, halt_tol=1e9, min_loops=2)
    assert halting(x, 4)[1] == 2
    with pytest.raises(ValueError):
        tg.LoopedTransformer.random(128, 4, 256, loops=1, inject="bogus")


def test_torch_op_if_available():
    torch = pytest.importorskip("torch")
    from ternary_gemm import torch_ops

    if not torch_ops.AVAILABLE:
        pytest.skip("torch.library.custom_op unavailable")
    w = _ternary(4, 64, 5)
    p = tg.PackedTernary(w, 4, 64, [1.0] * 4)
    h = torch_ops.register_weights(p)
    x = torch.randn(2, 64)
    y = torch.ops.ternary_gemm.gemm_i8(x, h)
    assert y.shape == (2, 4)
    assert torch.allclose(y, torch.tensor(tg.CpuGemm(p)(x.reshape(-1).tolist(), 2)).reshape(2, 4))
