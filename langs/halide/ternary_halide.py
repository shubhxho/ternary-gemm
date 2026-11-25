"""Ternary W2A8 GEMV as a Halide pipeline (Python bindings), JIT-compiled for the host CPU.

Algorithm and schedule are separate, as usual in Halide. The algorithm is the crate's contract
for batch 1: natural layout (weight ``16*w + j`` of a row is ``(words[w] >> 2j) & 3``, code
``w + 1``), ``acc = sum w * q`` in int32, ``y = f16((f32(acc) / s) * scale)`` under
``strict_float`` so Halide may not reassociate or contract the epilogue. The schedule splits
the 16 slots of a word into vector lanes with ``rfactor`` and parallelizes over rows.

``compile_cuda_object`` also lowers a GPU schedule for a CUDA target to an object file (no GPU
needed), as a compile check of Halide's PTX backend.
"""

from __future__ import annotations

import halide as hl


def build(n: int, k: int, schedule: str = "cpu"):
    kw = k // 16
    words = hl.ImageParam(hl.UInt(32), 2, "words")  # (word, row)
    q = hl.ImageParam(hl.Int(8), 1, "q")
    scale = hl.ImageParam(hl.Float(32), 1, "scale")
    s = hl.Param(hl.Float(32), "s")
    row, j = hl.Var("row"), hl.Var("j")
    r = hl.RDom([(0, kw), (0, 16)], "r")
    code = hl.cast(hl.Int(32), (words[r.x, row] >> hl.cast(hl.UInt(32), 2 * r.y)) & 3)
    acc = hl.Func("acc")
    acc[row] = hl.i32(0)
    acc[row] += (code - 1) * hl.cast(hl.Int(32), q[16 * r.x + r.y])
    out = hl.Func("out")
    out[row] = hl.cast(hl.Float(16), hl.strict_float((hl.cast(hl.Float(32), acc[row]) / s) * scale[row]))
    if schedule == "cpu":
        intm = acc.update().rfactor(r.y, j)
        intm.compute_at(acc, row).vectorize(j, 16).update().vectorize(j, 16)
        ro, ri = hl.Var("ro"), hl.Var("ri")
        out.split(row, ro, ri, 16, hl.TailStrategy.GuardWithIf).parallel(ro)
        acc.compute_at(out, ri)
    else:
        ro, ri = hl.Var("ro"), hl.Var("ri")
        out.gpu_tile(row, ro, ri, 64)
        acc.compute_at(out, ri)
    return out, words, q, scale, s


def gemv(words_u32, q_i8, scale_f32, s: float, n: int, k: int):
    """numpy in -> (acc int32 [n], y fp16 [n])."""
    import numpy as np

    out, wp, qp, sp, s_p = build(n, k)
    wp.set(hl.Buffer(np.ascontiguousarray(words_u32.reshape(n, k // 16))))
    qp.set(hl.Buffer(q_i8))
    sp.set(hl.Buffer(scale_f32))
    s_p.set(s)
    y = np.asarray(out.realize([n]))
    return y


def compile_cuda_object(path: str, n: int = 4096, k: int = 4096, capability: int = 86) -> None:
    out, wp, qp, sp, s_p = build(n, k, schedule="gpu")
    target = hl.Target(f"x86-64-linux-cuda-cuda_capability_{capability}")
    out.compile_to_object(path, [wp, qp, sp, s_p], "ternary_gemv_cuda", target)


if __name__ == "__main__":
    # Time the JIT-compiled CPU pipeline on a 256 x 4096 GEMV (this machine only).
    import time

    import numpy as np

    n, k = 256, 4096
    rng = np.random.default_rng(0)
    out, wp, qp, sp, s_p = build(n, k)
    wp.set(hl.Buffer(rng.integers(0, 2**32, size=(n, k // 16), dtype=np.uint32)))
    qp.set(hl.Buffer(rng.integers(-128, 128, size=k, dtype=np.int8)))
    sp.set(hl.Buffer(np.full(n, 0.01, dtype=np.float32)))
    s_p.set(1.0)
    out.compile_jit()
    buf = hl.Buffer(hl.Float(16), [n])
    for _ in range(50):
        out.realize(buf)
    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters):
        out.realize(buf)
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"halide gemv {n}x{k}: {us:.2f} us/call (mean incl. Python call overhead, this machine)")
