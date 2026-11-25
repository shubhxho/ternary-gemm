"""Triton ternary GEMV vs the Rust reference vectors (``langs/vectors``), on the CPU interpreter.

Run: ``TRITON_INTERPRET=1 pytest langs/triton`` after
``cargo run --release --example lang_vectors``. Also checks that the kernel lowers to PTX for
sm_80 and sm_120 (no GPU needed) when Triton's NVIDIA backend can run.
"""

from __future__ import annotations

import os
import pathlib
import struct

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from ternary_gemv import compile_ptx, ternary_gemv  # noqa: E402

VEC = pathlib.Path(__file__).resolve().parents[1] / "vectors"


def load(path: pathlib.Path):
    tok = path.read_text().split()
    it = iter(tok)
    n, k = int(next(it)), int(next(it))
    s = struct.unpack("<f", struct.pack("<I", int(next(it), 16)))[0]
    next(it)  # sum(q)
    scale = [struct.unpack("<f", struct.pack("<I", int(next(it), 16)))[0] for _ in range(n)]
    words = [int(next(it), 16) for _ in range(n * k // 16)]
    q = [int(next(it)) for _ in range(k)]
    acc = [int(next(it)) for _ in range(n)]
    out = [int(next(it), 16) for _ in range(n)]
    return n, k, s, scale, words, q, acc, out


CASES = sorted(VEC.glob("case*.txt"))


@pytest.mark.skipif(not CASES, reason="run `cargo run --release --example lang_vectors` first")
@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_matches_rust_reference_bit_exact(path):
    n, k, s, scale, words, q, acc, out = load(path)
    w = torch.tensor([x - (1 << 32) if x >= 1 << 31 else x for x in words], dtype=torch.int32).view(n, k // 16)
    got_acc, got = ternary_gemv(
        w,
        torch.tensor(q, dtype=torch.int8),
        torch.tensor(scale, dtype=torch.float32),
        s,
    )
    assert got_acc.tolist() == acc
    assert got.view(torch.int16).tolist() == [x - (1 << 16) if x >= 1 << 15 else x for x in out]


@pytest.mark.parametrize("arch", [80, 120])
def test_lowers_to_ptx(arch):
    if os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("TRITON_PTX_CHECK") != "1":
        pytest.skip("set TRITON_PTX_CHECK=1 to also lower to PTX (needs Triton's NVIDIA backend)")
    ptx = compile_ptx(arch)
    assert f".target sm_{arch}" in ptx
    assert "div.rn.f32" in ptx, "the epilogue must use an IEEE divide"
