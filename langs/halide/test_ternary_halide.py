"""Halide ternary GEMV vs the Rust reference vectors (``langs/vectors``)."""

from __future__ import annotations

import pathlib
import struct

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("halide")

from ternary_halide import compile_cuda_object, gemv  # noqa: E402

VEC = pathlib.Path(__file__).resolve().parents[1] / "vectors"
CASES = sorted(VEC.glob("case*.txt"))


def load(path):
    it = iter(path.read_text().split())
    n, k = int(next(it)), int(next(it))
    s = struct.unpack("<f", struct.pack("<I", int(next(it), 16)))[0]
    next(it)
    scale = np.array([int(next(it), 16) for _ in range(n)], dtype=np.uint32).view(np.float32)
    words = np.array([int(next(it), 16) for _ in range(n * k // 16)], dtype=np.uint32)
    q = np.array([int(next(it)) for _ in range(k)], dtype=np.int8)
    next_acc = [int(next(it)) for _ in range(n)]
    out = np.array([int(next(it), 16) for _ in range(n)], dtype=np.uint16)
    return n, k, s, scale, words, q, next_acc, out


@pytest.mark.skipif(not CASES, reason="run `cargo run --release --example lang_vectors` first")
@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_matches_rust_reference_bit_exact(path):
    n, k, s, scale, words, q, _acc, out = load(path)
    y = gemv(words, q, scale, s, n, k)
    assert y.dtype == np.float16
    assert y.view(np.uint16).tolist() == out.tolist()


def test_cuda_schedule_compiles(tmp_path):
    try:
        compile_cuda_object(str(tmp_path / "gemv_cuda.o"))
    except RuntimeError as e:  # wheel built without the NVPTX backend
        pytest.skip(f"Halide CUDA backend unavailable: {e}")
    assert (tmp_path / "gemv_cuda.o").stat().st_size > 0
