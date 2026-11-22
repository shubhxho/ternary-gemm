"""Check langs/futhark/ternary.fut against the Rust reference vectors.

Compiles the program with ``futhark c`` (sequential C backend; ``--backend multicore`` also
works), feeds each case through stdin in Futhark's textual value format and compares the integer
row sums and the fp16 output bits exactly.

Usage: python3 langs/futhark/check.py [vectors_dir] [--futhark PATH] [--backend c|multicore]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent


def load(path: pathlib.Path):
    it = iter(path.read_text().split())
    n, k = int(next(it)), int(next(it))
    s_bits = int(next(it), 16)
    next(it)
    scale = [int(next(it), 16) for _ in range(n)]
    words = [int(next(it), 16) for _ in range(n * k // 16)]
    q = [int(next(it)) for _ in range(k)]
    acc = [int(next(it)) for _ in range(n)]
    out = [int(next(it), 16) for _ in range(n)]
    return n, k, s_bits, scale, words, q, acc, out


def arr(xs, suffix):
    return "[" + ", ".join(f"{x}{suffix}" for x in xs) + "]"


def run(exe: pathlib.Path, entry: str, text: str) -> list[int]:
    o = subprocess.run([str(exe), "-e", entry], input=text, capture_output=True, text=True, check=True)
    body = o.stdout.strip().strip("[]")
    # Values print with a type suffix ("-17i32", "15360u16"); keep the leading integer.
    return [int(re.match(r"-?\d+", t.strip()).group(0)) for t in body.split(",") if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors", nargs="?", default="langs/vectors")
    ap.add_argument("--futhark", default="futhark")
    ap.add_argument("--backend", default="c", choices=["c", "multicore"])
    a = ap.parse_args()
    cases = sorted(pathlib.Path(a.vectors).glob("case*.txt"))
    if not cases:
        print("no vectors; run `cargo run --release --example lang_vectors`")
        return 2
    with tempfile.TemporaryDirectory() as td:
        exe = pathlib.Path(td) / "ternary"
        subprocess.run([a.futhark, a.backend, str(HERE / "ternary.fut"), "-o", str(exe)], check=True)
        bad = 0
        for c in cases:
            n, k, s_bits, scale, words, q, acc, out = load(c)
            kw = k // 16
            w_txt = "[" + ", ".join(arr(words[r * kw : (r + 1) * kw], "u32") for r in range(n)) + "]"
            q_txt = "[" + ", ".join(arr(q[w * 16 : (w + 1) * 16], "i8") for w in range(kw)) + "]"
            got_acc = run(exe, "gemv_acc", f"{w_txt} {q_txt}")
            got = run(exe, "gemv_bits", f"{w_txt} {q_txt} {s_bits}u32 {arr(scale, 'u32')}")
            ok = got_acc == acc and got == out
            bad += not ok
            print(f"futhark ({a.backend}) {c.stem} n={n} k={k}: {'OK (acc + fp16 bits exact)' if ok else 'MISMATCH'}")
        # Timing via the generated program's own runner (-t: microseconds per run, excluding I/O).
        n, k, s_bits, scale, words, q, _, _ = load(cases[-1])
        kw = k // 16
        w_txt = "[" + ", ".join(arr(words[r * kw : (r + 1) * kw], "u32") for r in range(n)) + "]"
        q_txt = "[" + ", ".join(arr(q[w * 16 : (w + 1) * 16], "i8") for w in range(kw)) + "]"
        tfile = pathlib.Path(td) / "t.txt"
        t0 = time.time()
        subprocess.run(
            [str(exe), "-e", "gemv_acc", "-r", "200", "-t", str(tfile)],
            input=f"{w_txt} {q_txt}",
            capture_output=True,
            text=True,
            check=True,
        )
        runs = [int(x) for x in tfile.read_text().split()]
        runs.sort()
        print(
            f"futhark ({a.backend}) gemv_acc {n}x{k}: median {runs[len(runs) // 2]} us/call "
            f"over {len(runs)} runs (this machine; wall {time.time() - t0:.1f}s incl. parsing)"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
