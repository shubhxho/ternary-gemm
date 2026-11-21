#!/usr/bin/env python3
"""Plot MEASURED benchmark results written by `bench --csv <file>`.

usage: plot_results.py results/<run>.csv   ->  results/<run>-latency.png, -bandwidth.png, -layer_stack.png

Only plots numbers present in the CSV (which bench.rs measured with CUDA events on the GPU it ran on).
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.bbox": "tight"})
COLORS = {"cuBLAS FP16 (GemmEx)": "#9aa0a6", "naive ternary (int8 act.)": "#dadce0",
          "ternary W2A8 gemv only": "#8ab4f8", "ternary W2A8 quantize + gemv": "#4285f4",
          "ternary W2A8 fused": "#1a73e8", "ternary W2A16 (fp16 act.)": "#34a853"}


def main(path):
    gpu, kern, layer = "unknown GPU", [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("# gpu"):
                gpu = f"{row[1]} ({row[2]})"
            elif row[0] == "kernel":
                kern.append(dict(n=int(row[1]), k=int(row[2]), label=row[3], batch=int(row[4]), name=row[5],
                                 us=float(row[6]), gbps=float(row[7]), pct=float(row[8])))
            elif row[0] == "layer_stack":
                layer.append(dict(batch=int(row[1]), path=row[2], us=float(row[4]), tok=float(row[5])))
    stem = os.path.splitext(path)[0]
    out = []
    for batch in sorted({r["batch"] for r in kern})[:1] + sorted({r["batch"] for r in kern})[-1:]:
        rows = [r for r in kern if r["batch"] == batch]
        if not rows:
            continue
        shapes = list(dict.fromkeys((r["n"], r["k"]) for r in rows))
        names = list(dict.fromkeys(r["name"] for r in rows))
        for metric, ylabel, suffix in [("us", "latency (us, lower is better)", "latency"),
                                       ("pct", "% of peak DRAM bandwidth", "bandwidth")]:
            fig, ax = plt.subplots(figsize=(10, 4.2))
            w = 0.8 / len(names)
            for i, nm in enumerate(names):
                vals = [next((r[metric] for r in rows if (r["n"], r["k"]) == s and r["name"] == nm), 0) for s in shapes]
                ax.bar([j + (i - len(names) / 2 + 0.5) * w for j in range(len(shapes))], vals, w,
                       label=nm, color=COLORS.get(nm, None))
            ax.set_xticks(range(len(shapes)))
            ax.set_xticklabels([f"{n}x{k}" for n, k in shapes])
            if metric == "us":
                ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Measured on {gpu}: batch {batch}")
            ax.legend(frameon=False, fontsize=8, ncol=3)
            p = f"{stem}-{suffix}-b{batch}.png"
            fig.savefig(p)
            plt.close(fig)
            out.append(p)
    if layer:
        fig, ax = plt.subplots(figsize=(8, 4))
        batches = sorted({r["batch"] for r in layer})
        paths = list(dict.fromkeys(r["path"] for r in layer))
        w = 0.8 / len(paths)
        for i, pth in enumerate(paths):
            vals = [next((r["tok"] for r in layer if r["batch"] == b and r["path"] == pth), 0) for b in batches]
            ax.bar([j + (i - len(paths) / 2 + 0.5) * w for j in range(len(batches))], vals, w, label=pth)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([f"batch {b}" for b in batches])
        ax.set_ylabel("linear-only tokens/s")
        ax.set_title(f"Measured on {gpu}: BitNet-2B linear stack (30 layers)")
        ax.legend(frameon=False, fontsize=8)
        p = f"{stem}-layer_stack.png"
        fig.savefig(p)
        plt.close(fig)
        out.append(p)
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
