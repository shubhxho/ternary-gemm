#!/usr/bin/env python3
"""Plot MEASURED latency results written by `latency --csv <file>`.

usage: plot_latency.py results/<run>.csv -> results/<run>-tail.png, results/<run>-hist-<model>.png
Only numbers present in the CSV are plotted; titles name the CPU / GPU they were measured on.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)
STYLE = {
    "CPU ternary scalar reference": ("#bdc1c6", ":"),
    "CPU ternary SIMD (CpuEngine)": ("#1a73e8", "-"),
    "CPU dense f32 SIMD": ("#5f6368", "--"),
    "GPU ternary graph + spin": ("#e8710a", "-"),
    "GPU ternary direct + spin": ("#f9ab00", "--"),
    "GPU ternary graph + blocking sync": ("#fcc934", ":"),
    "GPU ternary persistent (experimental)": ("#d93025", "-"),
    "GPU cuBLAS FP16 graph + spin": ("#188038", "--"),
}
PCTS = ["p50", "p90", "p99", "p99.9", "max"]


def main(path):
    cpu, gpu = None, None
    lat, hist = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == "# cpu":
                cpu = row[1]
            elif row[0] == "# gpu":
                gpu = row[1]
            elif row[0] == "lat":
                lat[row[1]][row[2]] = [float(v) for v in row[5:10]]
            elif row[0] == "hist":
                hist[row[1]][row[2]].append((float(row[3]), float(row[4]), int(row[5])))
    where = f"CPU {cpu}" + (f" + GPU {gpu}" if gpu else " (no GPU rows)")
    stem = os.path.splitext(path)[0]
    out = []
    models = list(lat)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.8), squeeze=False)
    for ax, m in zip(axes[0], models, strict=False):
        for p, vals in lat[m].items():
            c, ls = STYLE.get(p, (None, "-"))
            ax.plot(PCTS, vals, marker="o", ms=3, color=c, ls=ls, label=p)
        ax.set_yscale("log")
        ax.set_title(m, fontsize=9)
        ax.set_ylabel("end-to-end latency (us, log)")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Measured: batch-1 latency percentiles\n{where}", fontsize=9, fontweight="bold", y=1.06)
    p = f"{stem}-tail.png"
    fig.savefig(p)
    plt.close(fig)
    out.append(p)
    for m in models:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for pth, bins in hist[m].items():
            bins.sort()
            total = sum(b[2] for b in bins)
            xs, ys = [], []
            for lo, hi, cnt in bins:
                xs += [lo, hi]
                ys += [cnt / total, cnt / total]
            c, ls = STYLE.get(pth, (None, "-"))
            ax.plot(xs, ys, color=c, ls=ls, label=pth)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("latency per request (us, log bins, 20 per decade)")
        ax.set_ylabel("fraction of requests")
        ax.set_title(f"Measured: latency distribution / jitter, {m}\n{where}", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        p = f"{stem}-hist-{m.split()[0]}.png"
        fig.savefig(p)
        plt.close(fig)
        out.append(p)
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
