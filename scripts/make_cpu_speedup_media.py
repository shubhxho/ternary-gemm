#!/usr/bin/env python3
"""Generate the PR #7 CPU Packed2 speedup explainer video, poster, and GIF.

1920x1080 dark-theme animation, same visual language as make_tensor_core_media.py.
Every latency figure is a measured CPU p50 from docs/CPU_PERF.md. No GPU timings
appear, and the 8x vs fp16 figure is labelled as a theoretical batch-1 ceiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"

# Measured p50 (µs), one AVX-512 core, Xeon VM, 20k requests, before = d39d557.
# Source: docs/CPU_PERF.md headline table and the 4-thread table.
TINY_BEFORE, TINY_AFTER = 2.53, 1.11
SMALL_BEFORE, SMALL_AFTER = 20.2, 9.32
MEDIUM_BEFORE, MEDIUM_AFTER = 87.7, 22.3
MEDIUM_4T = 11.62
TINY_SPEEDUP, SMALL_SPEEDUP, MEDIUM_SPEEDUP = "2.3×", "2.2×", "3.9×"

IBM = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Sans/IBMPlexSans-VariableFont_wdth,wght.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

BG = "#0c0e12"
FG = "#e8eaed"
MUTED = "#8b919a"
ACCENT = "#8ec5e0"
GREEN = "#8fbf9f"
RED = "#e07a5f"
GOLD = "#e2c07a"
GRAY = "#4a5160"

DURATION = 38.0
FPS = 30
# Scene starts (inclusive).
S1, S2, S3, S4, S5, S6, S7 = 0.0, 5.5, 12.5, 16.8, 26.0, 31.2, 35.2


def _register():
    for p in (IBM, SERIF, SERIF_BOLD):
        if Path(p).exists():
            font_manager.fontManager.addfont(p)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def scene_local(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return ease((t - start) / (end - start))


def dark_rc():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
            "text.color": FG,
            "axes.labelcolor": FG,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def draw_header(ax, title: str, kicker: str, alpha: float = 1.0):
    ax.text(
        0.04, 0.94, "TERNARY-GEMM", transform=ax.transAxes, fontsize=13, color=ACCENT, alpha=alpha, fontweight="medium"
    )
    ax.text(0.04, 0.88, title, transform=ax.transAxes, fontsize=32, color=FG, alpha=alpha, fontweight="medium")
    ax.text(0.04, 0.81, kicker, transform=ax.transAxes, fontsize=16, color=MUTED, alpha=alpha)


def scene_pack(ax, t: float) -> None:
    a = scene_local(t, S1, S1 + 0.7)
    draw_header(
        ax,
        "A ternary weight is two bits.",
        "Each weight lives in {−1, 0, +1} and is packed in 2 bits.",
        a,
    )
    weights = [1, -1, 0, 1, 1, 0, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1]
    colors = {1: GREEN, 0: "#3d4450", -1: RED}
    labels = {1: "+1", 0: "0", -1: "−1"}

    appear = scene_local(t, S1 + 0.4, S1 + 2.0)
    for i, w in enumerate(weights):
        local = ease(max(0, min(1, appear * 16 - i)) if appear < 1 else 1)
        x = 0.8 + i * 1.15
        y = 5.6
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                1.0,
                1.15,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=colors[w],
                edgecolor="none",
                alpha=local * a,
            )
        )
        ax.text(x + 0.5, y + 0.58, labels[w], ha="center", va="center", fontsize=13, color="white", alpha=local * a)
    ax.text(9.6, 5.15, "16 weights", ha="center", color=MUTED, fontsize=12, alpha=appear * a)

    pack = scene_local(t, S1 + 2.2, S1 + 3.8)
    ax.text(
        9.6, 3.55, "one u32   ·   2 bits each   ·   c = w + 1", ha="center", color=MUTED, fontsize=13, alpha=pack * a
    )
    codes = [w + 1 for w in weights]
    bit_c = {0: RED, 1: "#3d4450", 2: GREEN}
    for i, c in enumerate(codes):
        x = 0.8 + i * 1.15
        y = 2.15 + (1 - pack) * 2.6
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                1.0,
                1.0,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor=bit_c[c],
                edgecolor="#0c0e12",
                linewidth=0.8,
                alpha=max(pack, 0.05) * a,
            )
        )
        ax.text(
            x + 0.5, y + 0.5, format(c, "02b"), ha="center", va="center", fontsize=10, color="white", alpha=pack * a
        )
    ax.text(
        9.6,
        1.5,
        "fp16 would have used 256 bits for the same 16 weights.",
        ha="center",
        color=GOLD,
        fontsize=15,
        alpha=scene_local(t, S1 + 4.0, S1 + 4.8) * a,
    )


def scene_packed2(ax, t: float) -> None:
    a = scene_local(t, S2, S2 + 0.7)
    draw_header(
        ax,
        "Packed2: one load, four VNNI steps.",
        "A row chunk is 64 bytes. Byte j holds k, k+64, k+128 and k+192.",
        a,
    )

    load = scene_local(t, S2 + 0.6, S2 + 1.6)
    ax.add_patch(
        patches.FancyBboxPatch(
            (1.0, 6.15),
            17.2,
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=ACCENT,
            edgecolor="none",
            alpha=load * a,
        )
    )
    ax.text(
        9.6,
        6.72,
        "one 64-byte load",
        ha="center",
        va="center",
        fontsize=18,
        color="#0c0e12",
        alpha=load * a,
        fontweight="medium",
    )

    # Four bit-pair lanes feeding four vpdpbusd steps.
    lanes = [
        ("bits 0–1", "k + 0", "vpdpbusd  1", "#6ea8c4"),
        ("bits 2–3", "k + 64", "vpdpbusd  2", "#7bb39a"),
        ("bits 4–5", "k + 128", "vpdpbusd  3", "#c9a56a"),
        ("bits 6–7", "k + 192", "vpdpbusd  4", "#c98774"),
    ]
    prog = scene_local(t, S2 + 1.7, S2 + 4.2)
    for i, (bits, klab, step, color) in enumerate(lanes):
        on = ease(max(0.0, min(1.0, prog * 4 - i)))
        x = 1.0 + i * 4.5
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 3.15),
                4.1,
                2.4,
                boxstyle="round,pad=0.04,rounding_size=0.16",
                facecolor="#161a22",
                edgecolor=color,
                linewidth=1.2,
                alpha=on * a,
            )
        )
        ax.text(x + 2.05, 5.05, bits, ha="center", fontsize=13, color=MUTED, alpha=on * a)
        ax.text(x + 2.05, 4.45, klab, ha="center", fontsize=18, color=FG, alpha=on * a)
        ax.text(
            x + 2.05, 3.65, step, ha="center", fontsize=15, color=color, alpha=on * a, fontfamily="DejaVu Sans Mono"
        )
        if on > 0.05:
            ax.annotate(
                "",
                xy=(x + 2.05, 5.55),
                xytext=(x + 2.05, 6.15),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3, mutation_scale=12),
                alpha=on * a,
            )

    reveal = scene_local(t, S2 + 4.5, S2 + 5.6)
    ax.text(
        9.6,
        2.15,
        "VBMI decode:  vpmultishiftqb  +  AND",
        ha="center",
        fontsize=20,
        color=GOLD,
        alpha=reveal * a,
        fontfamily="DejaVu Sans Mono",
    )
    ax.text(
        9.6,
        1.45,
        "64 packed bytes become 64 code bytes in two instructions.",
        ha="center",
        fontsize=15,
        color=MUTED,
        alpha=reveal * a,
    )


def scene_spin(ax, t: float) -> None:
    a = scene_local(t, S3, S3 + 0.7)
    draw_header(
        ax,
        "A spin-wait thread pool.",
        "Persistent workers wait on an epoch, so a parallel GEMV skips a spawn and a futex wake.",
        a,
    )

    epoch = scene_local(t, S3 + 0.6, S3 + 1.4)
    # Animate an epoch counter ticking 0 → 1 → 2 once the scene is in.
    tick = 0
    if t >= S3 + 2.2:
        tick = 1
    if t >= S3 + 3.0:
        tick = 2
    ax.text(9.6, 6.55, "epoch", ha="center", fontsize=14, color=MUTED, alpha=epoch * a)
    ax.text(
        9.6,
        6.0,
        str(tick),
        ha="center",
        va="center",
        fontsize=32,
        color=GOLD,
        alpha=epoch * a,
        fontweight="medium",
        fontfamily="DejaVu Sans Mono",
    )

    workers = [
        ("worker 0", "rows 0–15"),
        ("worker 1", "rows 16–31"),
        ("worker 2", "rows 32–47"),
        ("worker 3", "rows 48–63"),
    ]
    prog = scene_local(t, S3 + 0.8, S3 + 2.4)
    for i, (name, rows) in enumerate(workers):
        on = ease(max(0.0, min(1.0, prog * 4 - i)))
        x = 1.0 + i * 4.55
        active = tick >= 1
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 2.7),
                4.2,
                2.7,
                boxstyle="round,pad=0.04,rounding_size=0.16",
                facecolor="#161a22",
                edgecolor=ACCENT if active else "#2a303c",
                linewidth=1.2,
                alpha=on * a,
            )
        )
        ax.text(x + 2.1, 4.7, name, ha="center", fontsize=18, color=FG, alpha=on * a)
        ax.text(x + 2.1, 3.95, "spin, then yield", ha="center", fontsize=13, color=MUTED, alpha=on * a)
        ax.text(x + 2.1, 3.2, rows, ha="center", fontsize=15, color=ACCENT if active else MUTED, alpha=on * a)

    note = scene_local(t, S3 + 3.1, S3 + 3.8)
    ax.text(
        9.6,
        1.7,
        "Handoff is a few cache-line writes.  Below 2¹⁹ weights the engine stays single-threaded.",
        ha="center",
        fontsize=15,
        color=GOLD,
        alpha=note * a,
    )


def _bar(ax, x, y, width, height, color, alpha):
    if width <= 0 or alpha <= 0:
        return
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            max(width, 0.04),
            height,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
    )


def scene_bars(ax, t: float) -> None:
    a = scene_local(t, S4, S4 + 0.7)
    draw_header(
        ax,
        "Measured p50 on one AVX-512 core.",
        "Xeon VM, pinned, 20,000 requests. Before is d39d557, run in the same session.",
        a,
    )

    # Scale so the 87.7 µs before-bar fills most of the row.
    x0 = 3.6
    scale = 11.6 / MEDIUM_BEFORE
    before_g = scene_local(t, S4 + 0.7, S4 + 2.6)
    after_g = scene_local(t, S4 + 2.7, S4 + 4.8)
    labels_g = scene_local(t, S4 + 4.9, S4 + 5.7)
    four_g = scene_local(t, S4 + 6.0, S4 + 7.2)

    # Legend under the kicker.
    legend = [("before (d39d557)", GRAY), ("after (Packed2)", GREEN)]
    lx = 3.6
    for lab, col in legend:
        _bar(ax, lx, 7.48, 0.38, 0.28, col, a)
        ax.text(lx + 0.5, 7.62, lab, va="center", fontsize=13, color=MUTED, alpha=a)
        lx += 4.3
    _bar(ax, lx, 7.48, 0.38, 0.28, ACCENT, four_g * a)
    ax.text(lx + 0.5, 7.62, "medium, 4 threads", va="center", fontsize=13, color=MUTED, alpha=four_g * a)

    groups = [
        ("tiny", TINY_BEFORE, TINY_AFTER, TINY_SPEEDUP, 6.35),
        ("small", SMALL_BEFORE, SMALL_AFTER, SMALL_SPEEDUP, 4.70),
        ("medium", MEDIUM_BEFORE, MEDIUM_AFTER, MEDIUM_SPEEDUP, 3.05),
    ]
    bh = 0.52
    for name, before, after, speedup, y_top in groups:
        ax.text(x0 - 0.25, y_top - 0.05, name, ha="right", va="center", fontsize=16, color=FG, alpha=a)

        _bar(ax, x0, y_top, before * scale * before_g, bh, GRAY, a)
        if before_g > 0.98:
            ax.text(
                x0 + before * scale + 0.2,
                y_top + bh / 2,
                f"{before:g} µs",
                va="center",
                fontsize=14,
                color=FG,
                alpha=a,
            )

        _bar(ax, x0, y_top - 0.62, after * scale * after_g, bh, GREEN, a)
        if after_g > 0.98:
            ax.text(
                x0 + after * scale + 0.2,
                y_top - 0.62 + bh / 2,
                f"{after:g} µs",
                va="center",
                fontsize=14,
                color=FG,
                alpha=a,
            )
        ax.text(
            17.6,
            y_top - 0.18,
            speedup,
            ha="right",
            va="center",
            fontsize=22,
            color=GOLD,
            alpha=labels_g * a,
            fontweight="medium",
        )

    # Medium, 4 threads, drawn under the medium after-bar.
    y4 = 1.45
    _bar(ax, x0, y4, MEDIUM_4T * scale * four_g, bh, ACCENT, a)
    ax.text(
        x0 - 0.25, y4 + bh / 2, "medium, 4 threads", ha="right", va="center", fontsize=13, color=MUTED, alpha=four_g * a
    )
    if four_g > 0.98:
        ax.text(
            x0 + MEDIUM_4T * scale + 0.2,
            y4 + bh / 2,
            f"{MEDIUM_4T:g} µs",
            va="center",
            fontsize=14,
            color=FG,
            alpha=a,
        )


def scene_ceiling(ax, t: float) -> None:
    a = scene_local(t, S5, S5 + 0.7)
    draw_header(
        ax,
        "Why not 100×.",
        "At batch 1 the GEMV is bound by memory bandwidth. 8× vs fp16 is a ceiling, not a measurement.",
        a,
    )
    grow = scene_local(t, S5 + 0.7, S5 + 2.4)

    ax.add_patch(
        patches.FancyBboxPatch(
            (1.2, 5.15),
            16 * grow,
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=GRAY,
            edgecolor="none",
            alpha=a,
        )
    )
    ax.text(1.2, 6.5, "fp16 weights", color=MUTED, fontsize=14, alpha=a)
    ax.text(
        1.2 + 16 * grow + 0.3,
        5.7,
        "16 bits / weight" if grow > 0.95 else "",
        va="center",
        color=FG,
        fontsize=14,
        alpha=a,
    )

    ax.add_patch(
        patches.FancyBboxPatch(
            (1.2, 3.15),
            2 * max(grow, 0.02),
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=ACCENT,
            edgecolor="none",
            alpha=a,
        )
    )
    ax.text(1.2, 4.5, "ternary weights", color=MUTED, fontsize=14, alpha=a)
    ax.text(
        1.2 + 2 * max(grow, 0.02) + 0.3,
        3.7,
        "2 bits / weight" if grow > 0.95 else "",
        va="center",
        color=FG,
        fontsize=14,
        alpha=a,
    )

    reveal = scene_local(t, S5 + 2.6, S5 + 3.6)
    ax.text(9.6, 2.15, "8×", ha="center", fontsize=64, color=GOLD, alpha=reveal * a, fontweight="medium")
    ax.text(
        9.6,
        1.25,
        "The 2–4× here comes from fewer instructions and less per-call overhead, not fewer bytes.",
        ha="center",
        fontsize=15,
        color=MUTED,
        alpha=reveal * a,
    )


def scene_ports(ax, t: float) -> None:
    a = scene_local(t, S6, S6 + 0.7)
    draw_header(
        ax,
        "CUDA is compile-checked only.",
        "The new kernels have not been run on a GPU. No GPU speedups are claimed.",
        a,
    )

    kernels = ["cp.async", "split-K", "LUT", "warp-spec", "hand PTX"]
    kprog = scene_local(t, S6 + 0.6, S6 + 1.8)
    for i, name in enumerate(kernels):
        on = ease(max(0.0, min(1.0, kprog * 5 - i)))
        x = 1.0 + i * 3.5
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 5.35),
                3.2,
                1.15,
                boxstyle="round,pad=0.03,rounding_size=0.14",
                facecolor="#161a22",
                edgecolor="#2a303c",
                linewidth=1.1,
                alpha=on * a,
            )
        )
        ax.text(x + 1.6, 5.92, name, ha="center", va="center", fontsize=16, color=FG, alpha=on * a)
    ax.text(
        9.6,
        4.85,
        "sm_80 to sm_120, 0 spills   ·   compile-checked only",
        ha="center",
        fontsize=14,
        color=MUTED,
        alpha=kprog * a,
    )

    langs = ["Zig", "Mojo", "Julia", "Futhark", "Halide", "Triton"]
    lprog = scene_local(t, S6 + 2.0, S6 + 3.3)
    ax.text(9.6, 3.95, "Same GEMV, six more languages.", ha="center", fontsize=18, color=FG, alpha=lprog * a)
    for i, name in enumerate(langs):
        on = ease(max(0.0, min(1.0, lprog * 6 - i)))
        x = 0.85 + i * 3.0
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 2.15),
                2.75,
                1.2,
                boxstyle="round,pad=0.03,rounding_size=0.14",
                facecolor="#161a22",
                edgecolor=ACCENT,
                linewidth=1.1,
                alpha=on * a,
            )
        )
        ax.text(x + 1.375, 2.75, name, ha="center", va="center", fontsize=17, color=ACCENT, alpha=on * a)
    ax.text(
        9.6,
        1.45,
        "Each port matches the Rust reference bit for bit, including the fp16 output.",
        ha="center",
        fontsize=15,
        color=GOLD,
        alpha=scene_local(t, S6 + 3.3, S6 + 3.9) * a,
    )


def scene_final(ax, t: float) -> None:
    a = scene_local(t, S7, S7 + 0.7)
    draw_header(ax, "", "", a)
    ax.text(0.5, 0.72, "Packed2  ·  spin pool", transform=ax.transAxes, ha="center", fontsize=20, color=MUTED, alpha=a)
    ax.text(
        0.5,
        0.54,
        "2–4× faster on CPU",
        transform=ax.transAxes,
        ha="center",
        fontsize=54,
        color=GOLD,
        alpha=a,
        fontweight="medium",
    )
    ax.text(
        0.5,
        0.38,
        "measured p50, one AVX-512 core  ·  no GPU numbers",
        transform=ax.transAxes,
        ha="center",
        fontsize=18,
        color=FG,
        alpha=a,
    )
    late = scene_local(t, S7 + 1.2, S7 + 1.9)
    ax.text(
        0.5,
        0.20,
        "github.com/shubhxho/ternary-gemm",
        transform=ax.transAxes,
        ha="center",
        fontsize=20,
        color=ACCENT,
        alpha=late * a,
    )
    ax.text(
        0.5,
        0.12,
        "./scripts/bench.sh --cpu-threads 4",
        transform=ax.transAxes,
        ha="center",
        fontsize=15,
        color=MUTED,
        alpha=late * a,
        fontfamily="DejaVu Sans Mono",
    )


def draw_frame(ax, t: float) -> None:
    ax.clear()
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.axis("off")
    ax.set_facecolor(BG)
    if t < S2:
        scene_pack(ax, t)
    elif t < S3:
        scene_packed2(ax, t)
    elif t < S4:
        scene_spin(ax, t)
    elif t < S5:
        scene_bars(ax, t)
    elif t < S6:
        scene_ceiling(ax, t)
    elif t < S7:
        scene_ports(ax, t)
    else:
        scene_final(ax, t)


def write_video() -> tuple[Path, Path, Path]:
    MEDIA.mkdir(parents=True, exist_ok=True)
    dark_rc()
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(0, 0, 1, 1)
    nframes = int(DURATION * FPS)

    def update(i):
        draw_frame(ax, i / FPS)
        return []

    mp4 = MEDIA / "cpu-speedup-explainer.mp4"
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"],
    )
    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / FPS, blit=False)
    anim.save(mp4, writer=writer, dpi=100)

    # Poster: fully revealed bar-chart frame (the measured result).
    draw_frame(ax, S5 - 0.15)
    poster = MEDIA / "cpu-speedup-explainer-poster.png"
    fig.savefig(poster, dpi=100, facecolor=BG)

    # Short looping GIF of the bar-chart grow, downscaled.
    gif = MEDIA / "cpu-speedup-explainer.gif"
    gif_start = S4 + 0.6
    gif_times = [gif_start + i / 12.0 for i in range(int(6 * 12))]
    gif_fig, gif_ax = plt.subplots(figsize=(10.667, 6.0), dpi=60)
    gif_fig.patch.set_facecolor(BG)
    gif_fig.subplots_adjust(0, 0, 1, 1)

    def update_gif(i):
        draw_frame(gif_ax, gif_times[i])
        return []

    ganim = FuncAnimation(gif_fig, update_gif, frames=len(gif_times), blit=False)
    ganim.save(gif, writer=PillowWriter(fps=12))
    plt.close(gif_fig)
    plt.close(fig)
    return mp4, poster, gif


def main() -> None:
    _register()
    argparse.ArgumentParser(description=__doc__).parse_args()
    mp4, poster, gif = write_video()
    for p in (mp4, poster, gif):
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
