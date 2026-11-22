#!/usr/bin/env python3
"""Generate the tensor-core explainer video, poster, GIF, and report figures.

The video is a 1920x1080 dark-theme animation. Report figures are white-background,
minimal, and meant for the Connectionism-style technical report. Nothing in either
output is a measured GPU result.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, patheffects
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
FIGURES = ROOT / "docs" / "reports" / "figures"

# Measured CPU numbers (one core, Xeon VM, AVX-512 VNNI vs dense f32 AVX2).
CPU_TERNARY_US = 18.30
CPU_DENSE_US = 147.78
CPU_SPEEDUP = 8.1  # 147.78 / 18.30

IBM = "/usr/share/fonts/truetype/sand-box/google/IBM Plex Sans/IBMPlexSans-VariableFont_wdth,wght.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


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


# ---------------------------------------------------------------------------
# Report figures (white, quiet typography)
# ---------------------------------------------------------------------------

def setup_report_rc():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#222",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#222",
            "ytick.color": "#222",
            "text.color": "#222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )


def fig_packing(path: Path) -> None:
    setup_report_rc()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=180)
    # Left: 16 ternary weights
    ax = axes[0]
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color="#ccc", lw=0.6)
    weights = [1, -1, 0, 1, 1, 0, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1]
    colors = {1: "#1b4d3e", 0: "#b0b0b0", -1: "#7a2e1f"}
    ax.bar(range(16), weights, color=[colors[w] for w in weights], width=0.72, lw=0)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i) for i in range(16)], fontsize=8)
    ax.set_xlabel("weight index in a 16-wide block")
    ax.set_title("Ternary weights in {−1, 0, +1}", pad=8)
    # Right: bit packing
    ax = axes[1]
    ax.set_xlim(-0.5, 32.5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Packed into 32 bits (2 bits each)", pad=8)
    codes = [w + 1 for w in weights]
    bit_colors = ["#7a2e1f", "#d8d8d8", "#1b4d3e"]  # 0,1,2
    for i, c in enumerate(codes):
        x = i * 2
        rect = patches.FancyBboxPatch(
            (x, 1.4),
            1.9,
            1.2,
            boxstyle="square,pad=0",
            facecolor=bit_colors[c],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(x + 0.95, 2.0, format(c, "02b"), ha="center", va="center", fontsize=7, color="#111")
        ax.text(x + 0.95, 0.85, str(i), ha="center", va="center", fontsize=6, color="#666")
    ax.text(16, 3.4, "one u32  ·  16 codes  ·  c = w + 1", ha="center", fontsize=9, color="#444")
    ax.text(0.2, 0.25, "bit 0", ha="left", fontsize=7, color="#888")
    ax.text(31.8, 0.25, "bit 31", ha="right", fontsize=7, color="#888")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_roofline(path: Path) -> None:
    setup_report_rc()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=180)
    # Schematic roofline. Axes are qualitative; peaks are not measured on this work.
    # Intensity on log x, performance on log y, both in arbitrary units of "fp16-GEMV = 1".
    xs = [10 ** p for p in [i / 40 for i in range(-48, 80)]]
    # Memory roof for fp16: P = I (normalized so the fp16 GEMV sits on this line).
    # Ternary weight traffic is 8× smaller, so the memory roof is 8× higher.
    # A schematic tensor-core compute roof is drawn at P = 8; ternary hits it at I = 1,
    # fp16 at I = 8. This is a diagram, not a measured roofline.
    mem_fp16 = [i for i in xs]
    mem_tern = [8 * i for i in xs]
    compute = [8 for _ in xs]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(xs, mem_fp16, color="#666", lw=1.2, label="fp16 bandwidth roof (schematic)")
    ax.plot(xs, mem_tern, color="#1b4d3e", lw=1.4, label="ternary bandwidth roof (8× weight traffic)")
    ax.plot(xs, compute, color="#222", lw=1.2, linestyle=(0, (3, 2)), label="tensor-core compute roof (schematic)")
    ax.scatter([0.25], [2.0], s=40, color="#1b4d3e", zorder=5)
    ax.annotate(
        "batch 1 GEMV\nmemory-bound\n8× ceiling",
        (0.25, 2.0),
        xytext=(0.12, 4.8),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#1b4d3e", lw=0.6),
    )
    ax.scatter([1.0], [8.0], s=40, color="#1b4d3e", zorder=5)
    ax.annotate(
        "large-batch GEMM\ncompute-bound\n8× no longer holds",
        (1.0, 8.0),
        xytext=(1.4, 12.5),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#222", lw=0.6),
    )
    ax.set_xlim(0.08, 40)
    ax.set_ylim(0.15, 20)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte), schematic")
    ax.set_ylabel("Attainable throughput, schematic")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Why the 8× figure is a batch-1 bandwidth ceiling", pad=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_cpu(path: Path) -> None:
    setup_report_rc()
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=180)
    labels = ["dense f32 AVX2", "ternary SIMD\n(AVX-512 VNNI)"]
    vals = [CPU_DENSE_US, CPU_TERNARY_US]
    colors = ["#888", "#1b4d3e"]
    bars = ax.barh(labels, vals, color=colors, height=0.55, lw=0)
    for bar, v in zip(bars, vals):
        ax.text(v + 2, bar.get_y() + bar.get_height() / 2, f"{v:.2f} µs", va="center", fontsize=10)
    ax.set_xlim(0, 180)
    ax.set_xlabel("End-to-end latency, p50 (µs)")
    ax.set_title("Small signal model 256-1024-1024-16, one core, Xeon VM", pad=10)
    ax.text(
        0,
        -0.28,
        f"Measured. {CPU_SPEEDUP:.1f}×. Not a GPU number. GPU tensor-core measurements are pending.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_dataflow(path: Path) -> None:
    setup_report_rc()
    fig, ax = plt.subplots(figsize=(9.2, 3.2), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#f4f4f0"):
        r = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=fc, edgecolor="#222", linewidth=0.8,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.5, color="#444")

    box(0.15, 0.7, 2.2, 1.6, "2-bit weights", "natural layout, 16 codes / u32", "#f3efe7")
    box(2.75, 0.7, 2.2, 1.6, "Unpack in registers", "w = c − 1 → int8 / fp16", "#eaf0ea")
    box(5.35, 0.7, 2.2, 1.6, "MMA fragments", "A: 16×16   B: 16×8", "#e7eef3")
    box(7.95, 0.7, 1.9, 1.6, "mma.sync", "m16n8k16, sm_80+", "#efe7e7")
    for x in (2.35, 4.95, 7.55):
        ax.annotate(
            "",
            xy=(x + 0.38, 1.5),
            xytext=(x, 1.5),
            arrowprops=dict(arrowstyle="->", color="#222", lw=0.9),
        )
    ax.set_title("Tensor-core path, one K-tile of a warp", pad=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report_figures() -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [
        FIGURES / "fig1-packing.png",
        FIGURES / "fig2-roofline.png",
        FIGURES / "fig3-cpu-latency.png",
        FIGURES / "fig4-dataflow.png",
    ]
    fig_packing(paths[0])
    fig_roofline(paths[1])
    fig_cpu(paths[2])
    fig_dataflow(paths[3])
    return paths


# ---------------------------------------------------------------------------
# Explainer video (dark theme, 1920×1080)
# ---------------------------------------------------------------------------

BG = "#0c0e12"
FG = "#e8eaed"
MUTED = "#8b919a"
ACCENT = "#8ec5e0"
GREEN = "#8fbf9f"
RED = "#e07a5f"
GOLD = "#e2c07a"

DURATION = 32.0
FPS = 30
# Scene starts (inclusive).
S1, S2, S3, S4, S5 = 0.0, 6.0, 12.5, 20.0, 26.0


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
    ax.text(0.04, 0.94, "TERNARY-GEMM", transform=ax.transAxes, fontsize=13,
            color=ACCENT, alpha=alpha, fontweight="medium")
    ax.text(0.04, 0.88, title, transform=ax.transAxes, fontsize=32,
            color=FG, alpha=alpha, fontweight="medium")
    ax.text(0.04, 0.81, kicker, transform=ax.transAxes, fontsize=16,
            color=MUTED, alpha=alpha)


def scene_pack(ax, t: float) -> None:
    a = scene_local(t, S1, S1 + 0.8)
    draw_header(ax, "A ternary weight is two bits.",
                "Each weight lives in {−1, 0, +1}. Sixteen of them fill one 32-bit word.", a)
    weights = [1, -1, 0, 1, 1, 0, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1]
    colors = {1: GREEN, 0: "#3d4450", -1: RED}
    labels = {1: "+1", 0: "0", -1: "−1"}
    # Weight chips
    appear = scene_local(t, S1 + 0.4, S1 + 2.2)
    for i, w in enumerate(weights):
        local = ease(max(0, min(1, appear * 16 - i)) if appear < 1 else 1)
        x = 0.8 + i * 1.15
        y = 5.6
        r = patches.FancyBboxPatch((x, y), 1.0, 1.15, boxstyle="round,pad=0.02,rounding_size=0.12",
                                   facecolor=colors[w], edgecolor="none", alpha=local * a)
        ax.add_patch(r)
        ax.text(x + 0.5, y + 0.58, labels[w], ha="center", va="center", fontsize=13,
                color="white", alpha=local * a)
    ax.text(9.6, 5.15, "16 weights", ha="center", color=MUTED, fontsize=12, alpha=appear * a)

    # Pack into bits
    pack = scene_local(t, S1 + 2.4, S1 + 4.2)
    ax.text(9.6, 3.55, "one u32   ·   2 bits each   ·   c = w + 1", ha="center",
            color=MUTED, fontsize=13, alpha=pack * a)
    codes = [w + 1 for w in weights]
    bit_c = {0: RED, 1: "#3d4450", 2: GREEN}
    for i, c in enumerate(codes):
        x = 0.8 + i * 1.15
        y = 2.15 + (1 - pack) * 2.6
        r = patches.FancyBboxPatch((x, y), 1.0, 1.0, boxstyle="round,pad=0.01,rounding_size=0.08",
                                   facecolor=bit_c[c], edgecolor="#0c0e12", linewidth=0.8,
                                   alpha=max(pack, 0.05) * a)
        ax.add_patch(r)
        ax.text(x + 0.5, y + 0.5, format(c, "02b"), ha="center", va="center",
                fontsize=10, color="white", alpha=pack * a)
    ax.text(9.6, 1.5, "fp16 would have used 256 bits for the same 16 weights.",
            ha="center", color=GOLD, fontsize=15, alpha=scene_local(t, S1 + 4.4, S1 + 5.2) * a)


def scene_traffic(ax, t: float) -> None:
    a = scene_local(t, S2, S2 + 0.7)
    draw_header(ax, "Eight times less weight traffic than fp16.",
                "At batch 1 the GEMV is memory-bound on the weights. That 8× is a ceiling, not a measurement.", a)
    # Two memory bars
    grow = scene_local(t, S2 + 0.8, S2 + 3.0)
    # fp16: 16 units, ternary: 2 units
    ax.add_patch(patches.FancyBboxPatch((1.2, 5.1), 16 * grow, 1.15,
                                       boxstyle="round,pad=0.02,rounding_size=0.1",
                                       facecolor="#4a5160", edgecolor="none", alpha=a))
    ax.text(1.2, 6.45, "fp16 weights", color=MUTED, fontsize=14, alpha=a)
    ax.text(1.2 + 16 * grow + 0.3, 5.65, "16 bits / weight" if grow > 0.95 else "",
            va="center", color=FG, fontsize=14, alpha=a)

    ax.add_patch(patches.FancyBboxPatch((1.2, 3.1), 2 * max(grow, 0.02), 1.15,
                                       boxstyle="round,pad=0.02,rounding_size=0.1",
                                       facecolor=ACCENT, edgecolor="none", alpha=a))
    ax.text(1.2, 4.45, "ternary weights", color=MUTED, fontsize=14, alpha=a)
    ax.text(1.2 + 2 * max(grow, 0.02) + 0.3, 3.65, "2 bits / weight" if grow > 0.95 else "",
            va="center", color=FG, fontsize=14, alpha=a)

    reveal = scene_local(t, S2 + 3.2, S2 + 4.4)
    ax.text(9.6, 2.05, "8×", ha="center", fontsize=72, color=GOLD, alpha=reveal * a,
            fontweight="medium")
    ax.text(9.6, 1.15, "theoretical bandwidth ceiling at batch 1",
            ha="center", fontsize=16, color=MUTED, alpha=reveal * a)


def scene_unpack(ax, t: float) -> None:
    a = scene_local(t, S3, S3 + 0.7)
    draw_header(ax, "Unpack into tensor-core fragments.",
                "Then issue mma.sync.aligned.m16n8k16 on sm_80, sm_89 and sm_120.", a)
    stages = [
        (0.7, "2-bit word", "natural layout"),
        (5.3, "Unpack", "registers / smem\nw = c − 1"),
        (9.9, "Fragments", "A 16×16   B 16×8\nint8 or fp16"),
        (14.5, "mma.sync", "m16n8k16\nW2A8 / W2A16"),
    ]
    prog = scene_local(t, S3 + 0.8, S3 + 5.5)
    for i, (x, title, sub) in enumerate(stages):
        on = ease(max(0.0, min(1.0, prog * 4 - i)))
        ax.add_patch(patches.FancyBboxPatch((x, 2.3), 3.8, 3.3,
                                           boxstyle="round,pad=0.04,rounding_size=0.18",
                                           facecolor="#161a22", edgecolor=ACCENT if i == 3 else "#2a303c",
                                           linewidth=1.2, alpha=on * a))
        ax.text(x + 1.9, 4.7, title, ha="center", fontsize=18, color=FG, alpha=on * a)
        ax.text(x + 1.9, 3.5, sub, ha="center", fontsize=13, color=MUTED, alpha=on * a, linespacing=1.5)
        if i < 3:
            ax.annotate("", xy=(x + 4.15, 3.95), xytext=(x + 3.85, 3.95),
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4,
                                        mutation_scale=14),
                        alpha=on * a)
    ax.text(9.6, 1.4, "Prior art on this path:  TriRun  ·  BitBLAS W2A8",
            ha="center", fontsize=15, color=GOLD, alpha=scene_local(t, S3 + 5.6, S3 + 6.4) * a)


def scene_cpu(ax, t: float) -> None:
    a = scene_local(t, S4, S4 + 0.7)
    draw_header(ax, "Measured on the CPU, not the GPU.",
                "Small signal model 256-1024-1024-16. One core, Xeon VM. 8.1×.", a)
    grow = scene_local(t, S4 + 0.8, S4 + 3.0)
    # Horizontal bars, scale to 147.78
    scale = 12.5 / CPU_DENSE_US
    ax.add_patch(patches.FancyBboxPatch((3.2, 5.15), CPU_DENSE_US * scale * grow, 1.15,
                                       boxstyle="round,pad=0.02,rounding_size=0.1",
                                       facecolor="#4a5160", edgecolor="none", alpha=a))
    ax.text(3.05, 5.72, "dense f32  AVX2", ha="right", va="center", fontsize=14, color=MUTED, alpha=a)
    if grow > 0.98:
        ax.text(3.2 + CPU_DENSE_US * scale + 0.25, 5.72, f"{CPU_DENSE_US:.2f} µs",
                va="center", fontsize=16, color=FG, alpha=a)

    ax.add_patch(patches.FancyBboxPatch((3.2, 3.35), CPU_TERNARY_US * scale * grow, 1.15,
                                       boxstyle="round,pad=0.02,rounding_size=0.1",
                                       facecolor=GREEN, edgecolor="none", alpha=a))
    ax.text(3.05, 3.92, "ternary SIMD\nAVX-512 VNNI", ha="right", va="center", fontsize=13, color=MUTED, alpha=a)
    if grow > 0.98:
        ax.text(3.2 + CPU_TERNARY_US * scale + 0.25, 3.92, f"{CPU_TERNARY_US:.2f} µs",
                va="center", fontsize=16, color=FG, alpha=a)

    ax.text(9.6, 1.7, "This is a CPU result. GPU tensor-core latency has not been measured.",
            ha="center", fontsize=15, color=GOLD, alpha=scene_local(t, S4 + 3.2, S4 + 4.2) * a)


def scene_final(ax, t: float) -> None:
    a = scene_local(t, S5, S5 + 0.8)
    draw_header(ax, "", "", a)
    ax.text(0.5, 0.70, "Tensor-core path", transform=ax.transAxes,
            ha="center", fontsize=22, color=MUTED, alpha=a)
    ax.text(0.5, 0.52, "up to 8× vs fp16", transform=ax.transAxes,
            ha="center", fontsize=54, color=GOLD, alpha=a, fontweight="medium")
    ax.text(0.5, 0.36, "theoretical ceiling, GPU benchmark pending",
            transform=ax.transAxes, ha="center", fontsize=20, color=FG, alpha=a)
    ax.text(0.5, 0.18, "./scripts/bench.sh --tensor-cores",
            transform=ax.transAxes, ha="center", fontsize=16, color=ACCENT,
            alpha=scene_local(t, S5 + 1.6, S5 + 2.4) * a,
            fontfamily="DejaVu Sans Mono")
    ax.text(0.5, 0.10, "github.com/shubhxho/ternary-gemm",
            transform=ax.transAxes, ha="center", fontsize=13, color=MUTED,
            alpha=scene_local(t, S5 + 1.6, S5 + 2.4) * a)


def draw_frame(ax, t: float) -> None:
    ax.clear()
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.axis("off")
    ax.set_facecolor(BG)
    # Cross-fade between scenes by drawing the active one. Hard cuts are fine
    # at these scene boundaries; each scene fades itself in.
    if t < S2:
        scene_pack(ax, t)
    elif t < S3:
        scene_traffic(ax, t)
    elif t < S4:
        scene_unpack(ax, t)
    elif t < S5:
        scene_cpu(ax, t)
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

    mp4 = MEDIA / "tensor-core-explainer.mp4"
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"],
    )
    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / FPS, blit=False)
    anim.save(mp4, writer=writer, dpi=100)

    # Poster: last frame (final card)
    draw_frame(ax, DURATION - 0.05)
    poster = MEDIA / "tensor-core-explainer-poster.png"
    fig.savefig(poster, dpi=100, facecolor=BG)
    # Also a slightly earlier still for the README (final typography fully in)
    draw_frame(ax, DURATION - 0.4)
    fig.savefig(poster, dpi=100, facecolor=BG)

    # Short looping GIF of the packing + 8x scenes, downscaled.
    gif = MEDIA / "tensor-core-explainer.gif"
    # 6 seconds from the traffic scene + a moment of the final card, 12 fps, 640 wide.
    gif_times = [S2 + i / 12.0 for i in range(int(6 * 12))]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    args = parser.parse_args()
    if not args.video_only:
        paths = write_report_figures()
        print("figures:")
        for p in paths:
            print(" ", p, p.stat().st_size)
    if not args.figures_only:
        mp4, poster, gif = write_video()
        for p in (mp4, poster, gif):
            print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
