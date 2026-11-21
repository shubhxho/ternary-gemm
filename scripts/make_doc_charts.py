#!/usr/bin/env python3
"""Generate the README charts in docs/img/.

Every chart here is computed from shapes and arithmetic only. Charts with latency or throughput are
THEORETICAL (bytes / peak bandwidth); nothing in this script is a measurement.
Peak bandwidth: RTX 5060 spec, 128-bit GDDR7 at 28 Gbps = 448 GB/s.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(OUT, exist_ok=True)
PEAK = 448.0  # GB/s
SHAPES = [(2560, 2560, "q / o\n2560x2560"), (640, 2560, "k / v\n640x2560"), (6912, 2560, "gate / up\n6912x2560"), (2560, 6912, "down\n2560x6912")]
LAYER = [(2560, 2560), (640, 2560), (640, 2560), (2560, 2560), (6912, 2560), (6912, 2560), (2560, 6912)]
C_F16, C_I8, C_T = "#9aa0a6", "#5f6368", "#1a73e8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150, "savefig.bbox": "tight",
})

def note(ax, text):
    ax.text(0.0, -0.22, text, transform=ax.transAxes, fontsize=8.5, color="#5f6368", va="top")

def bytes_f16(n, k): return n * k * 2
def bytes_i8(n, k): return n * k + n * 4
def bytes_t(n, k): return n * k // 4 + n * 4

# (a) bytes per weight matrix (arithmetic, exact)
fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(SHAPES)); w = 0.26
for off, fn, c, lab in [(-w, bytes_f16, C_F16, "fp16 (2 B/weight)"), (0, bytes_i8, C_I8, "int8 (1 B/weight)"), (w, bytes_t, C_T, "ternary 2-bit (0.25 B/weight)")]:
    vals = [fn(n, k) / 1e6 for n, k, _ in SHAPES]
    bars = ax.bar([i + off for i in x], vals, w, color=c, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([s[2] for s in SHAPES])
ax.set_ylabel("weight bytes per matrix (MB)")
ax.set_title("Theoretical (not measured): weight bytes per BitNet-2B matrix")
ax.legend(frameon=False)
note(ax, "Bytes = N*K*bytes_per_weight (+4 B per row scale for int8/ternary). Not a measurement; it is what every decode step must read.")
fig.savefig(os.path.join(OUT, "bytes_per_matrix.png")); plt.close(fig)

# (b) roofline latency floor
fig, ax = plt.subplots(figsize=(8, 4.2))
for off, fn, c, lab in [(-0.2, bytes_f16, C_F16, "fp16 floor"), (0.2, bytes_t, C_T, "ternary floor")]:
    vals = [(fn(n, k) + k * 2 + n * 2) / PEAK / 1e3 for n, k, _ in SHAPES]
    bars = ax.bar([i + off for i in x], vals, 0.38, color=c, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f} us", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([s[2] for s in SHAPES])
ax.set_ylabel("minimum latency at batch 1 (us)")
ax.set_title("Theoretical (not measured): DRAM roofline latency floor, RTX 5060 @ 448 GB/s")
ax.legend(frameon=False)
note(ax, "t_min = (weights + activations + outputs) bytes / 448 GB/s. Real kernels are slower (launch overhead, tail effects). NOT MEASURED.")
fig.savefig(os.path.join(OUT, "roofline_latency_floor.png")); plt.close(fig)

# (c) tokens/s ceiling for the 30-layer linear stack
f16 = sum(bytes_f16(n, k) for n, k in LAYER) * 30
tern = sum(bytes_t(n, k) for n, k in LAYER) * 30
i8 = sum(bytes_i8(n, k) for n, k in LAYER) * 30
fig, ax = plt.subplots(figsize=(7, 3.8))
labels = ["fp16", "int8", "ternary 2-bit"]
vals = [PEAK * 1e9 / b for b in (f16, i8, tern)]
bars = ax.barh(labels, vals, color=[C_F16, C_I8, C_T])
for b, v, by in zip(bars, vals, (f16, i8, tern)):
    ax.text(v + 10, b.get_y() + b.get_height() / 2, f"{v:.0f} tok/s  ({by / 1e9:.2f} GB/token)", va="center", fontsize=9)
ax.set_xlim(0, max(vals) * 1.45)
ax.set_xlabel("tokens/s ceiling, batch 1 (linear layers only)")
ax.set_title("Theoretical (not measured): BitNet-2B linear-stack ceiling @ 448 GB/s")
note(ax, "Ceiling = 448 GB/s / weight bytes per token (30 layers x q,k,v,o,gate,up,down). Ignores attention, KV cache, lm_head, launches. NOT MEASURED.")
fig.savefig(os.path.join(OUT, "tokens_per_s_ceiling.png")); plt.close(fig)

# (d) architecture / data-flow diagram of the fused kernel
fig, ax = plt.subplots(figsize=(12.5, 6.4))
ax.set_xlim(0, 124); ax.set_ylim(0, 62); ax.axis("off")
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#3c4043", lw=1))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + 1.2, y + h - 6.2, body, ha="left", va="top", fontsize=8.6, family="DejaVu Sans Mono")
def arrow(x0, y0, x1, y1, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#3c4043"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, label, ha="center", fontsize=8.5, color="#3c4043")
ax.text(62, 60.5, "ternary_i8_fused: one launch per linear layer", ha="center", fontsize=14, weight="bold")
box(1, 33, 35, 23, "Packed weights (DRAM)", "code c = w + 1 in {0,1,2}\n16 codes / u32, 2 bits each\nweight 16kb+4j+i at bits 8i+2j\nchunk = 128 words (2048 w)\nlane l loads ONE uint4 per row\n(words for k-blocks q*L+l)\n0.25 B / weight", "#e8f0fe")
box(1, 4, 35, 24, "x  (fp16, [batch][K])", "per token, whole block:\n  ss = sum x^2   (RMSNorm)\n  m  = max |x*gamma|\n  inv = rsqrt(ss/K + eps)\n  s  = 127 / (m*inv)\n  q  = rint(x*gamma*inv*s)\n  sum_q = sum q", "#fef7e0")
box(45, 4, 31, 24, "Prologue -> shared mem", "int8 q[batch][K]\ns[batch], sum_q[batch]\n(each block re-quantizes:\n small vector, L2-resident;\n saves a launch + a\n global round trip)", "#fce8e6")
box(45, 33, 31, 23, "Decode in registers", "for j in 0..4:\n  c4 = (word >> 2j)\n       & 0x03030303\n-> 4 consecutive codes\n   as int8 bytes\nshift + AND per 4 weights,\nno lookup table", "#e6f4ea")
box(85, 18, 37, 38, "dp4a loop (1 warp = R rows)", "acc += dp4a(c4, x4)  # 4 MACs\nweights decoded once,\nreused for all batch tokens\nactivation reads contiguous\n(conflict-free in smem)\n\nwarp shuffle-xor reduction\n\nEpilogue (lane 0):\n y = (acc - sum_q) / s\n     * row_scale (+ residual)\n -> fp16, IEEE _rn ops:\n bit-exact vs CPU reference", "#f3e8fd")
arrow(37.5, 44.5, 44, 44.5, "uint4")
arrow(37.5, 16, 44, 16, "fp16")
arrow(77.5, 44.5, 84, 44.5, "codes")
arrow(77.5, 16, 84, 24, "int8")
ax.text(103.5, 13, "y  fp16 [batch][N]", ha="center", fontsize=10, weight="bold")
arrow(103.5, 18, 103.5, 15)
ax.text(1, 0.5, "Offset trick: sum c*x = sum w*x + sum x, so the kernel subtracts sum_q once per token instead of handling signs per weight.", fontsize=8.5, color="#5f6368")
fig.savefig(os.path.join(OUT, "fused_kernel_dataflow.png")); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
