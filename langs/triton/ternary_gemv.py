"""Ternary W2A8 GEMV in Triton.

Same contract as the Rust reference (``reference::gemm_i8``) for batch 1:

* weights in the ternary-gemm *natural* layout: weight ``c`` of a row at word ``c // 16``,
  bits ``2 * (c % 16)``, stored as the code ``w + 1``;
* int8 activations ``q`` with per-token scale ``s``;
* ``y[r] = fp16(acc[r] / s * row_scale[r])`` with ``acc[r] = sum_c w[r][c] * q[c]`` in int32,
  one IEEE round-to-nearest divide (``tl.math.div_rn``; Triton's plain ``/`` on fp32 lowers to an
  approximate divide on NVIDIA GPUs) and one multiply.

Runs on a GPU, or on the CPU through Triton's interpreter (``TRITON_INTERPRET=1``), which is how
the tests check it on machines without CUDA. ``compile_ptx`` lowers it for an NVIDIA target
(e.g. sm_120) without needing a GPU.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def ternary_gemv_kernel(
    w_ptr,  # int32 [N, KW] natural-layout words (bit pattern of the u32 words)
    q_ptr,  # int8 [KW * 16]
    scale_ptr,  # fp32 [N]
    acc_ptr,  # int32 [N] (integer row sums, for testing)
    out_ptr,  # fp16 [N]
    s,  # fp32 activation scale
    N,
    KW,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    shifts = tl.arange(0, 16) * 2
    lane = tl.arange(0, 16)
    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for w0 in range(0, KW, BLOCK_W):
        cols = w0 + tl.arange(0, BLOCK_W)
        cmask = cols < KW
        words = tl.load(
            w_ptr + rows[:, None] * KW + cols[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0,
        )
        # (BLOCK_N, BLOCK_W, 16) codes. An arithmetic shift is fine: the mask keeps the low 2 bits.
        codes = (words[:, :, None] >> shifts[None, None, :]) & 3
        w = codes - 1
        # Masked-off columns load q = 0, so whatever the padded codes decode to adds nothing.
        q = tl.load(q_ptr + cols[:, None] * 16 + lane[None, :], mask=cmask[:, None], other=0).to(tl.int32)
        acc += tl.sum(tl.sum(w * q[None, :, :], axis=2), axis=1)
    tl.store(acc_ptr + rows, acc, mask=rmask)
    t = acc.to(tl.float32)
    t = tl.math.div_rn(t, s)
    t = t * tl.load(scale_ptr + rows, mask=rmask, other=0.0)
    tl.store(out_ptr + rows, t.to(tl.float16), mask=rmask)


def ternary_gemv(words, q, row_scale, s, block_n: int = 16, block_w: int = 16):
    """Run the kernel. ``words``: int32 tensor [N, KW]; ``q``: int8 [KW*16]; ``row_scale``: fp32 [N].

    Returns ``(acc int32 [N], y fp16 [N])`` on the inputs' device.
    """
    import torch

    n, kw = words.shape
    acc = torch.empty(n, dtype=torch.int32, device=words.device)
    out = torch.empty(n, dtype=torch.float16, device=words.device)
    grid = (triton.cdiv(n, block_n),)
    # No FP contraction: the epilogue's divide and multiply must round separately.
    ternary_gemv_kernel[grid](
        words, q, row_scale, acc, out, float(s), n, kw, BLOCK_N=block_n, BLOCK_W=block_w, enable_fp_fusion=False
    )
    return acc, out


def compile_ptx(arch: int = 120, block_n: int = 16, block_w: int = 16) -> str:
    """Lower the kernel to PTX for ``sm_<arch>`` without a GPU (Triton's own LLVM + ptxas)."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sig = {
        "w_ptr": "*i32",
        "q_ptr": "*i8",
        "scale_ptr": "*fp32",
        "acc_ptr": "*i32",
        "out_ptr": "*fp16",
        "s": "fp32",
        "N": "i32",
        "KW": "i32",
        "BLOCK_N": "constexpr",
        "BLOCK_W": "constexpr",
    }
    src = ASTSource(
        fn=ternary_gemv_kernel,
        signature=sig,
        constexprs={"BLOCK_N": block_n, "BLOCK_W": block_w},
    )
    k = triton.compile(src, target=GPUTarget("cuda", arch, 32), options={"enable_fp_fusion": False})
    return k.asm["ptx"]
