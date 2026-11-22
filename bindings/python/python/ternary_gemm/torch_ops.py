"""PyTorch custom-op stub: `torch.ops.ternary_gemm.gemm_i8` backed by the CPU W2A8 GEMM.

Registered only when PyTorch (>= 2.4, `torch.library.custom_op`) is importable; importing this
module without torch is a no-op and `AVAILABLE` is False. The op is CPU-only and not
differentiable (inference); it round-trips through Python lists, so it is a correctness /
integration stub, not a fast path. A CUDA implementation would dispatch to the Rust GPU kernels.
"""

from __future__ import annotations

from . import CpuGemm, PackedTernary

try:
    import torch

    AVAILABLE = hasattr(torch, "library") and hasattr(torch.library, "custom_op")
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    AVAILABLE = False

_GEMMS: dict[int, tuple[CpuGemm, int]] = {}


def register_weights(packed: PackedTernary, threads: int = 1) -> int:
    """Register packed weights and return the integer handle the op takes."""
    handle = len(_GEMMS)
    _GEMMS[handle] = (CpuGemm(packed, threads), packed.n)
    return handle


if AVAILABLE:

    @torch.library.custom_op("ternary_gemm::gemm_i8", mutates_args=())
    def gemm_i8(x: torch.Tensor, handle: int) -> torch.Tensor:
        """`y = W . quant(x)` for x of shape [batch, k]; returns [batch, n] float32 (fp16 values)."""
        g, _ = _GEMMS[handle]
        x2 = x.detach().to(torch.float32).contiguous().cpu()
        batch = x2.shape[0]
        y = g(x2.reshape(-1).tolist(), batch)
        return torch.tensor(y, dtype=torch.float32).reshape(batch, -1).to(x.device)

    @gemm_i8.register_fake
    def _(x: torch.Tensor, handle: int) -> torch.Tensor:
        return x.new_empty((x.shape[0], _GEMMS[handle][1]), dtype=torch.float32)
