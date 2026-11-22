"""Ternary (BitNet b1.58) W2A8 GEMM, signal models and a looped transformer (CPU bindings).

Arrays are flat, row-major Python lists. See `ternary_gemm.torch_ops` for PyTorch integration.
"""

from ._ternary_gemm import CpuGemm, LoopedTransformer, PackedTernary, SignalModel, __version__

__all__ = ["CpuGemm", "LoopedTransformer", "PackedTernary", "SignalModel", "__version__"]
