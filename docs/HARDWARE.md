# FPGA skeleton (`hw/`)

A starting point for running the ternary GEMV integer core on an FPGA. It is a skeleton: the
datapath is verified in simulation against the Rust crate, but nothing here has been
synthesized, placed or timed on a device, and no FPGA performance or power numbers are claimed.

## What is there

| file | what |
|---|---|
| `rtl/ternary_dot16.v` | 16 ternary weights x 16 int8 activations -> int32, **no multipliers**: per lane a 3-way mux (`-a`, `0`, `+a`), then a 4-level adder tree, one register stage |
| `rtl/ternary_mac.v` | streams `(weight word, 16 activations)` beats, accumulates a row, emits the int32 row sum two cycles after the `last` beat; II = 1, back-to-back rows without bubbles |
| `tb/tb_ternary_mac.v` | self-checking testbench, `$fatal` on any mismatch |
| `hls/ternary_gemv.{h,cpp}` | the same contract in HLS C++ (Vitis pragmas: `PIPELINE II=1`, unrolled 16-lane dot, AXI interfaces) |
| `hls/ternary_gemv_tb.cpp` | g++ software testbench (C simulation) on the same vectors |
| `../examples/hw_vectors.rs` | generates vectors: weights packed by `PackedTernary` (natural layout), expectations from the **unpacked** weights, so the bit layout is verified too |

Edge cases in the vectors: all `-1` x `-128` (most negative sum), all `+1` x `-128`, all `+1`
x `127`, all-zero weights, alternating signs, then random rows.

## Interface contract

* Weight word: the crate's natural 2-bit layout, weight `t` at bits `[2t+1:2t]` as code
  `w + 1` (the same words stored in `.tgm` files and `PackedTernary::natural`). Code 3 is
  invalid; the RTL treats it as 0 and `PackedTernary::from_natural` rejects it.
* Activations: 16 bytes per beat, activation `t` at bits `[8t+7:8t]`, two's complement int8 (the
  crate's absmax quantizer output).
* Output: `sum_k w[k] * q[k]` as int32. `|sum| <= 128 * K`, so int32 is exact for any K below
  2^24. The f32 epilogue (`acc / s * row_scale (+ residual)`) stays on the host, exactly as in
  `reference::gemm_i8`, which keeps the FPGA result bit-identical to the CPU and GPU paths.

## Running

```bash
make -C hw all          # vectors + Icarus + Verilator (lint -Wall + sim) + HLS C sim
make -C hw iverilog     # or one at a time
```

Needs `iverilog` (>= 11), `verilator` (>= 5, for `--binary`) and `g++`; the vectors target runs
`cargo`. CI runs `make -C hw all` on every push.

## Why this maps well to FPGAs (and to looping)

* Ternary MACs are adds/subtracts: no DSP slices are needed for the dot products, so the
  parallelism is bounded by LUTs and routing, not DSP count.
* 2-bit weights quadruple on-chip capacity versus int8. A looped transformer
  (`docs/LOOPED.md`) needs only one block's weights resident: d_model 512 is about 1 MB, which
  fits in the on-chip RAM of mid-size parts, so repeated loops never touch external memory.

## Next steps

1. Wrap `ternary_mac` in an AXI-Stream shell with a weight BRAM and an activation buffer;
   instantiate P parallel MACs sharing one activation stream (one row each).
2. Synthesize with Vivado / Vitis HLS for a concrete part and report LUT/FF/BRAM use, Fmax and
   measured power from the board (measured numbers only).
3. Move the absmax quantizer and the f32 epilogue on-chip (needs one divider per row or a
   reciprocal table).
