# Autotune: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs, PTX compute_120)

Profile: GeForce RTX 5090 (model limits). All times are **measured** medians on this device (CUDA events, quantization included); the `projected` column is the roofline **projection** from public specs (`ternary_gemm::tuning::project_gemv`), not a measurement.

| N x K | batch | best config | kernel regs | best (us) | default (us) | speedup | projected bound (us) | model occupancy | candidates |
|---|---|---|---|---|---|---|---|---|---|
| 2560 x 2560 | 1 | 4w x 4r fused (`ternary_i8_fused_b1_r4`) | 48 | 6.17 | 8.20 | 1.33x | 0.92 (dram) | 40 warps/SM (registers) | 16 |
| 2560 x 2560 | 2 | 8w x 2r fused (`ternary_i8_fused_b2_r2`) | 61 | 8.22 | 10.25 | 1.25x | 0.92 (dram) | 32 warps/SM (registers) | 16 |
| 2560 x 2560 | 4 | 8w x 2r fused (`ternary_i8_fused_b4_r2`) | 64 | 10.29 | 12.31 | 1.20x | 0.93 (dram) | 32 warps/SM (registers) | 15 |
| 2560 x 2560 | 8 | 4w x 1r (`ternary_i8_b8_r1`) | 54 | 14.34 | 14.36 | 1.00x | 1.00 (int) | 36 warps/SM (registers) | 15 |
| 2560 x 2560 | 16 | 4w x 1r (`ternary_i8_b16_r1`) | 64 | 18.44 | 18.44 | 1.00x | 2.00 (int) | 32 warps/SM (registers) | 14 |
| 640 x 2560 | 1 | 8w x 1r fused (`ternary_i8_fused_b1_r1`) | 40 | 4.12 | 8.22 | 1.99x | 0.23 (dram) | 48 warps/SM (warps) | 16 |
| 640 x 2560 | 2 | 8w x 1r fused (`ternary_i8_fused_b2_r1`) | 40 | 6.17 | 10.25 | 1.66x | 0.23 (dram) | 48 warps/SM (warps) | 16 |
| 640 x 2560 | 4 | 4w x 1r (`ternary_i8_b4_r1`) | 46 | 10.25 | 10.27 | 1.00x | 0.24 (dram) | 40 warps/SM (registers) | 15 |
| 640 x 2560 | 8 | 4w x 1r (`ternary_i8_b8_r1`) | 54 | 12.31 | 12.31 | 1.00x | 0.25 (int) | 36 warps/SM (registers) | 14 |
| 640 x 2560 | 16 | 4w x 1r (`ternary_i8_b16_r1`) | 64 | 16.39 | 16.39 | 1.00x | 0.50 (int) | 32 warps/SM (registers) | 13 |
| 6912 x 2560 | 1 | 8w x 2r fused (`ternary_i8_fused_b1_r2`) | 40 | 6.17 | 8.22 | 1.33x | 2.49 (dram) | 48 warps/SM (warps) | 16 |
| 6912 x 2560 | 2 | 8w x 2r fused (`ternary_i8_fused_b2_r2`) | 61 | 8.20 | 10.25 | 1.25x | 2.49 (dram) | 32 warps/SM (registers) | 15 |
| 6912 x 2560 | 4 | 4w x 1r (`ternary_i8_b4_r1`) | 46 | 12.30 | 14.34 | 1.17x | 2.49 (dram) | 40 warps/SM (registers) | 15 |
| 6912 x 2560 | 8 | 4w x 1r (`ternary_i8_b8_r1`) | 54 | 16.39 | 18.44 | 1.12x | 2.70 (int) | 36 warps/SM (registers) | 14 |
| 6912 x 2560 | 16 | 4w x 1r (`ternary_i8_b16_r1`) | 64 | 26.63 | 28.67 | 1.08x | 5.41 (int) | 32 warps/SM (registers) | 14 |
| 2560 x 6912 | 1 | 8w x 2r fused (`ternary_i8_fused_b1_r2`) | 40 | 8.22 | 10.27 | 1.25x | 2.48 (dram) | 48 warps/SM (warps) | 16 |
| 2560 x 6912 | 2 | 8w x 2r fused (`ternary_i8_fused_b2_r2`) | 61 | 10.88 | 12.31 | 1.13x | 2.48 (dram) | 32 warps/SM (registers) | 15 |
| 2560 x 6912 | 4 | 2w x 4r (`ternary_i8_b4_r4`) | 120 | 14.36 | 16.41 | 1.14x | 2.49 (dram) | 16 warps/SM (registers) | 15 |
| 2560 x 6912 | 8 | 4w x 1r (`ternary_i8_b8_r1`) | 54 | 19.05 | 20.48 | 1.08x | 2.70 (int) | 36 warps/SM (registers) | 14 |
| 2560 x 6912 | 16 | 8w x 1r (`ternary_i8_b16_r1`) | 64 | 26.63 | 28.68 | 1.08x | 5.41 (int) | 32 warps/SM (registers) | 14 |
