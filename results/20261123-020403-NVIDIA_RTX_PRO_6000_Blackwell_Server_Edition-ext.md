GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), 500 launches per cell, wall time with one sync at the end

| N x K | kernel | us / GEMV | effective GB/s (2-bit weights) |
|---|---|---|---|
| 2560 x 2560 | dp4a default (ternary_i8) | 4.11 | 398 |
| 2560 x 2560 | hand-written PTX | 4.11 | 399 |
| 2560 x 2560 | CpAsync { rows_per_warp: 1 } | 6.15 | 266 |
| 2560 x 2560 | CpAsync { rows_per_warp: 2 } | 4.11 | 399 |
| 2560 x 2560 | CpAsync { rows_per_warp: 4 } | 4.11 | 399 |
| 2560 x 2560 | SplitK { splits: 0 } | 8.21 | 200 |
| 2560 x 2560 | Lut | 30.74 | 53 |
| 2560 x 2560 | WarpSpecialized | 4.32 | 379 |
| 6912 x 2560 | dp4a default (ternary_i8) | 4.11 | 1076 |
| 6912 x 2560 | hand-written PTX | 4.11 | 1077 |
| 6912 x 2560 | CpAsync { rows_per_warp: 1 } | 10.26 | 431 |
| 6912 x 2560 | CpAsync { rows_per_warp: 2 } | 8.20 | 539 |
| 6912 x 2560 | CpAsync { rows_per_warp: 4 } | 6.15 | 719 |
| 6912 x 2560 | SplitK { splits: 0 } | 8.21 | 539 |
| 6912 x 2560 | Lut | 30.72 | 144 |
| 6912 x 2560 | WarpSpecialized | 8.20 | 540 |
| 2560 x 6912 | dp4a default (ternary_i8) | 6.15 | 719 |
| 2560 x 6912 | hand-written PTX | 6.15 | 719 |
| 2560 x 6912 | CpAsync { rows_per_warp: 1 } | 8.23 | 537 |
| 2560 x 6912 | CpAsync { rows_per_warp: 2 } | 6.16 | 718 |
| 2560 x 6912 | CpAsync { rows_per_warp: 4 } | 6.15 | 719 |
| 2560 x 6912 | SplitK { splits: 0 } | 8.21 | 539 |
| 2560 x 6912 | Lut | 77.80 | 57 |
| 2560 x 6912 | WarpSpecialized | 6.16 | 718 |
| 4096 x 11008 | dp4a default (ternary_i8) | 6.16 | 1830 |
| 4096 x 11008 | hand-written PTX | 6.16 | 1831 |
| 4096 x 11008 | CpAsync { rows_per_warp: 1 } | 10.25 | 1100 |
| 4096 x 11008 | CpAsync { rows_per_warp: 2 } | 8.20 | 1375 |
| 4096 x 11008 | CpAsync { rows_per_warp: 4 } | 6.16 | 1830 |
| 4096 x 11008 | SplitK { splits: 0 } | 10.25 | 1100 |
| 4096 x 11008 | Lut | 119.07 | 95 |
| 4096 x 11008 | WarpSpecialized | 8.20 | 1374 |
| 1024 x 16384 | dp4a default (ternary_i8) | 6.16 | 681 |
| 1024 x 16384 | hand-written PTX | 6.15 | 681 |
| 1024 x 16384 | CpAsync { rows_per_warp: 1 } | 6.15 | 682 |
| 1024 x 16384 | CpAsync { rows_per_warp: 2 } | 4.11 | 1020 |
| 1024 x 16384 | CpAsync { rows_per_warp: 4 } | 6.15 | 681 |
| 1024 x 16384 | SplitK { splits: 0 } | 8.20 | 511 |
| 1024 x 16384 | Lut | 173.29 | 24 |
| 1024 x 16384 | WarpSpecialized | 6.16 | 681 |
