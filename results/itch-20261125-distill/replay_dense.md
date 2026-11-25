engine: dense f32 [64, 256, 256, 1] (avx2+fma dots if available)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 12.5 ns (included in each stage below)
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     40.6 |     34.2 |     46.7 |      79.2 |     2060.9 |  3904666.6 |
| book update (all messages)         |    7189693 |    140.0 |     95.0 |    267.5 |     625.8 |     2440.0 |   546718.0 |
| parse + book (decision msgs)       |    6971855 |    178.4 |    129.2 |    303.3 |     686.7 |     3271.7 |  3905838.3 |
| features + normalize               |    6971855 |    316.3 |    283.3 |    513.3 |     829.2 |     4239.2 |   225780.3 |
| inference                          |    6971855 |   6553.9 |   6295.1 |   7226.7 |   10018.4 |    23808.5 | 88732724.2 |
| end-to-end msg -> decision         |    6971855 |   7048.5 |   6766.7 |   7799.2 |   11280.9 |    26703.6 | 88733024.2 |

replay throughput (untimed pass): 0.14 M msgs/s, 0.14 M decisions/s, 50.349 s wall
timed pass: 49.745 s wall, 49.730 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 93.0%
