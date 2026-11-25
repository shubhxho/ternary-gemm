engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-gfni lanes]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 12.5 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     26.3 |     23.3 |     33.3 |      48.3 |      205.0 |    98702.8 |
| book update (all messages)         |    7189693 |     96.2 |     68.3 |    172.5 |     405.0 |     3860.8 |  1052470.1 |
| parse + book (decision msgs)       |    6971855 |    115.9 |     92.5 |    191.7 |     417.5 |     2030.0 |  1052488.4 |
| features + normalize               |    6971855 |    268.4 |    251.7 |    437.5 |     671.7 |     3684.2 |    40009.3 |
| inference                          |    6971855 |    645.5 |    631.7 |    708.3 |     960.0 |     3731.7 | 20236219.8 |
| end-to-end msg -> decision         |    6971855 |   1029.8 |    976.7 |   1221.7 |    1732.5 |     7482.5 | 20236733.1 |

replay throughput (untimed pass): 0.99 M msgs/s, 0.96 M decisions/s, 7.246 s wall
timed pass: 7.809 s wall, 7.800 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 62.7%
