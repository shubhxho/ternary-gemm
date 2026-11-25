slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-gfni lanes]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     23.9 |     23.3 |     30.0 |      39.2 |      133.3 |    64691.9 |
| book update (all messages)         |    7189693 |     82.6 |     63.3 |    137.5 |     336.7 |     1639.2 |  1111041.6 |
| parse + book (decision msgs)       |    6971855 |    102.9 |     85.8 |    157.5 |     344.2 |     1678.3 |  1111071.6 |
| features + normalize               |    6971855 |    241.9 |    230.0 |    381.7 |     579.2 |     3047.5 |  1283232.4 |
| inference                          |    6971855 |    594.2 |    555.0 |    650.0 |     853.3 |     2762.5 |  3493735.6 |
| end-to-end msg -> decision         |    6971855 |    939.0 |    899.2 |   1103.3 |    1472.5 |     5687.4 |  3494032.2 |

replay throughput (untimed pass): 1.12 M msgs/s, 1.09 M decisions/s, 6.418 s wall
timed pass: 7.078 s wall, 7.070 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 63.3%
wrote after_tern_3.json
