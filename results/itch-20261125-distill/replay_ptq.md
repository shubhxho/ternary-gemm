engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-gfni lanes]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 13.3 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     28.7 |     24.2 |     32.5 |      49.2 |     1988.3 |   406887.1 |
| book update (all messages)         |    7189693 |     96.6 |     70.0 |    173.3 |     404.2 |     2026.7 | 12185315.3 |
| parse + book (decision msgs)       |    6971855 |    120.7 |     94.2 |    193.3 |     425.0 |     3499.2 | 12185342.8 |
| features + normalize               |    6971855 |    264.6 |    245.8 |    425.8 |     685.8 |     3340.9 |   284914.8 |
| inference                          |    6971855 |    642.0 |    613.3 |    691.7 |     995.0 |     3252.5 | 12838472.0 |
| end-to-end msg -> decision         |    6971855 |   1027.3 |    967.5 |   1215.8 |    1889.2 |     6818.4 | 12838864.5 |

replay throughput (untimed pass): 1.04 M msgs/s, 1.00 M decisions/s, 6.946 s wall
timed pass: 7.826 s wall, 7.820 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 62.5%
