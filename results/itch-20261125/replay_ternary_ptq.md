engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4166 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     34.4 |     26.7 |     33.3 |      46.7 |     3020.0 |    71371.1 |
| book update (all messages)         |    7189693 |    108.9 |     72.5 |    197.4 |     471.5 |     3120.8 |  8703241.0 |
| parse + book (decision msgs)       |    6971855 |    138.1 |     97.5 |    221.6 |     530.7 |     3485.7 |  8703261.0 |
| features + normalize               |    6971855 |    295.2 |    262.4 |    459.0 |     674.0 |     5095.3 | 20758825.8 |
| inference                          |    6971855 |   1138.5 |   1079.7 |   1114.7 |    1560.4 |     4967.0 | 86123350.9 |
| end-to-end msg -> decision         |    6971855 |   1571.8 |   1472.1 |   1711.2 |    3451.5 |    11769.3 | 86127161.5 |

replay throughput (untimed pass): 0.62 M msgs/s, 0.60 M decisions/s, 11.524 s wall
timed pass: 11.624 s wall, 11.620 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 72.4%
