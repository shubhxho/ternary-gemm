engine: dense f32 [64, 256, 256, 1] (avx2+fma dots if available)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4163 ns/tick; timer overhead p50 11.7 ns (included in each stage below)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     36.1 |     29.1 |     36.6 |      50.0 |     2177.1 |  9063250.9 |
| book update (all messages)         |    7189693 |    131.6 |     83.3 |    238.9 |     588.6 |     4083.7 | 10960892.5 |
| parse + book (decision msgs)       |    6971855 |    160.8 |    111.6 |    266.4 |     632.7 |     3049.6 | 10960921.7 |
| features + normalize               |    6971855 |    350.6 |    303.0 |    605.3 |     854.2 |     3649.9 |  1573699.4 |
| inference                          |    6971855 |   4227.1 |   4073.7 |   4458.3 |    6188.3 |    11118.7 | 137295741.2 |
| end-to-end msg -> decision         |    6971855 |   4738.5 |   4551.5 |   5020.3 |    7299.8 |    14170.0 | 137296145.8 |

replay throughput (untimed pass): 0.22 M msgs/s, 0.21 M decisions/s, 32.826 s wall
timed pass: 33.677 s wall, 33.630 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 89.2%
