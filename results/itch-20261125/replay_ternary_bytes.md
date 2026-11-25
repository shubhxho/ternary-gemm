engine: ternary weights as 1-byte codes [64, 256, 256, 1] CpuEngine::without_packed [avx512-vnni] (int8-weight cost proxy)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4172 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     28.0 |     27.5 |     35.0 |      46.7 |      146.0 |   235142.4 |
| book update (all messages)         |    7189693 |    122.7 |     77.6 |    206.9 |     515.6 |     2146.7 |  5031265.5 |
| parse + book (decision msgs)       |    6971855 |    145.7 |    104.3 |    230.3 |     524.8 |     1957.3 |  5031285.5 |
| features + normalize               |    6971855 |    325.4 |    275.3 |    536.5 |     784.3 |     4241.7 |  3171326.0 |
| inference                          |    6971855 |   1781.5 |   1689.5 |   1875.6 |    2467.9 |     6114.8 | 139629848.4 |
| end-to-end msg -> decision         |    6971855 |   2252.5 |   2130.9 |   2455.4 |    3631.8 |    11914.1 | 139630321.5 |

replay throughput (untimed pass): 0.49 M msgs/s, 0.47 M decisions/s, 14.725 s wall
timed pass: 16.475 s wall, 16.260 s CPU (99% of one core)
inference share of mean end-to-end decision latency: 79.1%
