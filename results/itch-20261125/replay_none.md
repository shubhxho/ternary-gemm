engine: none (features only)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4166 ns/tick; timer overhead p50 12.5 ns (included in each stage below)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     30.9 |     21.7 |     30.0 |      67.5 |     2737.4 |  5799337.6 |
| book update (all messages)         |    7189693 |    116.2 |     67.5 |    220.8 |     519.1 |     3474.8 |  4987342.1 |
| parse + book (decision msgs)       |    6971855 |    139.7 |     88.3 |    240.0 |     542.5 |     3608.2 |  5800147.6 |
| features + normalize               |    6971855 |    299.0 |    270.0 |    471.6 |     705.8 |     3557.3 | 54511567.8 |
| inference                          |    6971855 |     19.3 |     19.2 |     21.7 |      26.7 |       37.5 |    12215.3 |
| end-to-end msg -> decision         |    6971855 |    457.9 |    407.5 |    619.1 |     943.3 |     6066.4 | 54511741.1 |

replay throughput (untimed pass): 2.88 M msgs/s, 2.79 M decisions/s, 2.497 s wall
timed pass: 6.424 s wall, 6.260 s CPU (97% of one core)
inference share of mean end-to-end decision latency: 4.2%
