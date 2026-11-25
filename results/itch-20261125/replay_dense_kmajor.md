engine: dense f32 [64, 256, 256, 1] k-major [avx512f k-major fma]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4173 ns/tick; timer overhead p50 11.7 ns (included in each stage below)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     33.7 |     31.7 |     39.2 |      50.9 |      308.8 |    40635.1 |
| book update (all messages)         |    7189693 |    115.3 |     81.0 |    217.8 |     565.0 |     1729.3 |  1418925.0 |
| parse + book (decision msgs)       |    6971855 |    145.3 |    111.8 |    244.5 |     582.6 |     3142.3 |  1418945.0 |
| features + normalize               |    6971855 |    333.0 |    285.4 |    581.7 |     832.9 |     3475.4 |   226587.2 |
| inference                          |    6971855 |   3605.8 |   3463.7 |   3852.6 |    4581.2 |    13807.1 | 73507841.4 |
| end-to-end msg -> decision         |    6971855 |   4084.1 |   3937.7 |   4433.5 |    5540.2 |    15687.5 | 73508184.4 |

replay throughput (untimed pass): 0.26 M msgs/s, 0.25 M decisions/s, 28.062 s wall
timed pass: 29.072 s wall, 29.060 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 88.3%
