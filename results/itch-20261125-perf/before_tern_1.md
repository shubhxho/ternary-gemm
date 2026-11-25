slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     30.2 |     25.8 |     33.3 |      43.3 |     2110.8 |    73557.4 |
| book update (all messages)         |    7189693 |    100.0 |     71.7 |    179.2 |     405.8 |     1827.5 | 35127031.8 |
| parse + book (decision msgs)       |    6971855 |    126.0 |     96.7 |    200.0 |     429.2 |     2594.1 | 35127071.0 |
| features + normalize               |    6971855 |    282.5 |    259.2 |    437.5 |     640.8 |     3155.8 |  2741456.8 |
| inference                          |    6971855 |   1108.8 |   1074.2 |   1182.5 |    1418.3 |     3045.8 | 16788029.5 |
| end-to-end msg -> decision         |    6971855 |   1517.4 |   1452.5 |   1687.5 |    2108.3 |     7165.7 | 35135345.9 |

replay throughput (untimed pass): 0.69 M msgs/s, 0.67 M decisions/s, 10.365 s wall
timed pass: 11.178 s wall, 11.180 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 73.1%
wrote before_tern_1.json
