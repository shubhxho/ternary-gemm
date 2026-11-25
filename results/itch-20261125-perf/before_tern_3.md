slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     26.0 |     25.8 |     34.2 |      42.5 |       63.3 |    17449.0 |
| book update (all messages)         |    7189693 |     91.1 |     72.5 |    148.3 |     342.5 |     1773.3 |  2989703.4 |
| parse + book (decision msgs)       |    6971855 |    112.4 |     98.3 |    170.0 |     343.3 |      682.5 |  2989734.2 |
| features + normalize               |    6971855 |    254.9 |    247.5 |    390.8 |     580.0 |     2954.1 |   274674.7 |
| inference                          |    6971855 |   1122.9 |   1083.3 |   1212.5 |    1417.5 |     2511.6 | 18421738.7 |
| end-to-end msg -> decision         |    6971855 |   1490.2 |   1455.0 |   1672.5 |    1980.8 |     6005.8 | 18422135.4 |

replay throughput (untimed pass): 0.70 M msgs/s, 0.68 M decisions/s, 10.294 s wall
timed pass: 10.936 s wall, 10.930 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 75.4%
wrote before_tern_3.json
