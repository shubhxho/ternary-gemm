slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     27.3 |     27.5 |     35.8 |      47.5 |      139.2 |    33451.3 |
| book update (all messages)         |    7189693 |    103.4 |     75.8 |    202.5 |     465.0 |     1663.3 |   678640.1 |
| parse + book (decision msgs)       |    6971855 |    127.1 |    101.7 |    224.2 |     474.2 |      921.7 |   678674.2 |
| features + normalize               |    6971855 |    296.4 |    275.0 |    489.2 |     729.2 |     3389.1 |   510228.8 |
| inference                          |    6971855 |   1148.6 |   1085.8 |   1264.2 |    1595.8 |     4195.8 |  5310547.9 |
| end-to-end msg -> decision         |    6971855 |   1572.0 |   1515.8 |   1812.5 |    2475.8 |     6615.8 |  5310817.0 |

replay throughput (untimed pass): 0.71 M msgs/s, 0.69 M decisions/s, 10.138 s wall
timed pass: 11.434 s wall, 11.430 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 73.1%
wrote before_tern_2.json
