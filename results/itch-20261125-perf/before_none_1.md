slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: none (features only)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     21.1 |     21.7 |     25.0 |      33.3 |      106.7 |    24018.1 |
| book update (all messages)         |    7189693 |     75.1 |     59.2 |    120.0 |     281.7 |     1498.3 |  4360593.7 |
| parse + book (decision msgs)       |    6971855 |     92.2 |     79.2 |    136.7 |     281.7 |      586.7 |  4360615.3 |
| features + normalize               |    6971855 |    221.6 |    215.8 |    325.8 |     494.2 |     3134.1 |   465422.2 |
| inference                          |    6971855 |     20.0 |     20.0 |     21.7 |      25.0 |       30.0 |    24974.8 |
| end-to-end msg -> decision         |    6971855 |    333.9 |    324.2 |    446.7 |     622.5 |     3586.6 |  4361027.0 |

replay throughput (untimed pass): 3.20 M msgs/s, 3.10 M decisions/s, 2.250 s wall
timed pass: 2.638 s wall, 2.630 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 6.0%
wrote before_none_1.json
