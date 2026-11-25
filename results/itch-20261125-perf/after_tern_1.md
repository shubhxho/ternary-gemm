slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-gfni lanes]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     27.6 |     23.3 |     29.2 |      40.0 |     2747.5 |   472596.2 |
| book update (all messages)         |    7189693 |     84.7 |     61.7 |    153.3 |     375.0 |     1839.1 |  4645333.5 |
| parse + book (decision msgs)       |    6971855 |    107.8 |     83.3 |    170.0 |     395.8 |     3060.0 |  4645348.5 |
| features + normalize               |    6971855 |    249.2 |    235.0 |    402.5 |     608.3 |     3313.3 |  4092736.6 |
| inference                          |    6971855 |    582.3 |    554.2 |    615.8 |     833.3 |     2110.8 | 27206486.4 |
| end-to-end msg -> decision         |    6971855 |    939.3 |    895.0 |   1097.5 |    1501.6 |     5577.4 | 27206781.4 |

replay throughput (untimed pass): 1.14 M msgs/s, 1.10 M decisions/s, 6.331 s wall
timed pass: 7.076 s wall, 7.070 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 62.0%
wrote after_tern_1.json
