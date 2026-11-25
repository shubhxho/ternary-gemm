slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-gfni lanes]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     23.2 |     22.5 |     29.2 |      35.8 |      115.8 |    28965.5 |
| book update (all messages)         |    7189693 |     79.6 |     62.5 |    132.5 |     331.7 |     1553.3 |   260892.4 |
| parse + book (decision msgs)       |    6971855 |     99.2 |     84.2 |    150.8 |     336.7 |      685.0 |    56990.3 |
| features + normalize               |    6971855 |    234.8 |    225.0 |    367.5 |     560.0 |     3081.6 | 10922683.3 |
| inference                          |    6971855 |    575.1 |    552.5 |    615.0 |     716.7 |     1845.8 |  3142748.8 |
| end-to-end msg -> decision         |    6971855 |    909.2 |    881.7 |   1060.0 |    1310.0 |     4616.6 | 10933636.5 |

replay throughput (untimed pass): 1.15 M msgs/s, 1.12 M decisions/s, 6.251 s wall
timed pass: 6.845 s wall, 6.850 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 63.3%
wrote after_tern_2.json
