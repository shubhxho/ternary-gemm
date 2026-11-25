engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
ternary engine bit-exact vs scalar reference on 2000 replayed decisions

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     27.9 |     29.2 |     35.8 |      48.3 |      163.3 |    24048.3 |
| book update (all messages)         |    7189693 |    108.3 |     74.2 |    209.2 |     551.7 |     1908.3 |   598485.8 |
| parse + book (decision msgs)       |    6971855 |    132.3 |    101.7 |    231.7 |     560.8 |     1620.0 |   598528.3 |
| features + normalize               |    6971855 |    319.2 |    275.8 |    566.7 |     842.5 |     3326.7 |   740764.6 |
| inference                          |    6971855 |   1149.6 |   1088.3 |   1225.0 |    1972.5 |     4774.2 | 35169001.9 |
| end-to-end msg -> decision         |    6971855 |   1601.0 |   1518.3 |   1871.7 |    2876.7 |     7298.3 | 35169707.8 |

replay throughput (untimed pass): 0.70 M msgs/s, 0.68 M decisions/s, 10.306 s wall
timed pass: 11.697 s wall, 11.690 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 71.8%
