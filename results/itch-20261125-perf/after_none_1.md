slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: none (features only)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     20.8 |     21.7 |     26.7 |      36.7 |      100.0 |    20313.9 |
| book update (all messages)         |    7189693 |     72.4 |     58.3 |    115.0 |     285.8 |     1440.0 |   124472.8 |
| parse + book (decision msgs)       |    6971855 |     89.3 |     78.3 |    133.3 |     283.3 |      502.5 |    73006.7 |
| features + normalize               |    6971855 |    202.4 |    195.8 |    317.5 |     485.0 |     2766.6 |  1029106.8 |
| inference                          |    6971855 |     14.2 |     14.2 |     15.8 |      18.3 |       34.2 |    12317.4 |
| end-to-end msg -> decision         |    6971855 |    306.0 |    295.0 |    431.7 |     621.7 |     2960.8 |  1029341.8 |

replay throughput (untimed pass): 3.38 M msgs/s, 3.28 M decisions/s, 2.128 s wall
timed pass: 2.404 s wall, 2.410 s CPU (100% of one core)
inference share of mean end-to-end decision latency: 4.6%
wrote after_none_1.json
