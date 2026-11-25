slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: none (features only)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 11.7 ns (included in each stage below)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     20.7 |     21.7 |     25.8 |      34.2 |       80.0 |    89664.1 |
| book update (all messages)         |    7189693 |     75.4 |     59.2 |    122.5 |     292.5 |     1575.0 |   184066.5 |
| parse + book (decision msgs)       |    6971855 |     91.6 |     78.3 |    138.3 |     289.2 |      530.0 |   184089.8 |
| features + normalize               |    6971855 |    222.4 |    225.0 |    333.3 |     504.2 |     2825.0 |   112435.5 |
| inference                          |    6971855 |     19.9 |     20.0 |     21.7 |      25.0 |       29.2 |    25510.6 |
| end-to-end msg -> decision         |    6971855 |    333.9 |    328.3 |    455.0 |     635.0 |     3166.6 |   184519.8 |

replay throughput (untimed pass): 3.10 M msgs/s, 3.01 M decisions/s, 2.318 s wall
timed pass: 2.605 s wall, 2.620 s CPU (101% of one core)
inference share of mean end-to-end decision latency: 6.0%
wrote before_none_2.json
