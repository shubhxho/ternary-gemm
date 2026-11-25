slice: 219414116 bytes, symbols [(14, "AAPL"), (331, "AMD"), (1718, "CSCO"), (4118, "INTC"), (5217, "MSFT")]
engine: none (features only)
cpu simd level: avx512-vnni
messages: 7189693, decisions (RTH book updates, two-sided): 6971855, unknown order refs: 0
tsc: 0.4167 ns/tick; timer overhead p50 14.2 ns (included in each stage below)
features bit-exact vs features_reference on 6971855 decisions (warm-up pass)

| stage (ns)                         |          n |     mean |      p50 |      p90 |      p99 |    p99.9 |        max |
|---|---|---|---|---|---|---|---|
| parse (all messages)               |    7189693 |     21.2 |     21.7 |     25.8 |      32.5 |       98.3 |    61661.8 |
| book update (all messages)         |    7189693 |     72.0 |     57.5 |    113.3 |     285.0 |     1454.1 |   116230.3 |
| parse + book (decision msgs)       |    6971855 |     89.7 |     77.5 |    131.7 |     287.5 |      524.2 |   116246.1 |
| features + normalize               |    6971855 |    202.9 |    200.8 |    312.5 |     485.8 |     2989.1 |   354158.3 |
| inference                          |    6971855 |     14.4 |     14.2 |     15.8 |      18.3 |       30.0 |    35257.9 |
| end-to-end msg -> decision         |    6971855 |    307.0 |    297.5 |    422.5 |     605.8 |     3312.5 |   354266.7 |

replay throughput (untimed pass): 3.24 M msgs/s, 3.14 M decisions/s, 2.221 s wall
timed pass: 2.427 s wall, 2.440 s CPU (101% of one core)
inference share of mean end-to-end decision latency: 4.7%
wrote after_none_2.json
