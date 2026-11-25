# ITCH replay: a small order-book model, end to end

Kernel microbenchmarks (PR #10) say how fast a matmul runs. This experiment asks the trading
question instead: replay real Nasdaq market data, rebuild the order book, score every book update
with a small model, and measure how much of the message-to-decision time is inference, and
whether the ternary model keeps its signal on a held-out period.

Everything below was **measured on a CPU-only machine** (8-vCPU Intel Xeon VM with AVX-512
VNNI/VBMI, shared, pinned to one core). **No GPU number in this document was measured on this
workload.** GPU rows are either quoted from PR #10 (a different workload) or marked as estimates.

## Reproduce

```bash
scripts/itch_download.sh     # ~4.8 GB public Nasdaq file -> data/itch/slice.itch (5 symbols, ~220 MB)
scripts/itch_replay.sh       # CPU: features, replay every engine pinned to a core, signal quality
RETRAIN=1 scripts/itch_replay.sh   # also retrain the models (numpy only, ~3 min on 8 cores)
scripts/itch_gpu.sh          # on a CUDA box: GPU replay (batch 1) + GPU throughput at batch 1..4096
```

Raw data is never committed (`.gitignore`). The committed results and the two trained models are
in [`results/itch-20261125/`](../results/itch-20261125/).

## Data slice

| | |
|---|---|
| Source | Nasdaq TotalView-ITCH 5.0 public sample `01302019.NASDAQ_ITCH50.gz` from `emi.nasdaq.com/ITCH/Nasdaq ITCH/` (4,764,426,091 bytes, 368,366,634 messages, one full trading day, 30 Jan 2019) |
| Slice | AAPL, MSFT, INTC, CSCO, AMD: every message for those five symbols plus system events and their stock-directory entries, cut by `itch filter` into a valid framed ITCH file. 7,189,693 messages, 219,414,116 bytes |
| Book check | Order-by-order book (adds, executes, executes-with-price, partial cancels, deletes, replaces); 0 references to unknown orders over the whole day |
| Decisions | A decision = a book change for one of the five symbols during 09:30–16:00 ET with both sides present and uncrossed: 6,971,855 |

Caveat: 30 Jan 2019 was the day after Apple's earnings, so AAPL activity is unusual. One day, five
large-cap names; not a representative universe.

## Model and target

* **Target ("next mid-price move")**: direction of the next change of that symbol's mid price
  (up vs down). Also reported: rank correlation (IC) with the mid change over the next 1 second.
* **Features** (`src/itch.rs`, 32 per snapshot, input = current + previous snapshot = 64): spread,
  queue imbalance at 1/2/3/5 levels, log sizes and depth, microprice − mid, order-flow imbalance
  (Cont–Kukanov–Stoikov) and two EWMAs, signed execution flow EWMAs, mid returns over 1/10/100
  events, inter-event time, event type/side/size/distance from touch, time of day, level counts.
  Standardized with train-period mean/std, clipped to ±5.
* **Model**: `[64, 256, 256, 1]` ReLU MLP, the same shape as the repo's `tiny` signal model that
  PR #10 benchmarked, so the latency numbers line up.
  * dense f32: Adam / AdamW, binary cross-entropy. The original recipe (below) used a 2M-sample
    random subsample of the training period; the improved recipe uses every labelled train sample.
  * ternary QAT: initialized from dense, then fine-tuned with ternary weights (absmean per row) and
    int8 per-sample activations in the forward pass, straight-through backward. The improved recipe
    also distills the frozen dense teacher's logits.
  * ternary PTQ: the dense weights ternarized after training, no fine-tuning.
* **Split (chronological, 5-minute embargo gaps)**: train 09:30–12:30, validation 12:35–13:30
  (model selection by AUC), test 13:35–16:00. A sample is only kept if the move that labels it
  also happens inside its own split. Normalization, training and selection never touch the test
  period.
* All model scores in the table come from the Rust engines themselves (`itch replay --preds-out`),
  not from Python. The ternary `CpuEngine` was checked bit-exact against the scalar reference on
  the first 2,000 replayed decisions of every run.

## Signal quality (test period, 2,671,033 decisions, 50.8% up)

| model | AUC (next mid move) | accuracy | IC Spearman (1 s) | IC Pearson (1 s) | sign agreement with dense |
|---|---|---|---|---|---|
| dense f32 | 0.7094 | 0.6424 | 0.2937 | 0.1947 | 1.0000 |
| **ternary QAT** | **0.7137** | **0.6481** | **0.3011** | **0.2045** | 0.8671 |
| ternary PTQ | 0.6763 | 0.6195 | 0.2506 | 0.1658 | 0.7360 |
| baseline: L1 queue imbalance | 0.6721 | 0.6249 | 0.2401 | 0.1426 | – |
| baseline: microprice − mid | 0.6643 | 0.6251 | 0.2302 | 0.1050 | – |
| baseline: 5-level depth imbalance | 0.5607 | 0.5488 | 0.0835 | 0.0456 | – |

Per-symbol test AUC, dense / ternary QAT: AAPL 0.665 / 0.668, AMD 0.718 / 0.723, CSCO 0.720 /
0.720, INTC 0.720 / 0.722, MSFT 0.743 / 0.746.

How to read it:

* **Ternary with QAT showed no measurable quality loss here.** It is 0.004 AUC above dense, but
  QAT got three extra fine-tuning epochs, this is one day, and there are no confidence intervals,
  so read it as "no loss", not "better".
* **Post-training ternarization does lose signal** (−0.033 AUC, only slightly above the one-line
  imbalance rule). QAT is required.
* Both learned models beat the best non-ML baseline (L1 queue imbalance) by about 0.04 AUC.
* This is a prediction-quality test, not a strategy: no costs, queue position, fills or P&L.


## Improved training recipe (same split, features, and architecture)

Same official protocol as the table above (train 09:30–12:30, val 12:35–13:30, test 13:35–16:00,
5-minute embargo, no label leakage, model selection by validation AUC only, Rust `itch replay
--preds-out` scores). Same 64-dim features and `[64, 256, 256, 1]` ReLU MLP. What changed is only
the training recipe in `scripts/itch_train.py --recipe improved` (default):

* every labelled training sample (3,612,183), not a 2M subsample
* AdamW (decoupled weight decay 1e-4) and a cosine learning-rate decay
* QAT fine-tuning with logit-MSE distillation from the frozen dense teacher (mix 0.5), 6 epochs,
  early stopping on val AUC

`--recipe legacy` reproduces the original hyperparameters. Unit tests for AUC, the no-leakage
split, absmean ternarization, distillation and the official windows: `python3 scripts/test_itch_train.py`.

Official test numbers from this recipe, scored by the Rust engines on the same slice
(`results/itch-20261125-distill/`):

| model | AUC (next mid move) | accuracy | IC Spearman (1 s) | IC Pearson (1 s) | sign agreement with dense |
|---|---|---|---|---|---|
| dense f32 | 0.7064 | 0.6419 | 0.2922 | 0.1959 | 1.0000 |
| ternary QAT | 0.7124 | 0.6464 | 0.3027 | 0.2083 | 0.9231 |
| ternary PTQ | 0.6883 | 0.5782 | 0.2623 | 0.1660 | 0.6074 |
| baseline: L1 queue imbalance | 0.6721 | 0.6249 | 0.2401 | 0.1426 | – |
| baseline: microprice − mid | 0.6643 | 0.6251 | 0.2302 | 0.1050 | – |
| baseline: 5-level depth imbalance | 0.5607 | 0.5488 | 0.0835 | 0.0456 | – |

Compared with the original official table (same test period, same features):

* **Next-mid AUC did not improve.** QAT 0.7124 vs 0.7137, dense 0.7064 vs 0.7094. That is a
  small move against the previous run, not a win, and is well within what one day of data can
  wobble. Do not quote this recipe as a higher official AUC.
* **PTQ did improve:** 0.6763 → 0.6883 AUC, 0.2506 → 0.2623 Spearman IC. The dense teacher is
  still ahead, so QAT remains the right inference model, but the untuned ternary is less of a
  cliff.
* **QAT tracks dense more closely** (sign agreement 0.8671 → 0.9231), which is what distillation
  is for, and the 1 s IC is slightly higher (Spearman 0.3011 → 0.3027, Pearson 0.2045 → 0.2083).
* The L1 imbalance baseline is 0.6721 in both runs, which is a check that the split and labels
  did not change.
* Filling the reserved `f[31]` slot with a mid-horizon OFI EWMA was tried and **not adopted**:
  it did not beat this recipe's validation AUC, so the feature extractor is unchanged.

Walk-forward expanding windows on the same day (5-minute embargoes, each fold retrained from
scratch, never used to pick the official model). Train always starts at 09:30 ET. These folds
share an expanding prefix so they are not independent; they mainly show time-of-day variation,
not a four-day standard error.

| fold | train end | test window | n_test | dense AUC | QAT AUC | PTQ AUC | L1 imbalance |
|---|---|---|---|---|---|---|---|
| fold1_am | 11:00 | 11:35–12:30 | 754,781 | 0.7148 | 0.7173 | 0.6588 | 0.6852 |
| fold2_mid | 12:00 | 12:35–13:30 | 569,890 | 0.6903 | 0.6940 | 0.6749 | 0.6670 |
| fold3_early_pm | 13:00 | 13:35–14:30 | 931,398 | 0.7104 | 0.7138 | 0.6736 | 0.6613 |
| fold4_late_pm | 14:00 | 14:35–16:00 | 1,640,244 | 0.7190 | 0.7243 | 0.6641 | 0.6779 |
| mean ± std |  |  |  | 0.7086 ± 0.0127 | 0.7124 ± 0.0130 | 0.6679 ± 0.0077 | 0.6728 ± 0.0107 |

QAT is above dense in every fold. The walk-forward QAT mean (0.7124) matches the official
afternoon test (0.7124). The previous official QAT number (0.7137) sits inside this spread.
The midday fold is the weakest for every learned model, including dense. Raw table:
`results/itch-20261125-distill/walkforward.md`.

Latency is a property of the architecture and the CPU kernel, not of these weights. This recipe
does not claim a new end-to-end latency number; use the hot-path table below.

## Hot-path round (Nov 25): ternary end to end 1.46 → 0.90 µs p50

Same slice, same model (`qat.tgmlp`), same harness, `taskset -c 3`. "Before" is `main` at
`75af843`, "after" is this branch; the two binaries were run interleaved in one session, three
replays each for the ternary engine and two each for `--engine none`. Cells are medians of the
runs, in ns. Raw files: [`results/itch-20261125-perf/`](../results/itch-20261125-perf/).

| stage (ternary engine) | p50 before | p50 after | p99 before | p99 after |
|---|---|---|---|---|
| parse (all messages) | 25.8 | 23.3 | 43.3 | 39.2 |
| book update (all messages) | 72.5 | 62.5 | 405.8 | 336.7 |
| features + normalize | 259.2 | **230.0** | 640.8 | **579.2** |
| inference | 1,083.3 | **554.2** | 1,418.3 | **833.3** |
| **end to end, message → decision** | **1,455.0** | **895.0** | **2,108.3** | **1,472.5** |

* End-to-end p99.9: 6,616 → 5,577 ns (median of 3; still dominated by VM preemption).
* Replay throughput (untimed pass): 0.70 → 1.14 M msgs/s.
* Inference only, back to back over all 6.97M feature vectors: 1,087.8 → 548.5 ns per decision.
* `--engine none` (parse + book + features only), end-to-end p50: 324 / 328 → 295 / 298 ns
  (two runs each); features p50 216 / 225 → 196 / 201 ns. p99 was unchanged within noise
  (623 / 635 → 622 / 606 ns), and the book-update stage alone did not move measurably there
  (59 → 58 ns), so the book-stage drop in the ternary table is mostly less cache pressure from
  the smaller inference, not the book change itself.
* Inference is now about 62% of mean end-to-end time (was 73–75%).

What changed:

1. **Inference** uses the new rows-in-lanes AVX-512 kernel with a fused epilogue and a quantizer
   that reuses the previous layer's max (details and per-model numbers in
   [`CPU_PERF.md`](CPU_PERF.md)). This is most of the gain.
2. **Features.** Each side's book tree is walked once (touch + 4 depth levels from one iterator
   instead of separate lookups), `log(dt)` is computed once, and `ln(1+x)` for integer arguments
   (sizes, counts, order-flow imbalance) comes from a 4,096-entry table, falling back to libm when
   the argument is not an exact small integer. A 32,768-entry table was also tried; it had worse
   p99 in an A/B, so the 16 KiB one was kept.
3. **Book level update** does one B-tree descent (`Entry` API) instead of a lookup followed by an
   insert/remove.

Bit-exactness: the original feature code is kept as `itch::features_reference`, and the warm-up
pass of every replay now runs it on its own book state and asserts identical feature bits on
every decision (6,971,855 decisions checked, logged as `features bit-exact vs features_reference`
and recorded as `features_bitexact_checked` in the JSON). The predictions written by
`--preds-out` are byte-identical to the committed run's `preds_ternary_qat.f32` (`cmp`). Unit
tests cover the table, the book update on 60k random messages, and the kernels.

Caveats: the "before" figures here (1,455 / 2,108 ns) are lower than the table below
(1,518 / 2,877 ns), which came from an earlier session; compare within a table, not across.
The VM is shared and not core-isolated, so p99 moves by ±15% between runs; the inference and
end-to-end p50 gains are far outside that range, the features gain (~11%) is near it.

The tables below are the original measurements from the first ITCH round, kept for reference.

## Latency: where the time goes (batch 1, one pinned core)

Each message is timed with the TSC (0.4166 ns/tick, about 12 ns timer overhead included in each
stage): parse → book update → features → inference. The ITCH slice is in memory before timing
starts, so there is no disk or network I/O. Second of two passes (the first warms up). ns.

| stage | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|
| parse (all 7.19M messages) | 29 | 36 | 48 | 163 |
| book update (all messages) | 74 | 209 | 552 | 1,908 |
| features + normalize | 276 | 567 | 843 | 3,327 |
| **inference, dense f32** (AVX2 row dots) | 4,074 | 4,458 | 6,188 | 11,119 |
| **inference, dense f32 k-major** (AVX-512 FMA) | 3,464 | 3,853 | 4,581 | 13,807 |
| **inference, ternary W2A8** (AVX-512 VNNI, packed 2-bit) | 1,088 | 1,225 | 1,973 | 4,774 |
| inference, ternary as 1-byte codes (int8-weight cost proxy) | 1,690 | 1,876 | 2,468 | 6,115 |

End-to-end, message in → decision out (decision messages only):

| engine | p50 | p99 | p99.9 | inference share of mean | replay throughput (untimed) |
|---|---|---|---|---|---|
| no model (parse + book + features) | 408 | 943 | 6,066 | – | 2.88 M msgs/s |
| dense f32 | 4,552 | 7,300 | 14,170 | 89% | 0.22 M msgs/s |
| dense f32 k-major (AVX-512) | 3,938 | 5,540 | 15,688 | 88% | 0.26 M msgs/s |
| **ternary W2A8** | **1,518** | **2,877** | **7,298** | 72% | **0.70 M msgs/s** |
| ternary PTQ (same engine) | 1,472 | 3,452 | 11,769 | 72% | 0.62 M msgs/s |
| int8-weight proxy | 2,131 | 3,632 | 11,914 | 79% | 0.49 M msgs/s |

Inference only, back to back over all 6.97M replayed feature vectors (batch 1, one core):
dense 4.20 µs, dense k-major 3.25 µs, ternary 1.05 µs, int8-weight proxy 1.83 µs per decision.
CPU utilization: one core at 99–100% during replay (a busy loop, as a feed handler would be).

What this says:

* **Inference is on the critical path for this model.** With a dense f32 model it is about 89% of
  the mean message-to-decision time; parse + book + features are roughly 0.4 µs.
* **Ternary cuts end-to-end p50 about 2.6–3.0× and p99 about 1.9–2.5× against the two dense
  baselines** (1.5 µs vs 3.9–4.6 µs p50). After that, inference is still about 72% of the time,
  so features and the book are the next things to optimize.
* **Why**: the f32 weights are 329 KB (they live in L2 and the dense path is L2-bandwidth bound),
  the packed 2-bit ternary weights are about 21 KB and fit in the 48 KB L1. The 1-byte-code
  variant (82 KB, the same bytes and instructions an int8-weight model would use) sits in
  between at 1.7 µs. So part of the win is "smaller weights" generally, and an int8 dense model
  would be a stronger baseline than f32. Its quality was not trained or measured.
* **Tails are noisy.** This is a shared VM (load average about 2 during the runs). The max column
  in the raw files reaches tens of milliseconds (preemption). p99.9 should be re-measured on an
  isolated core (`isolcpus`, no SMT, fixed frequency) before quoting it externally.

## Comparison with PR #10

PR #10 (open, from a contributor fork, not merged) fixes the tensor-core path and adds the first
GPU results on an RTX PRO 6000 Blackwell. None of its code is needed here: this harness uses the
CPU `CpuEngine` and the existing `signal::SignalEngine` GPU path, both already on `main`. This
branch is based on `main` (`028884b`).

| | PR #10 (kernel / synthetic) | this PR (ITCH replay) |
|---|---|---|
| Workload | GEMV shapes from BitNet-2B (2560×2560 up to 11008×4096) and random untrained signal models with random inputs | a trained 64-256-256-1 order-book model on 7.19M real ITCH messages |
| What is timed | kernel time (Nsight) or one inference request, host→signal→host | parse, book, features and inference per message, plus end to end |
| CPU, `tiny` 64-256-256-1, ternary p50 | 1.41 µs in the committed results file (AMD EPYC 9B45, AVX2 path). The PR text says 0.61 µs, which no committed file shows | 1.09 µs inference / 1.52 µs end to end (Xeon VM, AVX-512 VNNI packed path) |
| CPU dense f32 vs ternary, `tiny` | 3.43 vs 1.41 µs (2.4×) | 3.46–4.07 vs 1.09 µs (3.2–3.7×) |
| GPU, `tiny`, batch 1 | 20.44 µs graph + spin, 8.55 µs persistent kernel, 22.47 µs cuBLAS FP16 | **not measured** (no GPU); `scripts/itch_gpu.sh` runs it |
| Headline GPU kernel claims | 2.78 µs W2A8 GEMV at 2560×2560; up to 6.9× vs cuBLAS FP16; BitNet-2B stack 1.014 ms/step | not applicable: those shapes are about 80× more weights than this whole model |

Consistency:

* **Consistent: "the CPU wins small models at batch 1."** PR #10's own GPU round trip for this
  exact shape (20.4 µs, or 8.55 µs with the experimental persistent kernel) is longer than this
  harness's entire CPU message-to-decision p99 (2.9 µs). As an **estimate**, swapping the CPU
  inference for PR #10's GPU numbers would put the end to end at roughly 9–21 µs. That number
  comes from a different machine and should be measured with `scripts/itch_gpu.sh`.
* **Consistent: ternary beats dense f32 on the CPU for this shape**, by a similar or larger ratio
  (2.4× in PR #10, 3.2–3.7× here, different CPUs and SIMD paths).
* **Doesn't transfer: the GPU kernel speedups.** 2.78 µs and 6.9× are kernel-only times on
  LLM-sized matrices that are bandwidth-bound in GPU DRAM. This model's weights fit in L1, and the
  GPU's fixed launch and PCIe cost (about 20 µs per request in PR #10) dominates. They are the right
  numbers for LLM decode and the wrong ones to quote for this workload.
* **Open discrepancy: PR #10's description cites 0.61 µs CPU p50 for `tiny`, but its committed
  `-latency.md` shows 1.41 µs.** Use the committed figure until the run behind 0.61 µs is
  published.

## Not measured

* Any GPU on this workload (code is compile-checked: `cargo build --features cuda`; run `scripts/itch_gpu.sh`).
* Network ingress: no NIC, kernel bypass, MoldUDP64/SoupBinTCP session handling or packet decode. Replay is from memory.
* Order entry, risk checks, and the wire-to-wire latency a desk would actually quote.
* Quality of an int8 or fp16 dense model (only the int8 cost proxy is timed), and a dense model fine-tuned for as long as QAT.
* More than one day or other market regimes, other horizons, confidence intervals, and any strategy P&L after costs.
* Tail latency on an isolated, frequency-pinned core. FPGA. Energy.
