# Contributing

Thanks for helping. This crate is small on purpose: kernels that are easy to audit, plus a CPU
oracle that every path must match.

## Ground rules

* **Correctness first.** Every int8 path must stay **bit-exact** against `src/reference.rs` /
  `TernaryMlp::forward_reference`. Add a test for anything new (CPU emulation if it's a kernel).
* **Never commit fabricated numbers.** Benchmark reports in `results/` must come from a real run
  of `scripts/bench.sh` or `scripts/latency_cpu.sh`, and must say what hardware they ran on.
  Theoretical figures must be labeled "Theoretical (not measured)".
* Keep `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test --release` clean.
* MSRV is the `rust-version` in `Cargo.toml` (checked in CI).

## Workflow

```bash
make check          # fmt --check + clippy -D warnings + tests + rustdoc
make compile-check  # NVRTC + ptxas for sm_80..sm_120 (needs NVRTC + ptxas, see README)
make hooks          # install the pre-commit hook (fmt + clippy on staged Rust changes)
```

GPU tests skip automatically without a GPU. On a GPU machine run them with
`TERNARY_REQUIRE_GPU=1 cargo test --release -- --test-threads=1` so a missing GPU fails loudly.

## Commits and PRs

* One logical change per commit, with an imperative subject (`Add ...`, `Fix ...`).
* Describe how you verified it: which tests, which hardware (or "CPU only").
* Kernel changes: include the `compile_check` register/spill diff for the affected kernels.
