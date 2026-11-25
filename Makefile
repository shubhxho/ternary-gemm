# Developer shortcuts. Everything here is also what CI runs.
CARGO ?= cargo
PY ?= python3

# make langs [ZIG=... MOJO=... JULIA=... FUTHARK=... PYTHON=...]: other-language ports vs the Rust vectors
.PHONY: langs lang-vectors
.PHONY: all check fmt fmt-check clippy test doc compile-check latency-cpu bench hooks deny msrv fuzz latency-json compare results-validate roofline autotune clean

all: check

check: fmt-check clippy test doc

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all --check

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

test:
	$(CARGO) test --release --all-targets

doc:
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --lib

# Needs libnvrtc on LD_LIBRARY_PATH and PTXAS=/path/to/ptxas (pip wheels work, see README).
compile-check:
	$(CARGO) run --release --bin compile_check

# Signal-model latency on the CPU (no GPU needed); writes results/<time>-cpu-latency.*
latency-cpu:
	./scripts/latency_cpu.sh

# CPU latency as a schema-validated JSON document (pinned to core $(CORE), default 2).
CORE ?= 2
latency-json:
	mkdir -p results
	taskset -c $(CORE) $(CARGO) run --release --bin latency -- --cpu-only --energy --json results/latency-$$(date +%Y%m%d-%H%M%S).json

# make compare BASE=results/a.json NEW=results/b.json [THRESHOLD=0.10]
THRESHOLD ?= 0.10
compare:
	$(PY) scripts/compare_results.py compare $(BASE) $(NEW) --threshold $(THRESHOLD)

results-validate:
	$(PY) scripts/compare_results.py validate results/*.json

# Roofline projections and modelled launch configs (no GPU needed). ARCH=sm_120 by default.
ARCH ?= sm_120
roofline:
	$(CARGO) run --release --bin roofline -- $(ARCH)

# Autotune on the local GPU (fails without one).
autotune:
	$(CARGO) run --release --bin autotune -- --out results/$$(date +%Y%m%d-%H%M%S)-autotune --cache results/tune-cache.csv

# Full GPU benchmark (fails without a GPU).
bench:
	./scripts/bench.sh

deny:
	$(CARGO) deny check

msrv:
	$(CARGO) +1.88 build --all-targets --locked

hooks:
	git config core.hooksPath scripts/hooks
	@echo "pre-commit hook installed (scripts/hooks/pre-commit)"

fuzz: ## run every fuzz target for FUZZ_SECS (default 60) seconds (nightly + cargo-fuzz)
	cd fuzz && for t in tgm_parse pack_roundtrip quantize_simd; do cargo +nightly fuzz run $$t -- -max_total_time=$${FUZZ_SECS:-60} || exit 1; done

clean:
	$(CARGO) clean

lang-vectors:
	$(CARGO) run --release --example lang_vectors

langs:
	scripts/langs_check.sh
