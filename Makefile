# Developer shortcuts. Everything here is also what CI runs.
CARGO ?= cargo
PY ?= python3

.PHONY: all check fmt fmt-check clippy test doc compile-check latency-cpu bench hooks deny msrv fuzz clean

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
