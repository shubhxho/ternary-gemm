#!/usr/bin/env bash
# Check every other-language ternary GEMV under langs/ against the Rust reference vectors.
# Each toolchain is optional: a missing one is reported as SKIP, a failing one as FAIL.
#
# Toolchains are found on PATH or through these variables:
#   ZIG, MOJO, JULIA, FUTHARK, PYTHON (with triton / halide / torch / numpy / pytest installed)
# Usage: scripts/langs_check.sh            (from the repository root)
set -uo pipefail
cd "$(dirname "$0")/.."

ZIG=${ZIG:-zig}
MOJO=${MOJO:-mojo}
JULIA=${JULIA:-julia}
FUTHARK=${FUTHARK:-futhark}
PYTHON=${PYTHON:-python3}

cargo run --quiet --release --example lang_vectors >/dev/null || exit 1

status=0
declare -a summary
run() { # name, availability command, check command
    local name=$1 have=$2 cmd=$3
    if ! eval "$have" >/dev/null 2>&1; then
        summary+=("SKIP  $name (toolchain not found)")
        return
    fi
    echo "== $name"
    if eval "$cmd"; then
        summary+=("PASS  $name")
    else
        summary+=("FAIL  $name")
        status=1
    fi
}

run zig "$ZIG version" \
    "$ZIG test -O ReleaseFast langs/zig/ternary.zig && $ZIG run -O ReleaseFast -mcpu=native langs/zig/check.zig -- langs/vectors"
run triton "$PYTHON -c 'import triton, torch, pytest'" \
    "TRITON_INTERPRET=1 $PYTHON -m pytest -q langs/triton"
run mojo "$MOJO --version" \
    "$MOJO build -O3 langs/mojo/ternary.mojo -o target/ternary_mojo && target/ternary_mojo langs/vectors"
run julia "$JULIA --version" \
    "$JULIA -O3 langs/julia/ternary.jl langs/vectors"
run futhark "$FUTHARK --version" \
    "$PYTHON langs/futhark/check.py langs/vectors --futhark $FUTHARK && $PYTHON langs/futhark/check.py langs/vectors --futhark $FUTHARK --backend multicore"
run halide "$PYTHON -c 'import halide, numpy, pytest'" \
    "$PYTHON -m pytest -q langs/halide"

echo
printf '%s\n' "${summary[@]}"
exit $status
