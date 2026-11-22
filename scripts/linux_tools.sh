#!/usr/bin/env bash
# Fetch the Linux kernel submodule (shallow) and build the measurement tools this project uses
# from its source tree: perf (PMU counters, sampling) and turbostat (per-core frequency, C-states,
# RAPL package power). Builds in-tree; installs nothing system-wide.
#
#   ./scripts/linux_tools.sh            # fetch (if needed) + build perf and turbostat
#   ./scripts/linux_tools.sh fetch      # only fetch the submodule (depth 1)
#   ./scripts/linux_tools.sh perf|turbostat
#
# Binaries end up in third_party/linux/tools/perf/perf and
# third_party/linux/tools/power/x86/turbostat/turbostat.
set -euo pipefail
cd "$(dirname "$0")/.."
SUB=third_party/linux

fetch() {
  if [ -f "$SUB/Makefile" ]; then
    echo "linux submodule already present at $SUB ($(git -C "$SUB" log -1 --format=%h 2>/dev/null || echo '?'))"
    return
  fi
  # update = none in .gitmodules keeps plain `git submodule update` from cloning ~5 GB of history;
  # --checkout overrides that for this explicit request, --depth 1 keeps it to one commit.
  git submodule update --init --depth 1 --checkout "$SUB"
}

build_perf() {
  make -C "$SUB/tools/perf" -j"$(nproc)" NO_LIBTRACEEVENT=1 NO_JEVENTS=1
  echo "perf: $SUB/tools/perf/perf"
}

build_turbostat() {
  make -C "$SUB/tools/power/x86/turbostat" -j"$(nproc)"
  echo "turbostat: $SUB/tools/power/x86/turbostat/turbostat"
}

case "${1:-all}" in
  fetch) fetch ;;
  perf) fetch; build_perf ;;
  turbostat) fetch; build_turbostat ;;
  all) fetch; build_perf; build_turbostat ;;
  *) echo "usage: $0 [fetch|perf|turbostat|all]" >&2; exit 2 ;;
esac
