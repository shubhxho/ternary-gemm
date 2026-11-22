# Linux kernel submodule (`third_party/linux`)

The repository pins `torvalds/linux` as a git submodule at `third_party/linux` (commit
`4843f68`, a 7.04-rc merge). It is **not** needed to build, test or benchmark the crate, and it
is never cloned by default.

## Why it is here

* **Measurement tools built from the matching source tree**: `tools/perf` (PMU counters, cycle
  and cache-miss sampling of the CPU paths), `tools/power/x86/turbostat` (per-core frequency,
  C-states and RAPL package power next to `latency --energy`).
* **Power / RAPL plumbing**: `drivers/powercap/intel_rapl_*.c` and `arch/x86/events/rapl.c`
  document exactly what `src/energy.rs` reads from `/sys/class/powercap` (units, wrap-around,
  permissions).
* **Kernel-level latency work** for the trading-signal path: `isolcpus` / `nohz_full`,
  `sched_setattr` (SCHED_DEADLINE), IRQ affinity, `cpuidle` and `cpufreq` governors, and the
  NVIDIA driver interaction with pinned memory, all of which bound the tail latency the
  `latency` benchmark reports.

## Fetching it

`.gitmodules` sets `shallow = true` and `update = none`, so `git clone --recursive` and plain
`git submodule update --init` skip it (it is several GB with history). Fetch one commit
explicitly:

```bash
git submodule update --init --depth 1 --checkout third_party/linux
# or: ./scripts/linux_tools.sh fetch
```

`--checkout` overrides `update = none` for this one command; `--depth 1` keeps it to the
pinned commit (a few hundred MB download instead of several GB of history).

## Building the tools

```bash
./scripts/linux_tools.sh            # fetch if needed, build perf and turbostat in-tree
sudo third_party/linux/tools/power/x86/turbostat/turbostat --quiet --show Busy%,Bzy_MHz,PkgWatt \
    taskset -c 3 ./target/release/latency --cpu-only
```

Build dependencies (Debian/Ubuntu): `build-essential flex bison libelf-dev libcap-dev`
(perf picks up more features if `libdw-dev libunwind-dev libslang2-dev` are present).

## CI

CI never touches the submodule: every `actions/checkout` runs with `submodules: false` (the
default) and nothing recurses, so the kernel tree costs CI nothing. Cargo packaging excludes
`third_party/` too.
