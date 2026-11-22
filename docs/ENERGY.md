# Energy measurement

`ternary_gemm::energy` reads **hardware energy counters** before and after a workload. It never
estimates: if no counter is readable, energy is reported as unavailable.

| counter | scope | API | notes |
|---|---|---|---|
| RAPL (Intel, AMD) | CPU package(s), cumulative microjoules | `/sys/class/powercap/intel-rapl:N/energy_uj` (or `amd-rapl:N`) | top-level package zones are summed; subzones (`core`, `uncore`, `dram`) are skipped to avoid double counting; wrap-around handled with `max_energy_range_uj` |
| NVML (NVIDIA, feature `nvml`) | whole GPU board, cumulative millijoules | `nvmlDeviceGetTotalEnergyConsumption` | Volta and newer; `libnvidia-ml.so.1` is loaded with `dlopen` at runtime, so builds need no CUDA toolkit |

## Using it

```rust
use ternary_gemm::energy::EnergyMeter;
let meter = EnergyMeter::open();          // RAPL + (feature nvml) GPU 0
println!("counters: {}", meter.describe());
let r = meter.measure(100_000, || { /* one inference */ });
if let Some(ipj) = r.iters_per_joule() {
    println!("{:.3e} J/inference, {ipj:.0} inferences/J, {:.1} W", r.joules_per_iter().unwrap(), r.avg_watts().unwrap());
}
```

The latency benchmark has `--energy`: `cargo run --release --bin latency -- --energy` (CPU
paths, plus the GPU graph path when a GPU is present). `scripts/bench.sh` builds with `nvml`
and passes `--energy`, so GPU runs record energy next to latency.

## Reading the numbers correctly

* Counters are **package / board level**: they include idle power and everything else on the
  machine. Use an otherwise idle machine, pin the benchmark (`taskset`), run long enough that
  the counter resolution (RAPL updates about every 1 ms; NVML's energy counter is coarser) is
  small next to the total, and repeat.
* For GPU paths both counters matter: the host core spin-waits on the result (CPU package
  energy) while the GPU executes (board energy). The report adds both when both exist.
* Inferences per joule (= inferences/s per watt) is the per-watt figure of merit used in the
  thesis work; compare paths on the same machine in the same session only.

## Permissions

Since Linux 5.10 `energy_uj` is readable by root only (a mitigation for the PLATYPUS power side
channel). Options: run the benchmark with `sudo`, or grant read access for the session:
`sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj`. NVML energy needs no extra rights.

## This repository's measurements

The development VM exposes no RAPL zones and has no GPU, so **no energy numbers have been
measured yet**; `latency --energy` there prints "Energy NOT MEASURED". Numbers will come from
`scripts/bench.sh` on bare-metal hardware.
