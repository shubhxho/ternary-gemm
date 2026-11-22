# Solar vs solc — measured results

All numbers were measured on this machine. Nothing here is estimated.

## Machine

- CPU: Intel(R) Xeon(R) Processor (family 6, model 207, stepping 2)
- Cores: 8 logical CPUs, 8 core(s) per socket, 1 thread(s) per core, 1 socket(s)
- Kernel-reported cpu MHz: 2400 (no cpufreq sysfs)
- Caches: L1d 384 KiB (8 instances), L1i 256 KiB (8 instances), L2 16 MiB (8 instances), L3 320 MiB (1 instance)
- Memory: 15.63 GiB
- OS: Debian GNU/Linux 13 (trixie)
- Kernel: `Linux grok-bot-vm-299988671 6.12.113+ #1 SMP PREEMPT_DYNAMIC Thu Nov 19 19:32:43 UTC 2026 x86_64 GNU/Linux`
- Hypervisor: KVM
- Measured at: 2026-11-22 16:54:36 IST (Asia/Calcutta)

## Tool pins

- Solar: `solar 0.3.0-dev+commit.f1b75c64` built from `f1b75c64ec466637d9040418268b365211d0e939`
- solc: `0.8.39+commit.ce47f2c8.Linux.g++`
- rustc: `rustc 1.100.0 (0ef1bc386 2026-11-09)`
- hyperfine: `hyperfine 1.19.0`
- Solar threads: 8 (all logical CPUs). solc Standard JSON is effectively single-threaded.
- Optimizer: off for both compilers (`settings.optimizer.enabled=false`). solc full compile uses `viaIR=true` (required for these corpora); Solar uses its native pipeline. This measures the compilers, not a production optimizer-runs setting.

## Corpora

- **openzeppelin-contracts** `v5.7.0` (`cab19933c33c2ad1d4c7a84864a3601dddfd16f3`): 248 Solidity files, 37,993 LoC, 1,534,986 bytes. Production contracts/ tree, excluding contracts/mocks/. Remapping @openzeppelin/contracts/=contracts/. evmVersion=osaka, matching the project's foundry.toml.
- **uniswap-v3-core** `v1.0.0` (`e3589b192d0be27e100cd0daaf6c97204fdb1899`): 33 Solidity files, 3,253 LoC, 148,910 bytes. Production contracts/ tree, excluding contracts/test/. Targets Solidity =0.7.6. solc 0.8.39 rejects the pragma, so solc is not compared. Solar parses and lowers the corpus but analysis/codegen fail on 0.7-style implicit integer conversions, so those stages are omitted.
- **uniswap-v4-core** `v4.0.0` (`e50237c43811bd9b526eff40f26772152a42daba`): 46 Solidity files, 4,644 LoC, 218,337 bytes. Production src/ tree, excluding src/test/. Includes the solmate import closure of src/ProtocolFees.sol (solmate @ the v4-core v4.0.0 submodule pin). evmVersion=cancun, matching the project's foundry.toml. Optimizer is off; the project's production via-IR + 44_444_444-runs config is not measured. For solc only, the exact `pragma solidity 0.8.26` in PoolManager.sol is widened to `^0.8.26` so solc 0.8.39 will parse it.

## Wall time

Median and p95 over hyperfine runs (parse/lowering: 20 runs after 3 warmup; full compile: 10 runs after 2 warmup).

| corpus | stage | Solar median | Solar p95 | solc median | solc p95 | speedup (solc/solar, median) |
|---|---|---:|---:|---:|---:|---:|
| openzeppelin-contracts | parsing | 7.54 ms | 8.72 ms | 61.11 ms | 71.50 ms | 8.11× |
| openzeppelin-contracts | lowering | 12.62 ms | 17.04 ms | — | — | — |
| openzeppelin-contracts | full | 34.22 ms | 36.20 ms | 1.816 s | 1.914 s | 53.08× |
| uniswap-v3-core | parsing | 4.40 ms | 5.38 ms | — | — | — |
| uniswap-v3-core | lowering | 4.61 ms | 5.13 ms | — | — | — |
| uniswap-v4-core | parsing | 4.29 ms | 6.36 ms | 9.74 ms | 10.34 ms | 2.27× |
| uniswap-v4-core | lowering | 5.63 ms | 6.28 ms | — | — | — |
| uniswap-v4-core | full | 23.08 ms | 25.63 ms | 619.38 ms | 666.64 ms | 26.84× |

## Peak memory (RSS)

Median and p95 of GNU time `%M` (kibibytes) over 10 runs per job. This is the compiler process's peak resident set, not the wall-time sample.

| corpus | stage | Solar median | Solar p95 | solc median | solc p95 |
|---|---|---:|---:|---:|---:|
| openzeppelin-contracts | parsing | 77.6 MiB | 82.6 MiB | 30.2 MiB | 30.2 MiB |
| openzeppelin-contracts | lowering | 88.7 MiB | 88.9 MiB | — | — |
| openzeppelin-contracts | full | 111.2 MiB | 113.2 MiB | 197.3 MiB | 197.6 MiB |
| uniswap-v3-core | parsing | 46.4 MiB | 49.7 MiB | — | — |
| uniswap-v3-core | lowering | 52.5 MiB | 53.0 MiB | — | — |
| uniswap-v4-core | parsing | 52.7 MiB | 56.5 MiB | 12.2 MiB | 12.2 MiB |
| uniswap-v4-core | lowering | 53.7 MiB | 56.8 MiB | — | — |
| uniswap-v4-core | full | 76.2 MiB | 78.8 MiB | 70.2 MiB | 71.1 MiB |

## Caveats

- Numbers are from a single KVM virtual machine and will not match a bare-metal run. The kernel reports the CPU as a generic 'Intel(R) Xeon(R) Processor' at 2400 MHz; there is no cpufreq sysfs, so turbo behaviour is unknown.
- Solar is given --threads equal to all logical CPUs (its default). solc's Standard JSON pipeline is effectively single-threaded. That is how each compiler is normally used, but it is not a same-thread-count comparison.
- Full compile is measured with the optimizer off, writing Standard JSON (bytecode requested) to /dev/null. Solar uses its native pipeline. solc full compile uses viaIR=true: without it, solc 0.8.39 fails the OpenZeppelin corpus with "Stack too deep", and Uniswap v4-core's own Foundry config is via-IR. This is a compiler-performance comparison, not a replay of a project's production optimizer-runs setting.
- solc only implements --stop-after parsing, so there is no solc 'lowering' number. Solar --stop-after lowering covers AST-to-HIR lowering, inheritance, name resolution and basic type checking.
- Uniswap v3-core v1.0.0 requires Solidity 0.7.6. The pinned solc 0.8.39 rejects that pragma, so v3 has no solc numbers. Solar parses and lowers the production sources, but analysis and codegen fail on 0.7-style implicit integer conversions; those stages are omitted rather than reported as times.
- Uniswap v4-core is measured at evmVersion=cancun. OpenZeppelin is measured at evmVersion=osaka. Solar is given the original sources. solc is given the same sources except that exact 0.8.x pragmas (`pragma solidity 0.8.26;` in v4-core's PoolManager.sol) are widened to a caret range so the pinned solc 0.8.39 will accept them. No 0.7.x pragmas are rewritten; v3-core is not compared against solc.
- Stage control uses Standard JSON settings.stopAfter (parsing / lowering). Solar's CLI --stop-after is not honoured in --standard-json mode on this pin, so it is not used.
- Wall times are collected with hyperfine --shell=none, passing the Standard JSON path as a file argument (no bash redirection). That avoids shell startup dominating the short parse/lowering runs.
- Peak RSS is GNU time %M on Linux (kibibytes). It is sampled in a separate pass from hyperfine, so a time sample and a memory sample are not from the same process.

