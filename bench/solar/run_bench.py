#!/usr/bin/env python3
"""Reproducible Solar vs solc benchmark on pinned OpenZeppelin and Uniswap corpora.

Measures wall time (hyperfine: median and p95) and peak RSS (GNU time) for
parse, lowering and full compile. solc is compared only on stages it supports
for a given corpus (it only has --stop-after parsing, and solc 0.8.39 rejects
Uniswap v3-core's Solidity 0.7.6 pragma).
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS_PATH = ROOT / "pins.env"
CACHE = ROOT / ".cache"
CORPUS = Path(os.environ.get("CORPUS_DIR", ROOT / ".corpus"))
TOOLS = Path(os.environ.get("TOOLS_DIR", ROOT / ".tools"))

# Default run counts. Override with BENCH_WARMUP / BENCH_RUNS / BENCH_FULL_RUNS.
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
RUNS = int(os.environ.get("BENCH_RUNS", "20"))
FULL_WARMUP = int(os.environ.get("BENCH_FULL_WARMUP", "2"))
FULL_RUNS = int(os.environ.get("BENCH_FULL_RUNS", "10"))
MEM_RUNS_CAP = int(os.environ.get("BENCH_MEM_RUNS", "10"))

IMPORT_RE = re.compile(
    r'import\s+(?:(?:\{[^}]*\}|[\w.*]+(?:\s+as\s+\w+)?)\s+from\s+)?["\']([^"\']+)["\']'
)

# Exact 0.8.x pragmas (`pragma solidity 0.8.26;`) are rejected by a newer solc.
# Widen only those exact 0.8.x pins to a caret range so the pinned solc can parse the
# same sources. 0.7.x exact pragmas are left alone (solc is not compared on that corpus).
EXACT_08_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+=?(\d+\.\d+\.\d+)\s*;")


def widen_exact_08_pragmas(text: str) -> tuple[str, int]:
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        ver = m.group(1)
        major, minor, _patch = (int(x) for x in ver.split("."))
        if major == 0 and minor == 8:
            n += 1
            return f"pragma solidity ^{ver};"
        return m.group(0)

    return EXACT_08_PRAGMA_RE.sub(repl, text), n


def load_pins(path: Path = PINS_PATH) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        pins[k.strip()] = v.strip()
    return pins


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile. p is in 0..100."""
    if not sorted_vals:
        raise ValueError("empty sample")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[lo])
    w = k - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def summarize(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    return {
        "n": len(s),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(statistics.fmean(s)),
        "median": float(statistics.median(s)),
        "p95": percentile(s, 95),
        "stdev": float(statistics.stdev(s)) if len(s) > 1 else 0.0,
    }


def machine_specs() -> dict:
    def slurp(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    lscpu = slurp(["lscpu"])
    fields: dict[str, str] = {}
    for line in lscpu.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()

    mem_kb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                break
    except OSError:
        pass

    os_name = ""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        os_name = platform.platform()

    uname = slurp(["uname", "-a"])
    return {
        "hostname": platform.node(),
        "os": os_name,
        "kernel": uname,
        "arch": fields.get("Architecture") or platform.machine(),
        "cpu_model": fields.get("Model name") or platform.processor(),
        "cpu_vendor": fields.get("Vendor ID"),
        "cpu_family": fields.get("CPU family"),
        "cpu_model_number": fields.get("Model"),
        "cpu_stepping": fields.get("Stepping"),
        "logical_cpus": int(fields.get("CPU(s)") or os.cpu_count() or 0),
        "cores_per_socket": fields.get("Core(s) per socket"),
        "sockets": fields.get("Socket(s)"),
        "threads_per_core": fields.get("Thread(s) per core"),
        "hypervisor": fields.get("Hypervisor vendor"),
        "l1d": fields.get("L1d cache"),
        "l1i": fields.get("L1i cache"),
        "l2": fields.get("L2 cache"),
        "l3": fields.get("L3 cache"),
        "cpu_mhz_reported": None,
        "memory_bytes": (mem_kb * 1024) if mem_kb is not None else None,
        "memory_giB": round(mem_kb / (1024 * 1024), 2) if mem_kb is not None else None,
        "note": (
            "CPU frequency is the kernel-reported cpu MHz (no cpufreq sysfs on this VM). "
            "This is a KVM virtual machine; the L3 size is the host cache, not a private slice."
        ),
    }


def add_cpu_mhz(specs: dict) -> None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("cpu mhz"):
                specs["cpu_mhz_reported"] = float(line.split(":")[1].strip())
                return
    except OSError:
        return


def loc_and_bytes(texts: list[str]) -> tuple[int, int]:
    loc = 0
    nbytes = 0
    for t in texts:
        b = t.encode("utf-8")
        nbytes += len(b)
        loc += t.count("\n") + (0 if t.endswith("\n") or not t else 1)
    return loc, nbytes


def list_sol_files(root: Path, src_dirs: list[str], exclude_parts: set[str]) -> list[Path]:
    files: list[Path] = []
    for d in src_dirs:
        base = root / d
        if not base.is_dir():
            raise SystemExit(f"missing source dir: {base}")
        for f in sorted(base.rglob("*.sol")):
            if any(p in exclude_parts for p in f.relative_to(root).parts):
                continue
            files.append(f)
    return files


def resolve_import(importer_key: str, spec: str, remappings: list[tuple[str, str]]) -> str:
    if spec.startswith("."):
        base = Path(importer_key).parent
        return os.path.normpath(str(base / spec))
    for prefix, target in remappings:
        if spec.startswith(prefix):
            return os.path.normpath(target + spec[len(prefix) :])
    return spec


def import_closure(
    roots: list[Path],
    project_root: Path,
    remappings: list[tuple[str, str]],
    extra_roots: list[tuple[str, Path]] | None = None,
) -> dict[str, Path]:
    """Return {source_key: filesystem path} for roots plus recursively imported files."""
    key_to_path: dict[str, Path] = {}
    for f in roots:
        key_to_path[str(f.relative_to(project_root))] = f
    # Map remapped prefixes onto extra roots (e.g. solmate/= <solmate checkout>/)
    prefix_roots = extra_roots or []

    def locate(key: str) -> Path | None:
        if key in key_to_path:
            return key_to_path[key]
        candidate = project_root / key
        if candidate.is_file():
            return candidate
        for prefix, root in prefix_roots:
            if key.startswith(prefix):
                p = root / key[len(prefix) :]
                if p.is_file():
                    return p
        return None

    queue = list(key_to_path.keys())
    seen = set(queue)
    while queue:
        key = queue.pop()
        path = locate(key)
        if path is None:
            raise SystemExit(f"cannot resolve source {key} (imported from the corpus)")
        key_to_path[key] = path
        text = path.read_text(encoding="utf-8", errors="replace")
        for spec in IMPORT_RE.findall(text):
            nxt = resolve_import(key, spec, remappings)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return key_to_path


def write_stdjson(
    dest: Path,
    sources: dict[str, Path],
    remappings: list[str],
    evm_version: str,
    stop_after: str | None,
    full_compile: bool,
    via_ir: bool = False,
    rewrite_exact_08_pragmas: bool = False,
) -> int:
    rewrites = 0
    src: dict[str, dict[str, str]] = {}
    for key, path in sources.items():
        content = path.read_text(encoding="utf-8")
        if rewrite_exact_08_pragmas:
            content, k = widen_exact_08_pragmas(content)
            rewrites += k
        src[key] = {"content": content}
    if full_compile:
        output = {"*": {"*": ["evm.bytecode.object"]}}
    else:
        output = {"*": {"": []}}
    settings: dict = {
        "optimizer": {"enabled": False, "runs": 200},
        "viaIR": bool(via_ir),
        "evmVersion": evm_version,
        "outputSelection": output,
        "remappings": remappings,
    }
    if stop_after:
        settings["stopAfter"] = stop_after
    payload = {"language": "Solidity", "sources": src, "settings": settings}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=None, separators=(",", ":")))
    return rewrites


def run_checked(cmd: list[str], label: str) -> dict:
    raw = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = raw.stdout.decode("utf-8", errors="replace")
    stderr = raw.stderr.decode("utf-8", errors="replace")
    errors = []
    parsed = None
    try:
        parsed = json.loads(stdout) if stdout.strip().startswith("{") else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for err in parsed.get("errors") or []:
            sev = (err.get("severity") or "").lower()
            if sev in {"error", "bug"} or err.get("type") in {"Error", "CompilerError", "FatalError"}:
                errors.append(err.get("message") or json.dumps(err)[:200])
        # solar sometimes uses a different shape
        if not errors and parsed.get("error"):
            errors.append(str(parsed["error"])[:300])
    if raw.returncode != 0 and not errors:
        errors.append(
            (stderr.strip() or stdout.strip() or f"exit {raw.returncode}")[:500]
        )
    if errors:
        joined = "; ".join(errors[:8])
        raise SystemExit(f"{label} failed: {joined}")
    return {"returncode": raw.returncode, "stderr_len": len(stderr), "stdout_len": len(raw.stdout)}


def command_string(argv: list[str]) -> str:
    # hyperfine --shell=none splits this string itself; do not add shell quotes.
    for a in argv:
        if any(c.isspace() for c in a):
            raise SystemExit(f"refusing to pass a whitespace argument with hyperfine -N: {a!r}")
    return " ".join(argv)


def hyperfine_times(argv: list[str], warmup: int, runs: int, export_path: Path, name: str) -> list[float]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    # --shell=none avoids bash startup, which otherwise dominates sub-5 ms parser runs.
    hf = [
        "hyperfine",
        "-N",
        "--style",
        "basic",
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--export-json",
        str(export_path),
        "--command-name",
        name,
        "--",
        command_string(argv),
    ]
    print(f"  hyperfine ({warmup} warmup, {runs} runs): {name}", flush=True)
    proc = subprocess.run(hf, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"hyperfine failed for {name} (exit {proc.returncode})")
    data = json.loads(export_path.read_text())
    times = data["results"][0]["times"]
    if len(times) != runs:
        raise SystemExit(f"hyperfine returned {len(times)} samples, expected {runs}")
    return [float(t) for t in times]


def peak_rss_kb(argv: list[str], runs: int, label: str, out_dir: Path) -> list[int]:
    """Peak resident set size of the compiler process, in kibibytes (GNU time %M)."""
    time_bin = "/usr/bin/time"
    if not Path(time_bin).is_file():
        alt = shutil.which("time")
        if not alt:
            raise SystemExit("GNU time is required for peak RSS measurements")
        time_bin = alt
    samples: list[int] = []
    print(f"  peak RSS: {label} x{runs}", flush=True)
    rss_path = out_dir / f"{label}.rss.txt"
    for i in range(runs):
        # -o captures %M so it is not mixed with the compiler's stderr.
        proc = subprocess.run(
            [time_bin, "-f", "%M", "-o", str(rss_path), "--", *argv],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            raise SystemExit(f"memory run failed for {label} #{i+1} (exit {proc.returncode})")
        try:
            samples.append(int(rss_path.read_text().strip().splitlines()[-1]))
        except ValueError as exc:
            raise SystemExit(f"could not parse GNU time output: {rss_path.read_text()!r}") from exc
    return samples


def fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f" {s * 1e6:.1f} µs"
    if s < 1.0:
        return f" {s * 1e3:.2f} ms"
    return f" {s:.3f} s"


def fmt_kib(kb: float) -> str:
    if kb >= 1024:
        return f"{kb/1024:.1f} MiB"
    return f"{kb:.0f} KiB"


def speedup(solc_med: float | None, solar_med: float | None) -> str:
    if solc_med is None or solar_med is None or solar_med == 0:
        return "—"
    return f"{solc_med / solar_med:.2f}×"


def collect_machine() -> dict:
    specs = machine_specs()
    add_cpu_mhz(specs)
    return specs


def build_corpora() -> list[dict]:
    oz = CORPUS / "openzeppelin-contracts"
    v3 = CORPUS / "v3-core"
    v4 = CORPUS / "v4-core"
    solmate = CORPUS / "solmate"
    for p in (oz, v3, v4, solmate):
        if not (p / ".git").exists():
            raise SystemExit(f"missing corpus {p}; run fetch_corpus.sh first")

    oz_files = list_sol_files(oz, ["contracts"], exclude_parts={"mocks"})
    oz_sources = import_closure(oz_files, oz, [("@openzeppelin/contracts/", "contracts/")])

    v3_files = list_sol_files(v3, ["contracts"], exclude_parts={"test"})
    v3_sources = import_closure(v3_files, v3, [])

    v4_files = list_sol_files(v4, ["src"], exclude_parts={"test"})
    v4_sources = import_closure(
        v4_files,
        v4,
        [("solmate/", "solmate/")],
        extra_roots=[("solmate/", solmate)],
    )

    def meta(name, sources, evm, commit, ref, notes, supported):
        texts = [p.read_text(encoding="utf-8") for p in sources.values()]
        loc, nbytes = loc_and_bytes(texts)
        return {
            "name": name,
            "sources": sources,
            "evm": evm,
            "commit": commit,
            "ref": ref,
            "notes": notes,
            "supported": supported,
            "file_count": len(sources),
            "loc": loc,
            "bytes": nbytes,
        }

    pins = load_pins()
    return [
        meta(
            "openzeppelin-contracts",
            oz_sources,
            "osaka",
            pins["OZ_COMMIT"],
            pins["OZ_REV"],
            "Production contracts/ tree, excluding contracts/mocks/. Remapping @openzeppelin/contracts/=contracts/. evmVersion=osaka, matching the project's foundry.toml.",
            {"solar": ["parsing", "lowering", "full"], "solc": ["parsing", "full"]},
        ),
        meta(
            "uniswap-v3-core",
            v3_sources,
            "istanbul",
            pins["UNISWAP_V3_COMMIT"],
            pins["UNISWAP_V3_REV"],
            "Production contracts/ tree, excluding contracts/test/. Targets Solidity =0.7.6. solc 0.8.39 rejects the pragma, so solc is not compared. Solar parses and lowers the corpus but analysis/codegen fail on 0.7-style implicit integer conversions, so those stages are omitted.",
            {"solar": ["parsing", "lowering"], "solc": []},
        ),
        meta(
            "uniswap-v4-core",
            v4_sources,
            "cancun",
            pins["UNISWAP_V4_COMMIT"],
            pins["UNISWAP_V4_REV"],
            "Production src/ tree, excluding src/test/. Includes the solmate import closure of src/ProtocolFees.sol (solmate @ the v4-core v4.0.0 submodule pin). evmVersion=cancun, matching the project's foundry.toml. Optimizer is off; the project's production via-IR + 44_444_444-runs config is not measured. For solc only, the exact `pragma solidity 0.8.26` in PoolManager.sol is widened to `^0.8.26` so solc 0.8.39 will parse it.",
            {"solar": ["parsing", "lowering", "full"], "solc": ["parsing", "full"]},
        ),
    ]


def compiler_cmd(compiler: str, solar: Path, solc: Path, threads: int) -> list[str]:
    # Stage control is done through Standard JSON `settings.stopAfter`. The Solar
    # CLI `--stop-after` flag is not honoured in `--standard-json` mode (verified
    # against this pin: type-checking still ran on Uniswap v3 when only the flag
    # was set).
    if compiler == "solar":
        return [
            str(solar),
            "--standard-json",
            "--color",
            "never",
            "--no-warnings",
            "--threads",
            str(threads),
        ]
    if compiler == "solc":
        return [str(solc), "--standard-json"]
    raise ValueError(compiler)


def plot(results: dict, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # One figure, two panels: median wall time and median peak RSS.
    # Grouped by corpus, stages along the x axis, solar vs solc as colors.
    jobs = results["jobs"]
    corpora = results["corpora"]
    corpus_names = [c["name"] for c in corpora]
    stages = ["parsing", "lowering", "full"]
    stage_labels = {"parsing": "parse", "lowering": "lowering", "full": "full compile"}

    fig, axes = plt.subplots(2, len(corpus_names), figsize=(13.5, 7.2), sharey=False)
    if len(corpus_names) == 1:
        axes = np.array(axes).reshape(2, 1)
    colors = {"solar": "#E4572E", "solc": "#2E5A88"}

    def lookup(corpus, stage, compiler):
        for j in jobs:
            if j["corpus"] == corpus and j["stage"] == stage and j["compiler"] == compiler:
                return j
        return None

    for col, corpus in enumerate(corpus_names):
        ax_t = axes[0, col]
        ax_m = axes[1, col]
        x = np.arange(len(stages))
        width = 0.36
        for k, compiler in enumerate(("solar", "solc")):
            offset = (k - 0.5) * width
            t_x, t_h, t_err, m_x, m_h, m_err = [], [], [], [], [], []
            for i, st in enumerate(stages):
                job = lookup(corpus, st, compiler)
                if not job:
                    continue
                med = job["time_s"]["median"]
                p95 = job["time_s"]["p95"]
                t_x.append(x[i] + offset)
                t_h.append(med)
                t_err.append(max(0.0, p95 - med))
                mmed = job["peak_rss_kib"]["median"]
                mp95 = job["peak_rss_kib"]["p95"]
                m_x.append(x[i] + offset)
                m_h.append(mmed / 1024.0)
                m_err.append(max(0.0, (mp95 - mmed) / 1024.0))
            if t_x:
                ax_t.bar(
                    t_x, t_h, width, yerr=t_err, color=colors[compiler],
                    label=compiler, capsize=3, error_kw={"linewidth": 0.8},
                )
            if m_x:
                ax_m.bar(
                    m_x, m_h, width, yerr=m_err, color=colors[compiler],
                    label=compiler, capsize=3, error_kw={"linewidth": 0.8},
                )
        for ax in (ax_t, ax_m):
            ax.set_xticks(x)
            ax.set_xticklabels([stage_labels[s] for s in stages])
            ax.set_xlim(-0.6, len(stages) - 0.4)
        ax_t.set_title(corpus, fontsize=11)
        ax_t.set_ylabel("median wall time (s)" if col == 0 else "")
        ax_m.set_ylabel("median peak RSS (MiB)" if col == 0 else "")
        ax_t.set_yscale("log")
        ax_t.set_ylim(1e-3, 4.0)
        missing = []
        for st in stages:
            has_solar = lookup(corpus, st, "solar") is not None
            has_solc = lookup(corpus, st, "solc") is not None
            if not has_solar and not has_solc:
                missing.append(f"no {stage_labels[st]}")
            elif not has_solc:
                missing.append(f"no solc {stage_labels[st]}")
            elif not has_solar:
                missing.append(f"no solar {stage_labels[st]}")
        if missing:
            ax_t.set_xlabel("; ".join(missing), fontsize=7, color="#555")
        if col == 0:
            ax_t.legend(frameon=False, loc="upper left")

    specs = results["machine"]
    fig.suptitle(
        "Solar vs solc — parse, lowering, full compile\n"
        f"{specs['cpu_model']}, {specs['logical_cpus']} cores, {specs['os']}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def write_markdown(results: dict, dest: Path) -> None:
    specs = results["machine"]
    tools = results["tools"]
    lines: list[str] = []
    lines.append("# Solar vs solc — measured results")
    lines.append("")
    lines.append("All numbers were measured on this machine. Nothing here is estimated.")
    lines.append("")
    lines.append("## Machine")
    lines.append("")
    lines.append(f"- CPU: {specs['cpu_model']} (family {specs.get('cpu_family')}, model {specs.get('cpu_model_number')}, stepping {specs.get('cpu_stepping')})")
    lines.append(
        f"- Cores: {specs['logical_cpus']} logical CPUs, "
        f"{specs.get('cores_per_socket')} core(s) per socket, "
        f"{specs.get('threads_per_core')} thread(s) per core, "
        f"{specs.get('sockets')} socket(s)"
    )
    mhz = specs.get("cpu_mhz_reported")
    if mhz:
        lines.append(f"- Kernel-reported cpu MHz: {mhz:.0f} (no cpufreq sysfs)")
    lines.append(f"- Caches: L1d {specs.get('l1d')}, L1i {specs.get('l1i')}, L2 {specs.get('l2')}, L3 {specs.get('l3')}")
    lines.append(f"- Memory: {specs.get('memory_giB')} GiB")
    lines.append(f"- OS: {specs['os']}")
    lines.append(f"- Kernel: `{specs['kernel']}`")
    if specs.get("hypervisor"):
        lines.append(f"- Hypervisor: {specs['hypervisor']}")
    lines.append(f"- Measured at: {results['measured_at_ist']}")
    lines.append("")
    lines.append("## Tool pins")
    lines.append("")
    lines.append(f"- Solar: `{tools['solar_version'].splitlines()[0]}` built from `{tools['solar_rev']}`")
    solc_ver = tools["solc_version"].strip()
    solc_line = next((ln.strip() for ln in solc_ver.splitlines() if ln.strip().lower().startswith("version:")), solc_ver.splitlines()[0])
    if solc_line.lower().startswith("version:"):
        solc_line = solc_line.split(":", 1)[1].strip()
    lines.append(f"- solc: `{solc_line}`")
    lines.append(f"- rustc: `{tools['rustc']}`")
    lines.append(f"- hyperfine: `{tools['hyperfine']}`")
    lines.append(f"- Solar threads: {tools['solar_threads']} (all logical CPUs). solc Standard JSON is effectively single-threaded.")
    lines.append("- Optimizer: off for both compilers (`settings.optimizer.enabled=false`). solc full compile uses `viaIR=true` (required for these corpora); Solar uses its native pipeline. This measures the compilers, not a production optimizer-runs setting.")
    lines.append("")
    lines.append("## Corpora")
    lines.append("")
    for c in results["corpora"]:
        lines.append(
            f"- **{c['name']}** `{c['ref']}` (`{c['commit']}`): "
            f"{c['file_count']} Solidity files, {c['loc']:,} LoC, {c['bytes']:,} bytes. {c['notes']}"
        )
    lines.append("")
    lines.append("## Wall time")
    lines.append("")
    lines.append("Median and p95 over hyperfine runs (parse/lowering: "
                 f"{results['config']['runs']} runs after {results['config']['warmup']} warmup; "
                 f"full compile: {results['config']['full_runs']} runs after {results['config']['full_warmup']} warmup).")
    lines.append("")
    lines.append("| corpus | stage | Solar median | Solar p95 | solc median | solc p95 | speedup (solc/solar, median) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    def job_map():
        m = {}
        for j in results["jobs"]:
            m[(j["corpus"], j["stage"], j["compiler"])] = j
        return m

    jm = job_map()
    for c in results["corpora"]:
        for stage in ("parsing", "lowering", "full"):
            a = jm.get((c["name"], stage, "solar"))
            b = jm.get((c["name"], stage, "solc"))
            if not a and not b:
                continue
            lines.append(
                "| {corpus} | {stage} | {sa} | {sp} | {ca} | {cp} | {x} |".format(
                    corpus=c["name"],
                    stage=stage,
                    sa=fmt_seconds(a["time_s"]["median"]).strip() if a else "—",
                    sp=fmt_seconds(a["time_s"]["p95"]).strip() if a else "—",
                    ca=fmt_seconds(b["time_s"]["median"]).strip() if b else "—",
                    cp=fmt_seconds(b["time_s"]["p95"]).strip() if b else "—",
                    x=speedup(
                        b["time_s"]["median"] if b else None,
                        a["time_s"]["median"] if a else None,
                    ),
                )
            )
    lines.append("")
    lines.append("## Peak memory (RSS)")
    lines.append("")
    lines.append(
        f"Median and p95 of GNU time `%M` (kibibytes) over {results['config']['mem_runs']} runs per job. "
        "This is the compiler process's peak resident set, not the wall-time sample."
    )
    lines.append("")
    lines.append("| corpus | stage | Solar median | Solar p95 | solc median | solc p95 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for c in results["corpora"]:
        for stage in ("parsing", "lowering", "full"):
            a = jm.get((c["name"], stage, "solar"))
            b = jm.get((c["name"], stage, "solc"))
            if not a and not b:
                continue
            lines.append(
                "| {corpus} | {stage} | {sa} | {sp} | {ca} | {cp} |".format(
                    corpus=c["name"],
                    stage=stage,
                    sa=fmt_kib(a["peak_rss_kib"]["median"]) if a else "—",
                    sp=fmt_kib(a["peak_rss_kib"]["p95"]) if a else "—",
                    ca=fmt_kib(b["peak_rss_kib"]["median"]) if b else "—",
                    cp=fmt_kib(b["peak_rss_kib"]["p95"]) if b else "—",
                )
            )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in results["caveats"]:
        lines.append(f"- {c}")
    lines.append("")
    dest.write_text("\n".join(lines) + "\n")


def main() -> int:
    pins = load_pins()
    solar = Path(os.environ.get("SOLAR_BIN", TOOLS / "solar"))
    solc = Path(os.environ.get("SOLC_BIN", TOOLS / "solc"))
    if not solar.is_file() or not solc.is_file():
        raise SystemExit("solar/solc binaries missing; run setup.sh first")
    if not shutil.which("hyperfine"):
        raise SystemExit("hyperfine not found on PATH")

    threads = int(os.environ.get("SOLAR_THREADS") or os.cpu_count() or 1)
    CACHE.mkdir(parents=True, exist_ok=True)

    solar_ver = subprocess.check_output([str(solar), "--version"], text=True).strip()
    solc_ver = subprocess.check_output([str(solc), "--version"], text=True).strip()
    rustc_ver = subprocess.check_output(["rustc", "--version"], text=True).strip()
    hf_ver = subprocess.check_output(["hyperfine", "--version"], text=True).strip()

    specs = collect_machine()
    corpora = build_corpora()

    # IST label: the box clock is Asia/Calcutta.
    measured_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    try:
        # Prefer an explicit IST stamp if TZ is set; also format from local.
        import datetime
        from zoneinfo import ZoneInfo

        now = datetime.datetime.now(ZoneInfo("Asia/Calcutta"))
        measured_at_ist = now.strftime("%Y-%m-%d %H:%M:%S IST (Asia/Calcutta)")
    except Exception:
        measured_at_ist = measured_at + " (box local)"

    jobs_out: list[dict] = []
    caveats = [
        "Numbers are from a single KVM virtual machine and will not match a bare-metal run. The kernel reports the CPU as a generic 'Intel(R) Xeon(R) Processor' at 2400 MHz; there is no cpufreq sysfs, so turbo behaviour is unknown.",
        "Solar is given --threads equal to all logical CPUs (its default). solc's Standard JSON pipeline is effectively single-threaded. That is how each compiler is normally used, but it is not a same-thread-count comparison.",
        "Full compile is measured with the optimizer off, writing Standard JSON (bytecode requested) to /dev/null. Solar uses its native pipeline. solc full compile uses viaIR=true: without it, solc 0.8.39 fails the OpenZeppelin corpus with \"Stack too deep\", and Uniswap v4-core's own Foundry config is via-IR. This is a compiler-performance comparison, not a replay of a project's production optimizer-runs setting.",
        "solc only implements --stop-after parsing, so there is no solc 'lowering' number. Solar --stop-after lowering covers AST-to-HIR lowering, inheritance, name resolution and basic type checking.",
        "Uniswap v3-core v1.0.0 requires Solidity 0.7.6. The pinned solc 0.8.39 rejects that pragma, so v3 has no solc numbers. Solar parses and lowers the production sources, but analysis and codegen fail on 0.7-style implicit integer conversions; those stages are omitted rather than reported as times.",
        "Uniswap v4-core is measured at evmVersion=cancun. OpenZeppelin is measured at evmVersion=osaka. Solar is given the original sources. solc is given the same sources except that exact 0.8.x pragmas (`pragma solidity 0.8.26;` in v4-core's PoolManager.sol) are widened to a caret range so the pinned solc 0.8.39 will accept them. No 0.7.x pragmas are rewritten; v3-core is not compared against solc.",
        "Stage control uses Standard JSON settings.stopAfter (parsing / lowering). Solar's CLI --stop-after is not honoured in --standard-json mode on this pin, so it is not used.",
        "Wall times are collected with hyperfine --shell=none, passing the Standard JSON path as a file argument (no bash redirection). That avoids shell startup dominating the short parse/lowering runs.",
        "Peak RSS is GNU time %M on Linux (kibibytes). It is sampled in a separate pass from hyperfine, so a time sample and a memory sample are not from the same process.",
    ]

    mem_runs = min(MEM_RUNS_CAP, RUNS)

    for corpus in corpora:
        name = corpus["name"]
        print(f"\n== {name}: {corpus['file_count']} files, {corpus['loc']} LoC ==", flush=True)
        for stage, stop, full in (
            ("parsing", "parsing", False),
            ("lowering", "lowering", False),
            ("full", None, True),
        ):
            warmup = FULL_WARMUP if stage == "full" else WARMUP
            runs = FULL_RUNS if stage == "full" else RUNS
            this_mem_runs = min(MEM_RUNS_CAP, runs)
            for compiler in ("solar", "solc"):
                if stage not in corpus["supported"].get(compiler, []):
                    print(f"  skip {compiler}/{stage} (unsupported on this corpus)", flush=True)
                    continue
                tag = f"{name}-{compiler}-{stage}"
                json_path = CACHE / f"{tag}.json"
                # solc only honours stopAfter inside Standard JSON.
                rewrites = write_stdjson(
                    json_path,
                    corpus["sources"],
                    remappings=(
                        ["@openzeppelin/contracts/=contracts/"]
                        if name == "openzeppelin-contracts"
                        else (["solmate/=solmate/"] if name == "uniswap-v4-core" else [])
                    ),
                    evm_version=corpus["evm"],
                    # Honour the stage through Standard JSON so both compilers stop at the
                    # same point. Solar's CLI --stop-after is not applied in this mode.
                    stop_after=stop,
                    full_compile=full,
                    # solc cannot compile these corpora with the legacy pipeline:
                    # OZ fails with "Stack too deep"; v4-core's own Foundry config uses via-IR.
                    # Solar compiles the same input on its native pipeline (viaIR left false).
                    via_ir=(compiler == "solc" and full),
                    # PoolManager.sol (v4-core) pins `pragma solidity 0.8.26;`, which solc
                    # 0.8.39 rejects. Widen exact 0.8.x pragmas only in the solc input.
                    rewrite_exact_08_pragmas=(compiler == "solc"),
                )
                argv = compiler_cmd(compiler, solar, solc, threads) + [str(json_path)]
                print(f"  verify {tag}", flush=True)
                run_checked(argv, tag)
                hf_path = CACHE / f"{tag}.hyperfine.json"
                times = hyperfine_times(argv, warmup, runs, hf_path, tag)
                rss = peak_rss_kb(argv, this_mem_runs, tag, CACHE)
                jobs_out.append(
                    {
                        "corpus": name,
                        "compiler": compiler,
                        "stage": stage,
                        "command": command_string(argv),
                        "solc_pragma_rewrites": rewrites,
                        "via_ir": bool(compiler == "solc" and full),
                        "warmup": warmup,
                        "time_s": {**summarize(times), "samples": times},
                        "peak_rss_kib": {**summarize([float(x) for x in rss]), "samples": rss},
                    }
                )

    # Slim corpora for the committed JSON (no filesystem paths).
    corpora_pub = [
        {k: v for k, v in c.items() if k != "sources"}
        for c in corpora
    ]
    results = {
        "schema": "ternary-gemm.bench.solar/v1",
        "measured_at_ist": measured_at_ist,
        "machine": specs,
        "tools": {
            "solar_rev": pins["SOLAR_REV"],
            "solar_version": solar_ver,
            "solc_version": solc_ver,
            "solc_pin": pins["SOLC_VERSION"],
            "rustc": rustc_ver,
            "hyperfine": hf_ver,
            "solar_threads": threads,
        },
        "config": {
            "warmup": WARMUP,
            "runs": RUNS,
            "full_warmup": FULL_WARMUP,
            "full_runs": FULL_RUNS,
            "mem_runs": mem_runs,
            "optimizer": "off",
            "solc_via_ir_for_full_compile": True,
            "input": "standard-json, sources inlined, file-argument (no stdin redirect)",
            "hyperfine": "--shell=none",
        },
        "corpora": corpora_pub,
        "jobs": jobs_out,
        "caveats": caveats,
        "chart": "chart.png",
    }

    results_json = ROOT / "results.json"
    results_md = ROOT / "results.md"
    chart = ROOT / "chart.png"
    results_json.write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(results, results_md)
    plot(results, chart)
    print(f"\nwrote {results_json}")
    print(f"wrote {results_md}")
    print(f"wrote {chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
