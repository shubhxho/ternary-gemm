#!/usr/bin/env python3
"""Validate and compare benchmark result documents (docs/results-schema.json).

    scripts/compare_results.py validate results/run.json [...]
    scripts/compare_results.py compare base.json new.json [--metric p50_us,p99_us] [--threshold 0.10] [--no-fail]

`compare` matches rows by (model, path), prints a Markdown table of new/base ratios and exits 1
if any metric got slower by more than the threshold (unless --no-fail). It warns when the two
runs come from different machines, since cross-machine ratios are not regressions.
Standard library only; implements the JSON-Schema subset the schema uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent.parent / "docs" / "results-schema.json"
TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "null": type(None),
    "boolean": bool,
}


def _type_ok(value, t) -> bool:
    if isinstance(t, list):
        return any(_type_ok(value, x) for x in t)
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, TYPES[t])


def validate(doc, schema, path="$") -> list[str]:
    """Return a list of error strings (empty if valid)."""
    errs: list[str] = []
    if "const" in schema and doc != schema["const"]:
        errs.append(f"{path}: expected {schema['const']!r}, got {doc!r}")
    if "enum" in schema and doc not in schema["enum"]:
        errs.append(f"{path}: {doc!r} not in {schema['enum']}")
    if "type" in schema and not _type_ok(doc, schema["type"]):
        return errs + [f"{path}: expected type {schema['type']}, got {type(doc).__name__}"]
    if "minimum" in schema and isinstance(doc, (int, float)) and not isinstance(doc, bool):
        if doc < schema["minimum"]:
            errs.append(f"{path}: {doc} < minimum {schema['minimum']}")
    if isinstance(doc, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in doc:
                errs.append(f"{path}: missing required '{key}'")
        for key, value in doc.items():
            if key in props:
                errs += validate(value, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property '{key}'")
    if isinstance(doc, list) and "items" in schema:
        for i, item in enumerate(doc):
            errs += validate(item, schema["items"], f"{path}[{i}]")
    return errs


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cmd_validate(paths: list[str]) -> int:
    schema = load(str(SCHEMA))
    bad = 0
    for p in paths:
        errs = validate(load(p), schema)
        if errs:
            bad += 1
            print(f"{p}: INVALID")
            for e in errs[:20]:
                print(f"  {e}")
        else:
            print(f"{p}: ok")
    return 1 if bad else 0


def cmd_compare(base_p: str, new_p: str, metrics: list[str], threshold: float, fail: bool) -> int:
    schema = load(str(SCHEMA))
    base, new = load(base_p), load(new_p)
    for name, doc in (("base", base), ("new", new)):
        errs = validate(doc, schema)
        if errs:
            print(f"{name} document invalid: {errs[0]}")
            return 2
    be, ne = base["environment"], new["environment"]
    for key in ("cpu", "gpu", "logical_cpus"):
        if be.get(key) != ne.get(key):
            print(f"WARNING: {key} differs ({be.get(key)!r} vs {ne.get(key)!r}); ratios compare machines, not code.")
    rows = {(r["model"], r["path"]): r for r in base["results"]}
    regressions = 0
    print(f"base {be.get('git_sha') or '?'} vs new {ne.get('git_sha') or '?'} (threshold {threshold:.0%})\n")
    print("| model | path | " + " | ".join(f"{m} base | {m} new | ratio" for m in metrics) + " |")
    print("|---|---|" + "---|---|---|" * len(metrics))
    for r in new["results"]:
        b = rows.get((r["model"], r["path"]))
        if b is None:
            continue
        cells = []
        for m in metrics:
            ratio = r[m] / b[m] if b[m] > 0 else float("inf")
            flag = ""
            if ratio > 1 + threshold:
                regressions += 1
                flag = " REGRESSION"
            elif ratio < 1 - threshold:
                flag = " faster"
            cells.append(f"{b[m]:.3f} | {r[m]:.3f} | {ratio:.3f}{flag}")
        print(f"| {r['model']} | {r['path']} | " + " | ".join(cells) + " |")
    missing = set(rows) - {(r["model"], r["path"]) for r in new["results"]}
    for model, path in sorted(missing):
        print(f"(only in base: {model} / {path})")
    print(f"\n{regressions} regression(s) beyond {threshold:.0%}")
    return 1 if regressions and fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--metric", default="p50_us,p99_us")
    c.add_argument("--threshold", type=float, default=0.10)
    c.add_argument("--no-fail", action="store_true")
    a = ap.parse_args()
    if a.cmd == "validate":
        return cmd_validate(a.files)
    return cmd_compare(a.base, a.new, a.metric.split(","), a.threshold, not a.no_fail)


if __name__ == "__main__":
    sys.exit(main())
