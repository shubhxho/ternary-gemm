#!/usr/bin/env python3
"""Signal quality on the held-out test period for the ITCH replay models and baselines.

usage: itch_eval.py samples.bin split.npz [name=preds.f32 ...] [--json out.json]

Every `preds.f32` is the model output written by `itch replay --preds-out` for every decision,
in replay order (so it lines up with samples.bin). Metrics on the test split (13:35-16:00 ET):
  * AUC and accuracy for the direction of the next mid-price change (up vs down),
  * IC: Spearman and Pearson correlation of the score with the mid change over the next 1 s (ticks),
  * for models: sign agreement with the dense model.
Non-ML baselines use raw features from samples.bin: L1 queue imbalance, microprice - mid and
5-level depth imbalance.
"""

import json
import sys

import numpy as np

REC = np.dtype([("ts", "<u8"), ("mid2", "<u4"), ("loc", "<u2"), ("pad", "<u2"), ("x", "<f4", (64,))])


def rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), np.float64)
    r[order] = np.arange(1, len(a) + 1)
    s = a[order]
    b = np.r_[0, np.nonzero(np.diff(s))[0] + 1, len(s)]
    big = np.nonzero(np.diff(b) > 1)[0]
    for i in big:
        lo, hi = b[i], b[i + 1]
        r[order[lo:hi]] = (lo + 1 + hi) / 2
    return r


def auc(score, y):
    r = rank(score)
    p = y.sum()
    n = len(y) - p
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_json:
        args.remove(out_json)
    rec = np.memmap(args[0], dtype=REC, mode="r")
    sp = np.load(args[1])
    te = sp["test"]
    y = sp["y"][te].astype(np.int64)
    fwd = sp["fwd1s"][te]
    ok = np.isfinite(fwd)
    X = np.asarray(rec["x"][te][:, :32], np.float32)
    scores = {
        "baseline: L1 queue imbalance": X[:, 1],
        "baseline: microprice - mid": X[:, 9],
        "baseline: 5-level depth imbalance": X[:, 4],
    }
    models = {}
    for kv in args[2:]:
        name, path = kv.split("=", 1)
        p = np.fromfile(path, "<f4")
        assert len(p) == len(rec), f"{path}: {len(p)} preds vs {len(rec)} decisions"
        models[name] = p[te]
    rows = []
    dense = models.get("dense f32")
    fr = rank(fwd[ok])
    for name, s in {**models, **scores}.items():
        s = np.asarray(s, np.float64)
        row = {
            "model": name,
            "n_test": int(len(y)),
            "auc_next_move": auc(s, y),
            "acc_next_move": float(((s > 0) == (y == 1)).mean()),
            "ic_spearman_1s": corr(rank(s[ok]), fr),
            "ic_pearson_1s": corr(s[ok], fwd[ok].astype(np.float64)),
        }
        if dense is not None and name in models:
            row["sign_agreement_vs_dense"] = float(((s > 0) == (dense > 0)).mean())
        rows.append(row)
    print(f"test decisions: {len(y)} (up fraction {y.mean():.4f}); 1 s IC on {ok.sum()} with a full 1 s horizon")
    print("| model | AUC (next mid move) | accuracy | IC Spearman (1 s) | IC Pearson (1 s) | sign agreement vs dense |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        ag = f"{r['sign_agreement_vs_dense']:.4f}" if "sign_agreement_vs_dense" in r else "-"
        print(
            f"| {r['model']} | {r['auc_next_move']:.4f} | {r['acc_next_move']:.4f} | {r['ic_spearman_1s']:.4f} | {r['ic_pearson_1s']:.4f} | {ag} |"
        )
    # per-symbol AUC for the models
    locs = np.asarray(rec["loc"][te])
    per = {}
    for name, s in models.items():
        per[name] = {int(loc): auc(np.asarray(s[locs == loc], np.float64), y[locs == loc]) for loc in np.unique(locs)}
    print("per-symbol AUC (locate -> AUC):", json.dumps(per))
    if out_json:
        json.dump({"rows": rows, "per_symbol_auc": per}, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
