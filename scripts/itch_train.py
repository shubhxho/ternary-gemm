#!/usr/bin/env python3
"""Train the ITCH order-book signal model: dense f32 MLP, then ternary QAT from it.

Input: the `samples.bin` written by `itch replay --engine none --samples-out samples.bin`
(272-byte records: u64 ts_ns, u32 mid2, u16 locate, u16 pad, f32 features[64], raw/unnormalized).

Target ("next mid-price move"): for each decision, the direction of the next change of the mid
price of the same symbol (up = 1, down = 0). Decisions with no later mid change are dropped.

Chronological split (one trading day, times US/Eastern) — the official, comparable protocol:
  train 09:30-12:30, val 12:35-13:30, test 13:35-16:00 (5-minute embargo gaps), and a sample is
  only kept if the mid change that labels it also happens before its split ends (no label
  leakage across boundaries). The test period is never used for training, normalization,
  model selection or hyperparameter choices.

Models (same shape as the repo's `tiny` signal model, [64, 256, 256, 1], ReLU):
  * dense: f32 weights, AdamW, binary cross-entropy on the logit.
  * qat:   initialized from dense, then fine-tuned with ternary weights (absmean per row) and int8
           per-sample absmax activations in the forward pass, straight-through estimator backward.
           The improved recipe adds logit distillation from the frozen dense teacher.
           Exported as latent f32 weights; Rust re-ternarizes them with the same absmean rule.
  * PTQ needs no file: running the dense file through the ternary engine is post-training
           ternarization.

Recipes:
  * legacy    — the PR #11 hyperparameters (2M-sample subsample, constant LR, L2 1e-5, no distill)
                so that run can be reproduced exactly.
  * improved  — all labelled train samples, AdamW, cosine LR, dense-teacher distillation on QAT.
                Model selection is still validation AUC only.

Optional `--walk-forward` trains the same recipe on expanding morning-to-afternoon windows and
reports fold AUCs. Those folds are a robustness check; they do not replace the official split
and are not used to pick the official model.

Writes <out>/dense.tgmlp, <out>/qat.tgmlp, <out>/split.npz (indices + labels) and train.json.
numpy only (no torch).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time

import numpy as np

REC = np.dtype([("ts", "<u8"), ("mid2", "<u4"), ("loc", "<u2"), ("pad", "<u2"), ("x", "<f4", (64,))])
H = 3600 * 10**9
M = 60 * 10**9
SPLITS = {
    "train": (int(9.5 * H), int(12.5 * H)),
    "val": (int(12.5 * H) + 5 * M, int(13.5 * H)),
    "test": (int(13.5 * H) + 5 * M, 16 * H),
}
DIMS = [64, 256, 256, 1]

# Expanding-window folds on the same day. Train always starts at 09:30 ET. Each fold has a
# 5-minute embargo between train/val and between val/test. Test labels must fall inside the
# fold's test window (same no-leakage rule as the official split).
WALKFWD = (
    # name, train_hi, val_lo, val_hi, test_lo, test_hi  (hours from midnight ET)
    ("fold1_am", 11.0, 11.0 + 5 / 60, 11.5, 11.5 + 5 / 60, 12.5),
    ("fold2_mid", 12.0, 12.0 + 5 / 60, 12.5, 12.5 + 5 / 60, 13.5),
    ("fold3_early_pm", 13.0, 13.0 + 5 / 60, 13.5, 13.5 + 5 / 60, 14.5),
    ("fold4_late_pm", 14.0, 14.0 + 5 / 60, 14.5, 14.5 + 5 / 60, 16.0),
)


def labels(rec):
    """Direction of the next mid change per symbol, its timestamp, and the 1 s-ahead mid change (ticks)."""
    n = len(rec)
    y = np.full(n, -1, np.int8)
    t_next = np.zeros(n, np.uint64)
    fwd1s = np.full(n, np.nan, np.float32)
    for loc in np.unique(rec["loc"]):
        idx = np.nonzero(rec["loc"] == loc)[0]
        mid = rec["mid2"][idx].astype(np.int64)
        ts = rec["ts"][idx]
        chg = np.nonzero(np.diff(mid) != 0)[0] + 1
        nxt = np.searchsorted(chg, np.arange(len(mid)), side="right")
        ok = nxt < len(chg)
        j = np.where(ok, chg[np.minimum(nxt, len(chg) - 1)], 0)
        d = mid[j] - mid
        y[idx[ok]] = (d[ok] > 0).astype(np.int8)
        t_next[idx[ok]] = ts[j[ok]]
        k = np.searchsorted(ts, ts + 10**9, side="right") - 1
        fwd1s[idx] = (mid[k] - mid) / 200.0
        fwd1s[idx[ts + 10**9 > ts[-1]]] = np.nan
    return y, t_next, fwd1s


def split_indices(ts, y, t_next, lo, hi):
    """Samples in [lo, hi) whose labelling mid-change also occurs before `hi`."""
    return np.nonzero((ts >= lo) & (ts < hi) & (y >= 0) & (t_next < hi))[0]


def official_indices(ts, y, t_next):
    return {name: split_indices(ts, y, t_next, lo, hi) for name, (lo, hi) in SPLITS.items()}


def ternarize(w):
    s = np.maximum(np.abs(w).mean(axis=1, keepdims=True), 1e-5)
    return np.clip(np.round(w / s), -1, 1) * s


def quant_act(x):
    s = 127.0 / np.maximum(np.abs(x).max(axis=1, keepdims=True), 1e-5)
    return np.clip(np.round(x * s), -128, 127) / s


def forward(params, x, qat):
    acts, cache = x, []
    for i, (w, b) in enumerate(params):
        a = quant_act(acts) if qat else acts
        wq = ternarize(w) if qat else w
        z = a @ wq.T + b
        cache.append((a, wq, z))
        acts = np.maximum(z, 0) if i + 1 < len(params) else z
    return acts[:, 0], cache


def backward(params, cache, dlogit):
    grads = [None] * len(params)
    dz = dlogit[:, None].astype(np.float32)
    for i in range(len(params) - 1, -1, -1):
        a, wq, z = cache[i]
        grads[i] = (dz.T @ a, dz.sum(0))
        if i > 0:
            da = dz @ wq  # STE through quant_act
            dz = da * (cache[i - 1][2] > 0)
    return grads


def bce(logit, y):
    return float(np.mean(np.logaddexp(0, logit) - y * logit))


def auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    b = np.r_[0, np.nonzero(np.diff(s))[0] + 1, len(s)]
    for lo, hi in zip(b[:-1], b[1:], strict=True):
        if hi - lo > 1:
            ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    p = y.sum()
    n = len(y) - p
    if p == 0 or n == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def predict(params, x, qat, bs=8192):
    """Logits for `x` in batches (avoids a multi-GB activation graph)."""
    out = np.empty(len(x), np.float32)
    for s in range(0, len(x), bs):
        out[s : s + bs], _ = forward(params, x[s : s + bs], qat)
    return out


def cosine_lr(step, total, lr_max, lr_min):
    if total <= 1:
        return lr_max
    t = min(step, total - 1) / (total - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


def init_params(seed):
    rng = np.random.default_rng(seed + 1)
    params = []
    for k, n in zip(DIMS[:-1], DIMS[1:], strict=True):
        params.append(((rng.standard_normal((n, k)) * np.sqrt(2.0 / k)).astype(np.float32), np.zeros(n, np.float32)))
    return params


def train(params, Xtr, ytr, Xva, yva, qat, epochs, lr, bs, seed, log, weight_decay=1e-5, cosine=False, lr_min=1e-5, teacher=None, distill=0.0):
    """Adam(W) loop. Returns (best_val_auc, best_params, best_epoch).

    If `teacher` and `distill` > 0, the student also matches the teacher's dense logits with MSE
    (the teacher is frozen and always evaluated in dense f32, even when the student is QAT).
    """
    rng = np.random.default_rng(seed)
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best = (-1.0, [(w.copy(), b.copy()) for w, b in params], -1)
    steps_per = max(1, (len(Xtr) - bs + 1) // bs)
    total_steps = max(1, epochs * steps_per)
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm) - bs + 1, bs):
            i = perm[s : s + bs]
            logit, cache = forward(params, Xtr[i], qat)
            p = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
            dlogit = (p - ytr[i]) / bs
            if teacher is not None and distill > 0:
                tlog, _ = forward(teacher, Xtr[i], False)
                dlogit = (1.0 - distill) * dlogit + distill * (2.0 * (logit - tlog) / bs)
            grads = backward(params, cache, dlogit)
            t += 1
            step_lr = cosine_lr(t - 1, total_steps, lr, lr_min) if cosine else lr
            for li, ((w, b), (gw, gb)) in enumerate(zip(params, grads, strict=True)):
                if not cosine:
                    # legacy: L2 is coupled into the weight gradient (PR #11).
                    gw = gw + weight_decay * w
                    decay_w = 0.0
                else:
                    decay_w = weight_decay
                for arr, g, mm, vv, decay in (
                    (w, gw, m[li][0], v[li][0], decay_w),
                    (b, gb, m[li][1], v[li][1], 0.0),
                ):
                    mm *= b1
                    mm += (1 - b1) * g
                    vv *= b2
                    vv += (1 - b2) * g * g
                    hat = (mm / (1 - b1**t)) / (np.sqrt(vv / (1 - b2**t)) + eps)
                    # AdamW (improved recipe): decay is decoupled from the moment estimates.
                    arr -= step_lr * (hat + decay * arr)
        lv = predict(params, Xva, qat)
        a = auc(lv, yva)
        rec = {"qat": qat, "epoch": ep, "val_auc": a, "val_bce": bce(lv, yva), "s": time.time() - t0}
        log.append(rec)
        print(
            f"  {'qat' if qat else 'dense'} epoch {ep}: val AUC {a:.4f} bce {bce(lv, yva):.4f} ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if a > best[0]:
            best = (a, [(w.copy(), b.copy()) for w, b in params], ep)
    # restore best
    for (w, b), (bw, bb) in zip(params, best[1], strict=True):
        w[:] = bw
        b[:] = bb
    return best


def export(path, params, mean, std):
    with open(path, "wb") as f:
        f.write(b"TGMLP1\0\0")
        f.write(struct.pack("<I", len(DIMS)))
        f.write(struct.pack(f"<{len(DIMS)}I", *DIMS))
        f.write(struct.pack("<I", 1))  # relu
        f.write(mean.astype("<f4").tobytes())
        f.write(std.astype("<f4").tobytes())
        for w, b in params:
            f.write(w.astype("<f4").tobytes())
            f.write(b.astype("<f4").tobytes())


def normalize_stats(Xtr_raw):
    mean = Xtr_raw.mean(0)
    std = Xtr_raw.std(0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def apply_norm(X, mean, std):
    return np.clip((X - mean) / std, -5, 5).astype(np.float32)


def load_xy(rec, idx, mean=None, std=None):
    X = np.asarray(rec["x"][idx], np.float32)
    if mean is None:
        mean, std = normalize_stats(X)
    return apply_norm(X, mean, std), mean, std


def recipe_defaults(name):
    if name == "legacy":
        return dict(
            max_train=2_000_000,
            epochs=6,
            qat_epochs=4,
            lr=2e-4,
            qat_lr=1e-4,
            weight_decay=1e-5,
            cosine=False,
            distill=0.0,
        )
    if name == "improved":
        return dict(
            max_train=0,  # all labelled train samples
            epochs=8,
            qat_epochs=6,
            lr=2e-4,
            qat_lr=1.5e-4,
            weight_decay=1e-4,
            cosine=True,
            distill=0.5,
        )
    raise ValueError(f"unknown recipe {name}")


def subsample(idx, max_train, seed):
    if max_train and len(idx) > max_train:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(idx, max_train, replace=False))
    return idx


def run_split(rec, y, tr_idx, va_idx, te_idx, out, args, tag):
    """Train dense + QAT on one chronological split. Returns a metrics dict. Test is scored once
    at the end and is not used for selection."""
    tr = subsample(tr_idx, args.max_train, args.seed)
    Xtr, mean, std = load_xy(rec, tr)
    ytr = y[tr].astype(np.float32)
    Xva, _, _ = load_xy(rec, va_idx, mean, std)
    yva = y[va_idx].astype(np.float32)
    params = init_params(args.seed)
    log = []
    print(f"[{tag}] training dense f32 on {len(tr)} samples (val {len(va_idx)})", flush=True)
    dauc, dparams, dep = train(
        params,
        Xtr,
        ytr,
        Xva,
        yva,
        False,
        args.epochs,
        args.lr,
        1024,
        args.seed,
        log,
        weight_decay=args.weight_decay,
        cosine=args.cosine,
        lr_min=min(args.lr, 1e-5),
    )
    print(f"[{tag}] ternary QAT (init from dense, distill={args.distill})", flush=True)
    qparams = [(w.copy(), b.copy()) for w, b in dparams]
    qauc, qparams, qep = train(
        qparams,
        Xtr,
        ytr,
        Xva,
        yva,
        True,
        args.qat_epochs,
        args.qat_lr,
        1024,
        args.seed + 7,
        log,
        weight_decay=args.weight_decay,
        cosine=args.cosine,
        lr_min=min(args.qat_lr, 1e-5),
        teacher=dparams if args.distill > 0 else None,
        distill=args.distill,
    )
    export(f"{out}/dense.tgmlp", dparams, mean, std)
    export(f"{out}/qat.tgmlp", qparams, mean, std)

    # Held-out scores: official test, or the fold's test. Not used for selection.
    metrics = {
        "tag": tag,
        "n_train_used": int(len(tr)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "dense_best_epoch": dep,
        "dense_val_auc": dauc,
        "qat_best_epoch": qep,
        "qat_val_auc": qauc,
    }
    if len(te_idx):
        Xte, _, _ = load_xy(rec, te_idx, mean, std)
        yte = y[te_idx].astype(np.float32)
        dte = predict(dparams, Xte, False)
        qte = predict(qparams, Xte, True)
        # PTQ: dense weights through the QAT forward (absmean + int8 acts, no fine-tune)
        pte = predict(dparams, Xte, True)
        # L1 queue imbalance is feature 1 of the current 32-wide snapshot (unnormalized).
        imb = np.asarray(rec["x"][te_idx][:, 1], np.float64)
        metrics.update(
            {
                "dense_test_auc": auc(dte, yte),
                "qat_test_auc": auc(qte, yte),
                "ptq_test_auc": auc(pte, yte),
                "imbalance_test_auc": auc(imb, yte),
                "dense_test_bce": bce(dte, yte),
                "qat_test_bce": bce(qte, yte),
            }
        )
        print(
            f"[{tag}] TEST (held out, not used for selection): "
            f"dense AUC {metrics['dense_test_auc']:.4f}  "
            f"QAT {metrics['qat_test_auc']:.4f}  "
            f"PTQ {metrics['ptq_test_auc']:.4f}  "
            f"L1 imbalance {metrics['imbalance_test_auc']:.4f}",
            flush=True,
        )
    metrics["log"] = log
    return metrics


def run_walkforward(rec, y, t_next, ts, root, args):
    """Expanding-window folds. Each fold retrains from scratch; no official-test labels leak in."""
    rows = []
    for name, tr_hi, va_lo, va_hi, te_lo, te_hi in WALKFWD:
        tr = split_indices(ts, y, t_next, int(9.5 * H), int(tr_hi * H))
        va = split_indices(ts, y, t_next, int(va_lo * H), int(va_hi * H))
        te = split_indices(ts, y, t_next, int(te_lo * H), int(te_hi * H))
        print(f"\n== walk-forward {name}: train {len(tr)} val {len(va)} test {len(te)}", flush=True)
        fold_dir = f"{root}/walkforward/{name}"
        import os

        os.makedirs(fold_dir, exist_ok=True)
        met = run_split(rec, y, tr, va, te, fold_dir, args, name)
        # drop the bulky log from the fold table; keep it on disk
        json.dump(met, open(f"{fold_dir}/train.json", "w"), indent=1)
        rows.append({k: v for k, v in met.items() if k != "log"})
    dens = np.array([r["dense_test_auc"] for r in rows], np.float64)
    qats = np.array([r["qat_test_auc"] for r in rows], np.float64)
    ptqs = np.array([r["ptq_test_auc"] for r in rows], np.float64)
    imbs = np.array([r["imbalance_test_auc"] for r in rows], np.float64)
    summary = {
        "folds": rows,
        "dense_test_auc_mean": float(dens.mean()),
        "dense_test_auc_std": float(dens.std(ddof=1)) if len(dens) > 1 else 0.0,
        "qat_test_auc_mean": float(qats.mean()),
        "qat_test_auc_std": float(qats.std(ddof=1)) if len(qats) > 1 else 0.0,
        "ptq_test_auc_mean": float(ptqs.mean()),
        "ptq_test_auc_std": float(ptqs.std(ddof=1)) if len(ptqs) > 1 else 0.0,
        "imbalance_test_auc_mean": float(imbs.mean()),
        "imbalance_test_auc_std": float(imbs.std(ddof=1)) if len(imbs) > 1 else 0.0,
    }
    json.dump(summary, open(f"{root}/walkforward.json", "w"), indent=1)
    lines = [
        "walk-forward expanding windows (same day, 5-minute embargoes; not the official split)",
        "| fold | n_train | n_test | dense AUC | QAT AUC | PTQ AUC | L1 imbalance AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['n_train_used']} | {r['n_test']} | "
            f"{r['dense_test_auc']:.4f} | {r['qat_test_auc']:.4f} | {r['ptq_test_auc']:.4f} | "
            f"{r['imbalance_test_auc']:.4f} |"
        )
    lines.append(
        f"| mean ± std |  |  | {summary['dense_test_auc_mean']:.4f} ± {summary['dense_test_auc_std']:.4f} | "
        f"{summary['qat_test_auc_mean']:.4f} ± {summary['qat_test_auc_std']:.4f} | "
        f"{summary['ptq_test_auc_mean']:.4f} ± {summary['ptq_test_auc_std']:.4f} | "
        f"{summary['imbalance_test_auc_mean']:.4f} ± {summary['imbalance_test_auc_std']:.4f} |"
    )
    text = "\n".join(lines) + "\n"
    open(f"{root}/walkforward.md", "w").write(text)
    print("\n" + text, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--out", default=".")
    ap.add_argument("--recipe", choices=("improved", "legacy"), default="improved")
    ap.add_argument("--max-train", type=int, default=None, help="0 = use every labelled train sample")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--qat-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--qat-lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--distill", type=float, default=None, help="QAT logit-MSE mix with the dense teacher (0 = off)")
    ap.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--walk-forward", action="store_true", help="also run expanding-window folds (robustness only)")
    ap.add_argument("--only-walk-forward", action="store_true", help="skip the official split and only run walk-forward folds")
    a = ap.parse_args()
    dflt = recipe_defaults(a.recipe)
    for k, v in dflt.items():
        attr = k
        if getattr(a, attr) is None:
            setattr(a, attr, v)

    rec = np.memmap(a.samples, dtype=REC, mode="r")
    print(f"{len(rec)} decisions; recipe={a.recipe}")
    y, t_next, fwd1s = labels(rec)
    ts = np.asarray(rec["ts"])
    idx = official_indices(ts, y, t_next)
    for name, ix in idx.items():
        print(f"{name}: {len(ix)} labelled decisions, up fraction {y[ix].mean():.4f}")
    import os

    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/split.npz", y=y, t_next=t_next, fwd1s=fwd1s, **{k: v for k, v in idx.items()})

    official = None
    if not a.only_walk_forward:
        official = run_split(rec, y, idx["train"], idx["val"], idx["test"], a.out, a, "official")
    report = {
        "dims": DIMS,
        "recipe": a.recipe,
        "max_train": a.max_train,
        "epochs": a.epochs,
        "qat_epochs": a.qat_epochs,
        "lr": a.lr,
        "qat_lr": a.qat_lr,
        "weight_decay": a.weight_decay,
        "cosine": a.cosine,
        "distill": a.distill,
        "seed": a.seed,
        "splits": {k: int(len(v)) for k, v in idx.items()},
        "protocol": {
            "split": "train 09:30-12:30, val 12:35-13:30, test 13:35-16:00 ET, 5-minute embargo, no label leakage",
            "target": "direction of the next mid-price change",
            "features": "same 64-dim snapshots as PR #11 / results/itch-20261125",
            "architecture": "[64, 256, 256, 1] ReLU",
            "selection": "validation AUC only; test scored once at the end",
        },
        "previous_official_test": {
            "dense": 0.7094,
            "ternary_qat": 0.7137,
            "ternary_ptq": 0.6763,
            "imbalance": 0.6721,
            "source": "results/itch-20261125/quality.md (Rust replay preds, same split)",
        },
        **({k: v for k, v in official.items() if k != "log"} if official else {}),
        "log": official["log"] if official else [],
    }
    if a.walk_forward or a.only_walk_forward:
        report["walkforward"] = run_walkforward(rec, y, t_next, ts, a.out, a)
    json.dump(report, open(f"{a.out}/train.json", "w"), indent=1)
    if official:
        print(f"wrote {a.out}/dense.tgmlp {a.out}/qat.tgmlp {a.out}/train.json")
    else:
        print(f"wrote {a.out}/walkforward.md {a.out}/train.json")


if __name__ == "__main__":
    main()
