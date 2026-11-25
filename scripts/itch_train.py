#!/usr/bin/env python3
"""Train the ITCH order-book signal model: dense f32 MLP, then ternary QAT from it.

Input: the `samples.bin` written by `itch replay --engine none --samples-out samples.bin`
(272-byte records: u64 ts_ns, u32 mid2, u16 locate, u16 pad, f32 features[64], raw/unnormalized).

Target ("next mid-price move"): for each decision, the direction of the next change of the mid
price of the same symbol (up = 1, down = 0). Decisions with no later mid change are dropped.

Chronological split (one trading day, times US/Eastern):
  train 09:30-12:30, val 12:35-13:30, test 13:35-16:00 (5-minute embargo gaps), and a sample is
  only kept if the mid change that labels it also happens before its split ends (no label
  leakage across boundaries). The test period is never used for training, normalization or
  model selection.

Models (same shape as the repo's `tiny` signal model, [64, 256, 256, 1], ReLU):
  * dense: f32 weights, Adam, binary cross-entropy on the logit.
  * qat:   initialized from dense, then fine-tuned with ternary weights (absmean per row) and int8
           per-sample absmax activations in the forward pass, straight-through estimator backward.
           Exported as latent f32 weights; Rust re-ternarizes them with the same absmean rule.
  * PTQ needs no file: running the dense file through the ternary engine is post-training
    ternarization.

Writes <out>/dense.tgmlp, <out>/qat.tgmlp, <out>/split.npz (indices + labels) and train.json.
numpy only (no torch).
"""

import argparse
import json
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
        # index of the next sample (same symbol) whose mid differs
        chg = np.nonzero(np.diff(mid) != 0)[0] + 1  # positions where mid changes from previous
        nxt = np.searchsorted(chg, np.arange(len(mid)), side="right")
        ok = nxt < len(chg)
        j = np.where(ok, chg[np.minimum(nxt, len(chg) - 1)], 0)
        d = mid[j] - mid
        y[idx[ok]] = (d[ok] > 0).astype(np.int8)
        t_next[idx[ok]] = ts[j[ok]]
        # mid 1 s ahead: last sample at or before ts + 1 s
        k = np.searchsorted(ts, ts + 10**9, side="right") - 1
        fwd1s[idx] = (mid[k] - mid) / 200.0  # mid2 units -> ticks (1 tick = 100, mid2 = 2*mid)
        fwd1s[idx[ts + 10**9 > ts[-1]]] = np.nan
    return y, t_next, fwd1s


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
    # average ranks over ties
    s = score[order]
    b = np.r_[0, np.nonzero(np.diff(s))[0] + 1, len(s)]
    for lo, hi in zip(b[:-1], b[1:], strict=True):
        if hi - lo > 1:
            ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    p = y.sum()
    n = len(y) - p
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def train(params, Xtr, ytr, Xva, yva, qat, epochs, lr, bs, seed, log):
    rng = np.random.default_rng(seed)
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best = (-1.0, [(w.copy(), b.copy()) for w, b in params], -1)
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm) - bs + 1, bs):
            i = perm[s : s + bs]
            logit, cache = forward(params, Xtr[i], qat)
            p = 1 / (1 + np.exp(-logit))
            grads = backward(params, cache, (p - ytr[i]) / bs)
            t += 1
            for li, ((w, b), (gw, gb)) in enumerate(zip(params, grads, strict=True)):
                gw = gw + 1e-5 * w  # small L2
                for arr, g, mm, vv in ((w, gw, m[li][0], v[li][0]), (b, gb, m[li][1], v[li][1])):
                    mm *= b1
                    mm += (1 - b1) * g
                    vv *= b2
                    vv += (1 - b2) * g * g
                    arr -= lr * (mm / (1 - b1**t)) / (np.sqrt(vv / (1 - b2**t)) + eps)
        lv, _ = forward(params, Xva, qat)
        a = auc(lv, yva)
        log.append({"qat": qat, "epoch": ep, "val_auc": a, "val_bce": bce(lv, yva), "s": time.time() - t0})
        print(
            f"  {'qat' if qat else 'dense'} epoch {ep}: val AUC {a:.4f} bce {bce(lv, yva):.4f} ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if a > best[0]:
            best = (a, [(w.copy(), b.copy()) for w, b in params], ep)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-train", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--qat-epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--qat-lr", type=float, default=1e-4)
    a = ap.parse_args()

    rec = np.memmap(a.samples, dtype=REC, mode="r")
    print(f"{len(rec)} decisions")
    y, t_next, fwd1s = labels(rec)
    ts = np.asarray(rec["ts"])
    idx = {}
    for name, (lo, hi) in SPLITS.items():
        idx[name] = np.nonzero((ts >= lo) & (ts < hi) & (y >= 0) & (t_next < hi))[0]
        print(f"{name}: {len(idx[name])} labelled decisions, up fraction {y[idx[name]].mean():.4f}")
    np.savez(f"{a.out}/split.npz", y=y, t_next=t_next, fwd1s=fwd1s, **{k: v for k, v in idx.items()})

    rng = np.random.default_rng(a.seed)
    tr = idx["train"]
    if len(tr) > a.max_train:
        tr = np.sort(rng.choice(tr, a.max_train, replace=False))
    Xtr_raw = np.asarray(rec["x"][tr], np.float32)
    mean = Xtr_raw.mean(0)
    std = Xtr_raw.std(0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)  # constant pad features
    norm = lambda X: np.clip((X - mean) / std, -5, 5).astype(np.float32)  # noqa: E731
    Xtr = norm(Xtr_raw)
    del Xtr_raw
    ytr = y[tr].astype(np.float32)
    va = idx["val"]
    Xva, yva = norm(np.asarray(rec["x"][va], np.float32)), y[va].astype(np.float32)

    rng = np.random.default_rng(a.seed + 1)
    params = []
    for k, n in zip(DIMS[:-1], DIMS[1:], strict=True):
        params.append(((rng.standard_normal((n, k)) * np.sqrt(2.0 / k)).astype(np.float32), np.zeros(n, np.float32)))
    log = []
    print("training dense f32")
    dauc, dparams, dep = train(params, Xtr, ytr, Xva, yva, False, a.epochs, a.lr, 1024, a.seed, log)
    export(f"{a.out}/dense.tgmlp", dparams, mean, std)
    print("ternary QAT (init from dense)")
    qparams = [(w.copy(), b.copy()) for w, b in dparams]
    qauc, qparams, qep = train(qparams, Xtr, ytr, Xva, yva, True, a.qat_epochs, a.qat_lr, 1024, a.seed + 7, log)
    export(f"{a.out}/qat.tgmlp", qparams, mean, std)
    json.dump(
        {
            "dims": DIMS,
            "n_train_used": int(len(tr)),
            "splits": {k: int(len(v)) for k, v in idx.items()},
            "dense_best_epoch": dep,
            "dense_val_auc": dauc,
            "qat_best_epoch": qep,
            "qat_val_auc": qauc,
            "log": log,
        },
        open(f"{a.out}/train.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
