#!/usr/bin/env python3
"""Unit tests for itch_train.py helpers: AUC, labels, splits (no leakage), ternarization, distill.

Run: python3 scripts/test_itch_train.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("itch_train", ROOT / "scripts" / "itch_train.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_auc():
    y = np.array([0, 0, 1, 1], np.float32)
    check(abs(t.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) - 1.0) < 1e-12, "perfect AUC")
    check(abs(t.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) - 0.0) < 1e-12, "reversed AUC")
    # ties: both positives and negatives share a score
    a = t.auc(np.array([1.0, 1.0, 1.0, 1.0]), y)
    check(abs(a - 0.5) < 1e-12, f"constant score should be 0.5, got {a}")
    # Wilcoxon-Mann-Whitney on a known case
    y2 = np.array([1, 0, 1, 0], np.float32)
    s = np.array([3.0, 2.0, 1.0, 0.0])
    # ranks of positives: 4 and 2; AUC = ((4-1)+(2-1)) / (2*2) = 0.75
    check(abs(t.auc(s, y2) - 0.75) < 1e-12, "known AUC 0.75")


def test_labels_and_no_leakage():
    # one symbol, 6 decisions, mids: 100,100,102,102,98,98 (two mid changes)
    rec = np.zeros(6, t.REC)
    rec["ts"] = np.array([1, 2, 3, 4, 5, 6], np.uint64)
    rec["mid2"] = np.array([100, 100, 102, 102, 98, 98], np.uint32)
    rec["loc"] = 1
    y, t_next, fwd1s = t.labels(rec)
    # samples 0,1: next change is to 102 (up) at ts 3
    # samples 2,3: next change is to 98 (down) at ts 5
    # samples 4,5: no later change
    check(list(y) == [1, 1, 0, 0, -1, -1], f"labels {y}")
    check(list(t_next[:4]) == [3, 3, 5, 5], f"t_next {t_next}")
    # split ending at 5 keeps samples whose labelling change is before 5, i.e. ts-of-change < 5
    # so samples 0,1 (change at 3) yes; 2,3 (change at 5) no
    ix = t.split_indices(rec["ts"], y, t_next, 1, 5)
    check(list(ix) == [0, 1], f"no-leakage split got {ix}")
    # official-style: a sample in a window is dropped if its label arrives after the window
    check(t_next[0] < 5 and t_next[2] >= 5, "change times")


def test_ternarize_matches_absmean():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 16)).astype(np.float32)
    q = t.ternarize(w)
    s = np.abs(w).mean(axis=1)
    codes = np.clip(np.round(w / np.maximum(s[:, None], 1e-5)), -1, 1)
    check(np.allclose(q, codes * s[:, None], atol=0, rtol=0), "ternarize is per-row absmean")
    check(set(np.unique(np.round(codes)).tolist()) <= {-1, 0, 1}, "codes in {-1,0,1}")


def test_distill_and_train_smoke():
    rng = np.random.default_rng(1)
    n, d = 256, 64
    # linearly separable-ish data
    wtrue = rng.standard_normal(d).astype(np.float32)
    X = rng.standard_normal((n, d)).astype(np.float32)
    logit = X @ wtrue
    y = (logit > 0).astype(np.float32)
    # tiny one-layer net via the same helpers: we only check that a short QAT+distill step runs
    # and that distillation changes the weight gradient relative to labels-only.
    W = rng.standard_normal((1, d)).astype(np.float32) * 0.01
    b = np.zeros(1, np.float32)
    params = [(W.copy(), b.copy())]
    teacher = [(wtrue.reshape(1, d).astype(np.float32), np.zeros(1, np.float32))]
    # one forward/backward with and without distill
    logit_s, cache = t.forward(params, X[:32], False)
    p = 1 / (1 + np.exp(-logit_s))
    g0 = t.backward(params, cache, (p - y[:32]) / 32)
    tlog, _ = t.forward(teacher, X[:32], False)
    g1 = t.backward(params, cache, 0.5 * (p - y[:32]) / 32 + 0.5 * (2 * (logit_s - tlog) / 32))
    check(g0[0][0].shape == W.shape, "grad shape")
    check(not np.allclose(g0[0][0], g1[0][0]), "distill changes the weight gradient")

    # a few Adam steps should beat chance on this toy set (val on a hold-out of the same process)
    Xva = rng.standard_normal((128, d)).astype(np.float32)
    yva = ((Xva @ wtrue) > 0).astype(np.float32)
    log = []
    auc0 = t.auc(t.predict(params, Xva, False), yva)
    best = t.train(params, X, y, Xva, yva, False, epochs=3, lr=5e-2, bs=64, seed=0, log=log, weight_decay=0.0)
    check(best[0] > 0.8, f"toy dense val AUC {best[0]} (start {auc0})")
    check(best[0] >= auc0, "training should not return worse than init when it keeps the best ckpt")


def test_cosine_lr():
    check(abs(t.cosine_lr(0, 10, 1.0, 0.0) - 1.0) < 1e-12, "cosine start")
    check(abs(t.cosine_lr(9, 10, 1.0, 0.0) - 0.0) < 1e-12, "cosine end")
    mid = t.cosine_lr(4, 10, 1.0, 0.0)
    check(0.4 < mid < 0.6, f"cosine mid {mid}")


def test_official_windows():
    # embargo of 5 minutes between train/val and val/test, times in ns from midnight
    tr, va, te = t.SPLITS["train"], t.SPLITS["val"], t.SPLITS["test"]
    five = 5 * t.M
    check(va[0] - tr[1] == five, "train/val embargo")
    check(te[0] - va[1] == five, "val/test embargo")
    check(tr[0] == int(9.5 * t.H) and te[1] == 16 * t.H, "RTH bounds")
    # walk-forward folds also embargo and stay inside RTH
    for name, tr_hi, va_lo, va_hi, te_lo, te_hi in t.WALKFWD:
        check(va_lo * t.H - tr_hi * t.H == five or abs((va_lo - tr_hi) * t.H - five) < 1, f"{name} train/val embargo")
        check(te_lo > va_hi, f"{name} val before test")
        check(te_hi <= 16.0 and tr_hi > 9.5, f"{name} inside RTH")


if __name__ == "__main__":
    tests = [
        test_auc,
        test_labels_and_no_leakage,
        test_ternarize_matches_absmean,
        test_distill_and_train_smoke,
        test_cosine_lr,
        test_official_windows,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        sys.exit(1)
    print(f"{len(tests)} tests passed")
