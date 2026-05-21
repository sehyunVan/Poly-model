#!/usr/bin/env python3
"""Compare pre-ML (flat $5) vs post-ML (variable sizing) expected value."""
import json, math
import numpy as np
import joblib
from datetime import datetime, timezone

ROOT = "/home/ubuntu/poly-model"
p = joblib.load(ROOT + "/models/crypto_filter.pkl")
model = p["model"]
feat_cols = p["feature_cols"]

records = []
with open(ROOT + "/data/crypto_cache.jsonl") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("label") is None:
            continue
        if row.get("symbol") != "BTC":
            continue
        feats = row.get("features", {})
        cf = feats.get("clob_fill")
        if cf is None or not (0.72 <= cf <= 0.80):
            continue
        pred = row["prediction"]
        lbl  = row["label"]
        tw   = 1 if ((pred == "UP") == (lbl == 1)) else 0
        ts   = datetime.fromisoformat(row["ts"]).astimezone(timezone.utc)
        h    = ts.hour
        records.append({
            "abs_score":    abs(feats.get("score", 0.0)),
            "clob_fill":    cf,
            "ob_imbalance": feats.get("ob_imbalance", 0.5),
            "price_drift":  feats.get("price_drift", 0.0),
            "clob_vs_gamma":feats.get("clob_vs_gamma", 0.0) or 0.0,
            "hour_sin":     math.sin(2 * math.pi * h / 24),
            "hour_cos":     math.cos(2 * math.pi * h / 24),
            "fill":         cf,
            "win":          tw,
            "score":        abs(feats.get("score", 0.0)),
        })

X      = np.array([[r[f] for f in feat_cols] for r in records], dtype=np.float32)
y      = np.array([r["win"]   for r in records])
fills  = np.array([r["fill"]  for r in records])
scores = np.array([r["score"] for r in records])
probs  = model.predict_proba(X)[:, 1]
n      = len(records)


def ev_trade(win, fill, stake):
    net_win = stake * (1.0 - fill) / fill - stake * 0.02
    return win * net_win - (1.0 - win) * stake


def bet_size(score, prob):
    snorm = min(max((score - 0.25) / (0.65 - 0.25), 0.0), 1.0)
    pnorm = min(max((prob  - 0.65) / (1.0  - 0.65), 0.0), 1.0)
    c = 0.40 * snorm + 0.60 * pnorm
    return 5.0 * (0.60 + 0.40 * c)


passed = probs >= 0.65
npass  = int(passed.sum())

print("n=%d  base_WR=%.1f%%" % (n, y.mean() * 100))
print("Pass: %d (%.1f%%)  Skip: %d (%.1f%%)" % (
    npass, npass / n * 100, n - npass, (n - npass) / n * 100))
print()

print("Calibration (how well ML prob predicts actual WR):")
for lo, hi in [(0.65, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]:
    idx = [i for i in range(n) if lo <= probs[i] < hi]
    if len(idx) < 3:
        continue
    wr   = float(y[idx].mean())
    ab   = sum(bet_size(scores[i], probs[i]) for i in idx) / len(idx)
    af   = float(fills[idx].mean())
    ev_b = sum(ev_trade(y[i], fills[i], bet_size(scores[i], probs[i])) for i in idx) / len(idx)
    ev_flat = sum(ev_trade(y[i], fills[i], 5.0) for i in idx) / len(idx)
    print("  [%.2f-%.2f): n=%d  WR=%.1f%%  avg_bet=$%.2f  fill=%.3f  EV/trade=$%+.4f  vs flat $%+.4f" % (
        lo, hi, len(idx), wr * 100, ab, af, ev_b, ev_flat))

print()

pre_evs   = [ev_trade(y[i], fills[i], 5.0) for i in range(n)]
post_idx  = [i for i in range(n) if passed[i]]
post_evs  = [ev_trade(y[i], fills[i], bet_size(scores[i], probs[i])) for i in post_idx]
post_bets = [bet_size(scores[i], probs[i]) for i in post_idx]

TPD = 13.6
pre_daily  = sum(pre_evs) / n * TPD
post_daily = sum(post_evs) / npass * TPD * (npass / n)

print("=== DAILY EV COMPARISON (%.0f trades/day) ===" % TPD)
print("Pre-ML:  bet=$5.00  EV/trade=$%.4f  daily_EV=$%.2f" % (sum(pre_evs) / n, pre_daily))
print("Post-ML: bet=$%.2f  EV/trade=$%.4f  daily_EV=$%.2f" % (
    sum(post_bets) / npass, sum(post_evs) / npass, post_daily))
print("Delta: $%+.2f/day" % (post_daily - pre_daily))
print()
print("Note: calibration uses full dataset (in-sample). Walk-forward is the honest test.")
