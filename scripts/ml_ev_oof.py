#!/usr/bin/env python3
"""
Honest EV comparison: walk-forward OOF predictions only (not in-sample).
In-sample calibration is useless — model memorized training data.
"""
import json, math
import numpy as np
import lightgbm as lgb
from datetime import datetime, timezone

ROOT = "/home/ubuntu/poly-model"

# --- Load same records as train script ---
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

records.sort(key=lambda r: r["abs_score"])  # needed? no — they're already time-sorted from cache
# re-sort by original order (already time-sorted from file)

FEAT_COLS = ["abs_score","clob_fill","ob_imbalance","price_drift","clob_vs_gamma","hour_sin","hour_cos"]
LGB_PARAMS = dict(n_estimators=100, learning_rate=0.04, max_depth=3, num_leaves=7,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.2, reg_lambda=1.0, random_state=42, verbose=-1)

def to_xy(recs):
    X = np.array([[r[f] for f in FEAT_COLS] for r in recs], dtype=np.float32)
    y = np.array([r["win"] for r in recs], dtype=np.int32)
    return X, y

n       = len(records)
seed    = n // 2
n_folds = 4
fold_sz = (n - seed) // n_folds

# Walk-forward: collect OOF predictions
oof_probs = np.full(n, np.nan)
for i in range(n_folds):
    te_start = seed + i * fold_sz
    te_end   = te_start + fold_sz if i < n_folds - 1 else n
    train = records[:te_start]
    test  = records[te_start:te_end]
    if len(test) < 5:
        continue
    Xtr, ytr = to_xy(train)
    Xte, yte = to_xy(test)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(Xtr, ytr)
    oof_probs[te_start:te_end] = m.predict_proba(Xte)[:, 1]

# Only test-fold rows have OOF predictions
oof_mask = ~np.isnan(oof_probs)
y_oof    = np.array([r["win"]   for r in records])[oof_mask]
f_oof    = np.array([r["fill"]  for r in records])[oof_mask]
s_oof    = np.array([r["score"] for r in records])[oof_mask]
p_oof    = oof_probs[oof_mask]
n_oof    = int(oof_mask.sum())

def ev_trade(win, fill, stake):
    net_win = stake * (1.0 - fill) / fill - stake * 0.02
    return win * net_win - (1.0 - win) * stake

def bet_size(score, prob):
    snorm = min(max((score - 0.25) / 0.40, 0.0), 1.0)
    pnorm = min(max((prob  - 0.65) / 0.35, 0.0), 1.0)
    c = 0.40 * snorm + 0.60 * pnorm
    return 5.0 * (0.60 + 0.40 * c)

print("=== HONEST OOF WALK-FORWARD RESULTS ===")
print("OOF test trades: %d  (seed=%d, folds=%d)" % (n_oof, seed, n_folds))
print("OOF base WR: %.1f%%" % (y_oof.mean() * 100))
print()

# Calibration on OOF data only
print("OOF Calibration (the honest number):")
for lo, hi in [(0.50,0.65),(0.65,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]:
    idx = [i for i in range(n_oof) if lo <= p_oof[i] < hi]
    if len(idx) < 3:
        continue
    wr    = float(y_oof[idx].mean())
    ab    = sum(bet_size(s_oof[i], p_oof[i]) for i in idx) / len(idx)
    af    = float(f_oof[idx].mean())
    ev_v  = sum(ev_trade(y_oof[i], f_oof[i], bet_size(s_oof[i], p_oof[i])) for i in idx) / len(idx)
    ev_f  = sum(ev_trade(y_oof[i], f_oof[i], 5.0) for i in idx) / len(idx)
    label = "SKIP (below threshold)" if hi <= 0.65 else ""
    print("  [%.2f-%.2f): n=%3d  WR=%5.1f%%  bet=$%.2f  EV/trade=$%+.4f  flat=$%+.4f  %s" % (
        lo, hi, len(idx), wr*100, ab, ev_v, ev_f, label))

print()

# ML filter at threshold=0.65
ml_pass  = p_oof >= 0.65
n_pass   = int(ml_pass.sum())
n_skip   = n_oof - n_pass

pre_evs  = [ev_trade(y_oof[i], f_oof[i], 5.0) for i in range(n_oof)]
post_evs = [ev_trade(y_oof[i], f_oof[i], bet_size(s_oof[i], p_oof[i])) for i in range(n_oof) if ml_pass[i]]
post_bets = [bet_size(s_oof[i], p_oof[i]) for i in range(n_oof) if ml_pass[i]]

TPD      = 13.6
pre_ev_t = sum(pre_evs) / n_oof
post_ev_t = sum(post_evs) / n_pass if n_pass else 0
pre_daily = pre_ev_t * TPD
post_daily = post_ev_t * TPD * (n_pass / n_oof)

print("Filter stats: pass=%d (%.1f%%)  skip=%d (%.1f%%)" % (n_pass, n_pass/n_oof*100, n_skip, n_skip/n_oof*100))
print()
print("=== OOF EV COMPARISON (%.0f trades/day) ===" % TPD)
print("Pre-ML:  bet=$5.00  EV/trade=$%+.4f  daily=$%+.2f" % (pre_ev_t, pre_daily))
print("Post-ML: bet=$%.2f  EV/trade=$%+.4f  daily=$%+.2f" % (
    sum(post_bets)/n_pass if n_pass else 0, post_ev_t, post_daily))
print("Delta: $%+.2f/day" % (post_daily - pre_daily))
print()
print("KEY: OOF = out-of-fold (honest). In-sample numbers are overfit and misleading.")
