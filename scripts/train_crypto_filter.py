#!/usr/bin/env python3
"""
Train a LightGBM EV regressor for the crypto loop entry filter.

The model predicts expected PnL per dollar staked for each candidate signal,
replacing the old win-rate classifier.  A win at fill=0.76 nets +$0.327/dollar;
a win at fill=0.85 nets only +$0.173/dollar.  The old binary target treated both
identically — the EV target does not.

Target:
    net_win     = (1 - fill) / fill * (1 - fee_rate)
    target_epnl = trade_win * net_win − (1 − trade_win)   ∈ [−1.0, +max_net_win]

Gate in loop.py (ml_epnl_threshold):
    pred_epnl ≥ threshold  →  enter trade
    Sizing uses Kelly derived from implied win probability (EV → p → Kelly fraction).

Training data:
    data/crypto_cache.jsonl — BTC/ETH/SOL rows, full fill range [min_fill, max_fill]

Usage:
    cd ~/poly-model
    source .venv/bin/activate
    python scripts/train_crypto_filter.py [--threshold 0.00] [--symbols BTC,ETH,SOL] [--min-fill 0.45] [--max-fill 0.97]

Output:
    models/crypto_filter.pkl  (joblib payload dict, objective="epnl")
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH  = ROOT / "data" / "crypto_cache.jsonl"
OUTPUT_PATH = ROOT / "models" / "crypto_filter.pkl"

FEE_RATE = 0.02   # Polymarket CLOB taker fee

FEATURE_COLS = [
    "abs_score",        # |score| — signal strength (0.25–0.65)
    "clob_fill",        # CLOB ask price paid (full range 0.45–0.97)
    "ob_imbalance",     # order-book pressure ratio (0–1)
    "price_drift",      # current_price − open_price (signed)
    "clob_vs_gamma",    # CLOB fill − Gamma AMM price (information gap)
    "hour_sin",         # cyclical hour encoding — sin(2π·h/24)
    "hour_cos",         # cyclical hour encoding — cos(2π·h/24)
    "deribit_pcr",      # Deribit BTC PCR contrarian score (0 if unavailable)
    "tick_velocity",    # Polymarket CLOB UP-token price velocity (0 if unavailable)
    "trade_imbalance",  # Binance 60s buy/sell notional imbalance (0 if unavailable)
    "oracle_lag",       # Binance vs Chainlink oracle lag (0 if unavailable)
    "hawkes",           # Hawkes decayed buy/sell excitement ratio (0 if unavailable)
    "mlofi",            # Multi-Level OFI from Binance depth (0 if unavailable)
    "symbol_enc",       # symbol encoding: BTC=0, ETH=1, SOL=2
]

_SYMBOL_ENC = {"BTC": 0, "ETH": 1, "SOL": 2}

# Tight regularisation: small dataset, avoid overfitting.
# Regressor uses same tree params as old classifier — only objective differs.
LGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=3,
    num_leaves=7,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(min_fill: float = 0.45, max_fill: float = 0.97,
                 symbols: list[str] | None = None) -> list[dict]:
    """Load and featurise labeled rows from crypto_cache.jsonl across all fills."""
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]
    sym_set = set(symbols)
    records: list[dict] = []

    with CACHE_PATH.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)

            if row.get("label") is None:
                continue
            sym = row.get("symbol", "BTC")
            if sym not in sym_set:
                continue

            feats = row.get("features", {})
            cf = feats.get("clob_fill")
            if cf is None or not (min_fill <= cf <= max_fill):
                continue

            pred      = row["prediction"]
            lbl       = row["label"]
            trade_win = 1 if ((pred == "UP") == (lbl == 1)) else 0

            net_win     = (1.0 - cf) / cf * (1.0 - FEE_RATE)
            target_epnl = float(trade_win) * net_win - (1.0 - float(trade_win))

            ts   = datetime.fromisoformat(row["ts"]).astimezone(timezone.utc)
            hour = ts.hour

            records.append({
                "ts":               ts,
                "symbol":           sym,
                "abs_score":        abs(feats.get("score", 0.0)),
                "clob_fill":        cf,
                "ob_imbalance":     feats.get("ob_imbalance", 0.5),
                "price_drift":      feats.get("price_drift", 0.0),
                "clob_vs_gamma":    feats.get("clob_vs_gamma", 0.0) or 0.0,
                "hour_sin":         math.sin(2 * math.pi * hour / 24),
                "hour_cos":         math.cos(2 * math.pi * hour / 24),
                "deribit_pcr":      feats.get("deribit_pcr", 0.0) or 0.0,
                "tick_velocity":    feats.get("tick_velocity", 0.0) or 0.0,
                "trade_imbalance":  feats.get("trade_imbalance", 0.0) or 0.0,
                "oracle_lag":       feats.get("oracle_lag", 0.0) or 0.0,
                "hawkes":           feats.get("hawkes", 0.0) or 0.0,
                "mlofi":            feats.get("mlofi", 0.0) or 0.0,
                "symbol_enc":       float(_SYMBOL_ENC.get(sym, 0)),
                "trade_win":        trade_win,
                "target_epnl":      target_epnl,
            })

    records.sort(key=lambda r: r["ts"])
    return records


def to_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return feature matrix X and EV targets y (float, not binary)."""
    X = np.array([[r[f] for f in FEATURE_COLS] for r in records], dtype=np.float32)
    y = np.array([r["target_epnl"] for r in records], dtype=np.float32)
    return X, y


# ── PnL simulation ────────────────────────────────────────────────────────────

def simulate_pnl(
    results: list[dict],
    threshold: float,
    bet: float = 5.0,
) -> tuple[float, float, float, int, int]:
    """Simulate cumulative PnL on walk-forward held-out folds at an EV threshold.

    Returns (total_pnl, pass_rate, wr_passed, n_passed, n_total).
    """
    pnl_total = 0.0
    n_passed = n_wins = n_total = 0
    for r in results:
        mask  = r["pred_epnl"] >= threshold
        wins  = r["y_te_win"][mask]
        fills = r["fills_te"][mask]
        n_passed += int(mask.sum())
        n_wins   += int(wins.sum())
        n_total  += len(r["y_te_win"])
        if mask.sum() > 0:
            net_wins  = (1.0 - fills) / fills * (1.0 - FEE_RATE)
            pnl_total += float(np.where(wins == 1, bet * net_wins, -bet).sum())
    pass_rate = n_passed / n_total if n_total else 0.0
    wr        = n_wins   / n_passed if n_passed else 0.0
    return float(pnl_total), pass_rate, wr, n_passed, n_total


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward(records: list[dict], n_folds: int = 4) -> list[dict]:
    """
    Temporal walk-forward using LGBMRegressor on the EV target.
    Train on chronologically earlier data, test on each subsequent fold.
    """
    n = len(records)
    seed = n // 2
    fold_size = (n - seed) // n_folds

    results = []
    for i in range(n_folds):
        te_start = seed + i * fold_size
        te_end   = te_start + fold_size if i < n_folds - 1 else n

        train = records[:te_start]
        test  = records[te_start:te_end]

        if len(test) < 15:
            continue

        X_tr, y_tr = to_xy(train)
        X_te, y_te = to_xy(test)

        y_te_win = np.array([r["trade_win"]  for r in test], dtype=np.int32)
        fills_te = np.array([r["clob_fill"]  for r in test], dtype=np.float32)
        base_wr  = float(y_te_win.mean())
        base_epnl = float(y_te.mean())

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_tr, y_tr)

        pred_epnl = model.predict(X_te)
        results.append({
            "fold":       i + 1,
            "n_train":    len(train),
            "n_test":     len(test),
            "base_wr":    base_wr,
            "base_epnl":  base_epnl,
            "pred_epnl":  pred_epnl,
            "y_te_win":   y_te_win,
            "y_te_epnl":  y_te,
            "fills_te":   fills_te,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.00,
                    help="Minimum predicted EV/dollar to enter (stored in payload)")
    ap.add_argument("--min-fill",  type=float, default=0.45)
    ap.add_argument("--max-fill",  type=float, default=0.97)
    ap.add_argument("--symbols",   type=str,   default="BTC,ETH,SOL",
                    help="Comma-separated symbols to include (e.g. BTC,ETH,SOL)")
    ap.add_argument("--output",    default=str(OUTPUT_PATH))
    args = ap.parse_args()
    args.symbols_list = [s.strip() for s in args.symbols.split(",")]

    print("=" * 60)
    print("  Crypto entry-filter — LightGBM EV regressor training")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    records = load_records(args.min_fill, args.max_fill, args.symbols_list)
    n = len(records)
    if n < 60:
        print(f"ERROR: only {n} labeled rows — need ≥60 to train.")
        sys.exit(1)

    X_all, y_all = to_xy(records)
    base_wr   = float(np.array([r["trade_win"]   for r in records]).mean())
    base_epnl = float(y_all.mean())

    sym_counts = {s: sum(1 for r in records if r["symbol"] == s) for s in args.symbols_list}
    print(f"\nDataset   : {n} trades  [{args.min_fill:.2f}–{args.max_fill:.2f}]  symbols={sym_counts}")
    print(f"Date range: {records[0]['ts'].date()} → {records[-1]['ts'].date()}")
    print(f"Baseline WR  : {base_wr*100:.1f}%")
    print(f"Baseline EV  : {base_epnl:+.4f}/dollar  ({base_epnl*5:+.3f} per $5 bet)")

    # ── Walk-forward validation ────────────────────────────────────
    print("\n--- Walk-forward validation (4 folds, expanding window) ---")
    wf = walk_forward(records, n_folds=4)

    for r in wf:
        print(f"\n  Fold {r['fold']}  "
              f"(train={r['n_train']}, test={r['n_test']}, "
              f"base_wr={r['base_wr']*100:.1f}%  base_ev={r['base_epnl']:+.3f})")
        for th in [-0.10, -0.05, 0.00, 0.02, 0.05, 0.10, 0.20]:
            mask  = r["pred_epnl"] >= th
            n_p   = int(mask.sum())
            if n_p == 0:
                continue
            wins  = r["y_te_win"][mask]
            fills = r["fills_te"][mask]
            wr_p  = float(wins.mean())
            nw    = (1.0 - fills) / fills * (1.0 - FEE_RATE)
            pnl   = float(np.where(wins == 1, 5.0 * nw, -5.0).sum())
            pct_pass = n_p / r["n_test"]
            print(f"    thresh={th:+.2f}  pass={pct_pass*100:4.0f}%  "
                  f"WR={wr_p*100:.1f}%  PnL={pnl:+.2f}  (n={n_p})")

    # ── Combined PnL threshold sweep ──────────────────────────────
    print("\n--- Combined PnL threshold sweep (all folds, $5 bet) ---")
    best_th   = args.threshold
    best_pnl  = -999.0
    best_row  = None

    for th in [-0.10, -0.05, 0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
        pnl, pr, wr, n_p, n_tot = simulate_pnl(wf, th)
        if n_tot == 0 or n_p == 0:
            continue
        marker = ""
        if pnl > best_pnl:
            best_pnl  = pnl
            best_th   = th
            best_row  = (pnl, pr, wr, n_p, n_tot)
            marker    = "  ◄ best"
        print(f"  thresh={th:+.2f}  pass={pr*100:4.0f}%  WR={wr*100:.1f}%  "
              f"PnL={pnl:+.2f}{marker}")

    if best_row:
        pnl, pr, wr, n_p, n_tot = best_row
        print(f"\nRecommended threshold: {best_th:+.2f}  "
              f"(pass={pr*100:.0f}%, WR={wr*100:.1f}%, PnL={pnl:+.2f})")
        if abs(best_th - args.threshold) > 0.03:
            print(f"  (overriding CLI default {args.threshold:+.2f} → {best_th:+.2f})")
            args.threshold = best_th

    # ── Feature importance ─────────────────────────────────────────
    print("\n--- Feature importance (final model trained on all data) ---")
    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X_all, y_all)

    importances = sorted(
        zip(FEATURE_COLS, final.feature_importances_),
        key=lambda x: -x[1],
    )
    for feat, imp in importances:
        bar = "█" * int(imp / max(i for _, i in importances) * 30)
        print(f"  {feat:20s} {imp:5.0f}  {bar}")

    # ── EV calibration check ───────────────────────────────────────
    print("\n--- EV calibration (pred_epnl bucket → actual EV) ---")
    all_preds = np.concatenate([r["pred_epnl"] for r in wf])
    all_wins  = np.concatenate([r["y_te_win"]  for r in wf])
    all_fills = np.concatenate([r["fills_te"]  for r in wf])
    for lo, hi in [(-0.50, 0.00), (0.00, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.00)]:
        mask = (all_preds >= lo) & (all_preds < hi)
        if mask.sum() >= 5:
            actual_wr   = float(all_wins[mask].mean())
            net_wins    = (1.0 - all_fills[mask]) / all_fills[mask] * (1.0 - FEE_RATE)
            actual_epnl = float(np.where(all_wins[mask] == 1, net_wins, -1.0).mean())
            print(f"  pred [{lo:+.2f}–{hi:+.2f}): n={mask.sum():3d}  "
                  f"actual_WR={actual_wr*100:.1f}%  actual_EV={actual_epnl:+.3f}")

    # ── Save ──────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":        final,
        "objective":    "epnl",           # signals loop.py to use EV regressor path
        "feature_cols": FEATURE_COLS,
        "threshold":    float(args.threshold),
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "n_samples":    n,
        "base_wr":      base_wr,
        "base_epnl":    base_epnl,
        "min_fill":     args.min_fill,
        "max_fill":     args.max_fill,
        "symbols":      args.symbols_list,
        "fee_rate":     FEE_RATE,
    }
    joblib.dump(payload, output)
    print(f"\nModel saved → {output}")
    print(f"Objective   → EV regressor (objective='epnl')")
    print(f"EV threshold → {args.threshold:+.2f}")
    print("\nNext steps:")
    print("  1. Add  ml_epnl_threshold: <value>  to config/crypto_params.yaml")
    print("  2. SCP crypto_filter.pkl + crypto_params.yaml to server")
    print("  3. Restart the crypto screen session")


if __name__ == "__main__":
    main()
