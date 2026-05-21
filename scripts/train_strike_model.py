"""Train a calibrated probability model for the strike-scanner.

Loads training parquets from data/strike_training/, trains a LightGBM
binary classifier with isotonic calibration, and saves a joblib payload
the live scanner can consume.

Output: models/strike_model.pkl

Usage:
    python scripts/train_strike_model.py
    python scripts/train_strike_model.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 90
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "strike_training"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Feature schema (must match live scanner) ─────────────────────────────────
FEATURES = [
    "log_strike_current",
    "hours_to_close",
    "rv_1h", "rv_4h", "rv_24h", "rv_7d",
    "vol_regime_4h_24h",
    "ret_1h", "ret_4h", "ret_24h",
    "dist_from_high_24h", "dist_from_low_24h",
    "vol_ratio_1h_24h",
    "is_above",   # 1 if direction == "above" else 0
    "sym_btc", "sym_eth", "sym_sol",
]

# Vol table used by the current live scanner (so we can compute B-S comparison)
VOL_TABLE = {"BTCUSDT": 0.60, "ETHUSDT": 0.80, "SOLUSDT": 1.00}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(symbols: list[str], days: int) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        path = DATA_DIR / f"{sym}_{days}d.parquet"
        if not path.exists():
            print(f"  ⚠ missing: {path}")
            continue
        df = pd.read_parquet(path)
        df["symbol"] = sym
        frames.append(df)
        print(f"  loaded {sym}: {len(df):,} rows")
    if not frames:
        raise SystemExit("No training data found.")
    return pd.concat(frames, ignore_index=True)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_above"] = (df["direction"] == "above").astype(int)
    df["sym_btc"] = (df["symbol"] == "BTCUSDT").astype(int)
    df["sym_eth"] = (df["symbol"] == "ETHUSDT").astype(int)
    df["sym_sol"] = (df["symbol"] == "SOLUSDT").astype(int)
    return df


# ── Black-Scholes baseline (matches live scanner exactly) ────────────────────
def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_prob(current: float, strike: float, direction: str,
            vol_annual: float, hours_to_close: float) -> float:
    if current <= 0 or strike <= 0 or hours_to_close <= 0:
        return 0.5
    T_years = hours_to_close / (24.0 * 365.0)
    sigma_T = vol_annual * math.sqrt(max(T_years, 1.0 / (525_600.0)))
    if sigma_T <= 1e-9:
        return 1.0 if (
            (direction == "above" and current >= strike) or
            (direction == "below" and current <= strike)
        ) else 0.0
    d2 = (math.log(current / strike) - 0.5 * sigma_T * sigma_T) / sigma_T
    if direction == "above":
        return _normal_cdf(d2)
    return _normal_cdf(-d2)


def add_bs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the live scanner's Black-Scholes probability for each row."""
    df = df.copy()
    bs = np.zeros(len(df))
    for i, row in enumerate(df.itertuples()):
        sym = row.symbol
        vol = VOL_TABLE.get(sym, 0.80)
        bs[i] = bs_prob(row.current, row.strike, row.direction, vol,
                        row.hours_to_close)
    df["bs_prob"] = bs
    return df


# ── Train + calibrate ─────────────────────────────────────────────────────────
def train(df: pd.DataFrame, log_fn=print) -> dict:
    df = df.dropna(subset=FEATURES + ["label"]).sort_values("ts").reset_index(drop=True)
    log_fn(f"  Total rows after dropna: {len(df):,}")

    # 70 / 15 / 15 chronological split: train / cal / test
    n = len(df)
    i_tr = int(n * 0.70)
    i_cal = int(n * 0.85)

    df_tr = df.iloc[:i_tr]
    df_cal = df.iloc[i_tr:i_cal]
    df_te = df.iloc[i_cal:]
    log_fn(f"  Train: {len(df_tr):,}  Cal: {len(df_cal):,}  Test: {len(df_te):,}")

    X_tr = df_tr[FEATURES].values
    y_tr = df_tr["label"].values
    X_cal = df_cal[FEATURES].values
    y_cal = df_cal["label"].values
    X_te = df_te[FEATURES].values
    y_te = df_te["label"].values

    log_fn("  Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        reg_alpha=0.5,
        reg_lambda=0.5,
        objective="binary",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], callbacks=[])

    raw_te = model.predict_proba(X_te)[:, 1]
    auc_te = roc_auc_score(y_te, raw_te)
    bs_te = df_te["bs_prob"].values
    bs_ll = log_loss(y_te, np.clip(bs_te, 1e-6, 1 - 1e-6))
    raw_ll = log_loss(y_te, np.clip(raw_te, 1e-6, 1 - 1e-6))
    log_fn(f"  AUC test (raw): {auc_te:.4f}")
    log_fn(f"  log-loss B-S baseline:  {bs_ll:.4f}")
    log_fn(f"  log-loss raw model:     {raw_ll:.4f}")

    # Isotonic calibration on cal set
    log_fn("  Fitting isotonic calibration...")
    raw_cal = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)

    cal_te = iso.transform(raw_te)
    cal_ll = log_loss(y_te, np.clip(cal_te, 1e-6, 1 - 1e-6))
    cal_brier = brier_score_loss(y_te, cal_te)
    bs_brier = brier_score_loss(y_te, bs_te)
    log_fn(f"  log-loss calibrated:    {cal_ll:.4f}")
    log_fn(f"  Brier B-S:              {bs_brier:.5f}")
    log_fn(f"  Brier calibrated model: {cal_brier:.5f}")

    log_fn(f"\n  Improvement over B-S baseline:")
    log_fn(f"    log-loss reduction:   {(bs_ll - cal_ll):.4f}  ({(bs_ll - cal_ll) / bs_ll * 100:+.1f}%)")
    log_fn(f"    Brier reduction:      {(bs_brier - cal_brier):.5f}  ({(bs_brier - cal_brier) / bs_brier * 100:+.1f}%)")

    # Reliability table — model should be well-calibrated
    log_fn("\n  Calibration table (model vs realized):")
    cal_te_arr = np.asarray(cal_te)
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.0]
    bucket = pd.cut(cal_te_arr, bins=bins, include_lowest=True)
    cal_df = pd.DataFrame({"bucket": bucket, "pred": cal_te_arr, "actual": y_te})
    rel = cal_df.groupby("bucket", observed=True).agg(
        n=("actual", "size"),
        avg_pred=("pred", "mean"),
        actual_rate=("actual", "mean"),
    ).round(4)
    log_fn(rel.to_string())

    # Feature importance
    imp = sorted(zip(FEATURES, model.booster_.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])
    total = sum(g for _, g in imp)
    log_fn("\n  Feature importance:")
    for f, g in imp:
        log_fn(f"    {f:24s}  {g/total*100:5.1f}%")

    return {
        "model": model,
        "calibrator": iso,
        "features": FEATURES,
        "metrics": {
            "auc_test": float(auc_te),
            "logloss_bs": float(bs_ll),
            "logloss_raw": float(raw_ll),
            "logloss_calibrated": float(cal_ll),
            "brier_bs": float(bs_brier),
            "brier_calibrated": float(cal_brier),
            "n_train": len(df_tr),
            "n_cal": len(df_cal),
            "n_test": len(df_te),
        },
        "vol_table": VOL_TABLE,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--output", default=str(MODELS_DIR / "strike_model.pkl"))
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Loading symbols: {symbols} (days={args.days})")
    df = load_data(symbols, args.days)
    df = add_engineered(df)
    print("Computing B-S baseline for comparison...")
    df = add_bs_baseline(df)

    payload = train(df)

    out_path = Path(args.output)
    joblib.dump(payload, out_path)
    print(f"\nSaved model: {out_path}")
    print(f"Summary metrics: {json.dumps(payload['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
