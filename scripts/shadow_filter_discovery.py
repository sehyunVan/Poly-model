"""Filter discovery on shadow loop data (1h + 1d caches).

Combines ~11K unfiltered 5m trades from the two shadow data collectors and
searches for edge zones the live filters currently miss.

Outputs:
  - LightGBM feature importance ranking
  - Per-feature WR buckets (univariate)
  - Hour-of-day x symbol interaction
  - Top 5 high-EV partitions discovered by a shallow decision tree
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

DATA_DIR = Path("/home/ubuntu/poly-model/data")
SHADOW_FILES = ["crypto_1h_cache.jsonl", "crypto_1d_cache.jsonl"]


def load_shadow() -> pd.DataFrame:
    rows = []
    for fn in SHADOW_FILES:
        path = DATA_DIR / fn
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("label") is None:
                    continue
                feats = r.get("features", {}) or {}
                ts = r.get("ts", "")
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                except Exception:
                    hour = -1
                rows.append({
                    "source": fn,
                    "label": int(r["label"]),
                    "symbol": r.get("symbol", "?"),
                    "direction": r.get("prediction", "?"),
                    "bet_size": float(r.get("bet_size", 0) or 0),
                    "hour": hour,
                    "price_drift": float(feats.get("price_drift", 0) or 0),
                    "ob_imbalance": float(feats.get("ob_imbalance", 0) or 0),
                    "score": float(feats.get("score", 0) or 0),
                    "abs_score": abs(float(feats.get("score", 0) or 0)),
                    "clob_fill": float(feats.get("clob_fill", 0) or 0),
                    "clob_vs_gamma": float(feats.get("clob_vs_gamma", 0) or 0),
                    "deribit_pcr": float(feats.get("deribit_pcr", 0) or 0),
                    "tick_velocity": float(feats.get("tick_velocity", 0) or 0),
                    "trade_imbalance": float(feats.get("trade_imbalance", 0) or 0),
                    "oracle_lag": float(feats.get("oracle_lag", 0) or 0),
                    "hawkes": float(feats.get("hawkes", 0) or 0),
                    "mlofi": float(feats.get("mlofi", 0) or 0),
                })
    return pd.DataFrame(rows)


def bucket_wr(df: pd.DataFrame, col: str, bins: list, label: str = None) -> pd.DataFrame:
    label = label or col
    cut = pd.cut(df[col], bins=bins, include_lowest=True)
    g = df.groupby(cut, observed=True)["label"]
    out = pd.DataFrame({
        "n": g.size(),
        "wr": (g.mean() * 100).round(1),
    })
    out["edge_vs_50"] = (out["wr"] - 50.0).round(1)
    return out[out["n"] >= 50]


def symbol_hour(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="hour", columns="symbol", values="label",
        aggfunc=["count", "mean"],
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    for sym in df["symbol"].unique():
        col = f"mean_{sym}"
        if col in pivot.columns:
            pivot[col] = (pivot[col] * 100).round(1)
    return pivot


def main() -> None:
    df = load_shadow()
    print("=" * 70)
    print("SHADOW DATA — overview")
    print("=" * 70)
    print(f"Total trades: {len(df):,}")
    print(f"By source:")
    print(df.groupby("source").agg(n=("label", "size"), wr=("label", lambda s: f"{s.mean()*100:.1f}%")))
    print(f"\nBy symbol:")
    print(df.groupby("symbol").agg(n=("label", "size"), wr=("label", lambda s: f"{s.mean()*100:.1f}%")))
    print(f"\nBy direction (prediction):")
    print(df.groupby("direction").agg(n=("label", "size"), wr=("label", lambda s: f"{s.mean()*100:.1f}%")))

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("UNIVARIATE WR BUCKETS")
    print("=" * 70)
    print("\n[abs_score] (live filter blocks |score| >= 0.65)")
    print(bucket_wr(df, "abs_score", [0, 0.25, 0.40, 0.55, 0.65, 0.80, 1.0, 5.0]))
    print("\n[clob_fill] (live band [0.75, 0.90])")
    print(bucket_wr(df, "clob_fill", [0, 0.50, 0.65, 0.72, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]))
    print("\n[clob_vs_gamma]")
    print(bucket_wr(df, "clob_vs_gamma", [-1.0, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0]))
    print("\n[oracle_lag]")
    print(bucket_wr(df, "oracle_lag", [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]))
    print("\n[hawkes]")
    print(bucket_wr(df, "hawkes", [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]))
    print("\n[mlofi]")
    print(bucket_wr(df, "mlofi", [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]))
    print("\n[deribit_pcr]")
    print(bucket_wr(df, "deribit_pcr", [-5, -1, 0, 0.5, 1, 1.5, 2, 5]))
    print("\n[trade_imbalance]")
    print(bucket_wr(df, "trade_imbalance", [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]))

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOUR-OF-DAY (UTC) WR by symbol — live blocks [0,1,2,5,6]")
    print("=" * 70)
    print(symbol_hour(df).to_string())

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LIGHTGBM — feature importance (gain)")
    print("=" * 70)
    feature_cols = [
        "abs_score", "clob_fill", "clob_vs_gamma", "deribit_pcr",
        "tick_velocity", "trade_imbalance", "oracle_lag", "hawkes",
        "mlofi", "hour", "price_drift", "ob_imbalance",
    ]
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    # 80/20 chronological split (no shuffle — last 20% is most recent)
    split = int(len(df) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(Xtr, ytr)
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    auc_tr = roc_auc_score(ytr, p_tr)
    auc_te = roc_auc_score(yte, p_te)
    print(f"AUC train: {auc_tr:.3f}")
    print(f"AUC test:  {auc_te:.3f}  (>0.55 = real predictive signal)")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(1)
    print("\nFeature importance:")
    print(imp.to_string(index=False))

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HIGH-PROB ZONES — predicted P(win) > 0.60 in test set")
    print("=" * 70)
    pred_df = Xte.copy()
    pred_df["actual"] = yte.values
    pred_df["p_win"] = p_te
    pred_df["symbol"] = df["symbol"].iloc[split:].values
    pred_df["hour"] = df["hour"].iloc[split:].values
    high = pred_df[pred_df["p_win"] >= 0.60]
    if len(high) > 50:
        print(f"n={len(high)}  WR={high['actual'].mean()*100:.1f}%  vs baseline {y.mean()*100:.1f}%")
        print("\nFeature ranges in this zone (median, p25, p75):")
        for f in feature_cols:
            q25, q50, q75 = high[f].quantile([0.25, 0.50, 0.75])
            print(f"  {f:18s} median={q50:7.3f}  p25={q25:7.3f}  p75={q75:7.3f}")
    else:
        print(f"Too few high-prob rows: n={len(high)}")

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DECISION TREE — top interpretable rules (depth=4)")
    print("=" * 70)
    dt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=200, random_state=42,
    )
    dt.fit(Xtr, ytr)
    print(f"Tree test accuracy: {dt.score(Xte, yte):.3f}")
    print()
    print(export_text(dt, feature_names=feature_cols, max_depth=4))

    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LIVE-FILTER SHADOW MAP — what the live loop blocks vs lets through")
    print("=" * 70)

    live_filter = (
        (df["hour"].isin([0, 1, 2, 5, 6])) |
        (df["abs_score"] >= 0.65) |
        (df["clob_fill"] < 0.75) |
        (df["clob_fill"] > 0.90) |
        ((df["clob_fill"] >= 0.80) & (df["abs_score"] < 0.45)) |
        (df["symbol"] == "ETH")
    )
    blocked = df[live_filter]
    passed = df[~live_filter]
    print(f"Live filter blocks: {len(blocked):,}  WR={blocked['label'].mean()*100:.1f}%")
    print(f"Live filter passes: {len(passed):,}  WR={passed['label'].mean()*100:.1f}%")
    print(f"\nWithin BLOCKED — best sub-zones (n>=100, WR>60%):")
    for h in sorted(blocked["hour"].unique()):
        for s in blocked["symbol"].unique():
            sub = blocked[(blocked["hour"] == h) & (blocked["symbol"] == s)]
            if len(sub) >= 100 and sub["label"].mean() >= 0.60:
                print(f"  hour={h:2d}  sym={s}  n={len(sub):4d}  wr={sub['label'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
