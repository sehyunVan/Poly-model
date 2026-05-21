"""Build training data for the strike-scanner pricing model.

Fetches Binance historical 1m OHLCV, generates snapshots every N minutes,
attaches features (realized vol, momentum, distance metrics), and labels
each snapshot against multiple (strike_offset, horizon) combos.

Output: parquet file at data/strike_training/<symbol>_<days>d.parquet

Usage:
    python scripts/build_strike_training_data.py --symbol BTCUSDT --days 90
    python scripts/build_strike_training_data.py --symbol BTCUSDT --days 30 --quick

Phases (each cached so you can re-run without refetching):
    1. Fetch klines       → data/strike_training/raw_<symbol>_<days>d.parquet
    2. Build training set → data/strike_training/<symbol>_<days>d.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strike_training"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Schema constants ──────────────────────────────────────────────────────────
# Strike offsets relative to spot at snapshot time. Sign determines direction:
#   +offset → "above" market (must hit this high)
#   -offset → "below" market (must dip this low)
STRIKE_OFFSETS = [-0.30, -0.20, -0.10, -0.05, -0.02,
                  +0.02, +0.05, +0.10, +0.20, +0.30]
# Horizons in hours from snapshot to expiry.
HORIZONS_H = [1, 4, 12, 24, 72, 168]

SNAPSHOT_INTERVAL_MIN = 60  # generate one training snapshot per hour

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


# ── Phase 1: fetch klines ─────────────────────────────────────────────────────
def fetch_klines(symbol: str, days: int, log: logging.Logger) -> pd.DataFrame:
    """Fetch 1m OHLCV for `days` ending now; paginated 1000-bar requests."""
    cache = DATA_DIR / f"raw_{symbol}_{days}d.parquet"
    if cache.exists():
        log.info("Reading cached klines: %s", cache)
        return pd.read_parquet(cache)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    bars: list[list] = []
    cursor = start_ms

    log.info("Fetching %dd of 1m candles for %s (~%d bars)",
             days, symbol, days * 1440)
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": "1m",
                  "startTime": cursor, "limit": 1000}
        for attempt in range(3):
            try:
                r = requests.get(BINANCE_KLINES, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as exc:
                log.warning("klines request failed (%s), retry %d/3",
                            exc, attempt + 1)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("Binance klines failed after 3 retries")

        if not batch:
            break
        bars.extend(batch)
        cursor = batch[-1][0] + 60_000  # next minute after last bar
        # Be a little gentle on rate limits.
        time.sleep(0.05)

    log.info("Fetched %d bars", len(bars))
    df = pd.DataFrame(bars, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    df = df[["open_time", "open", "high", "low", "close",
             "volume", "quote_volume"]]
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df.to_parquet(cache)
    log.info("Cached raw klines → %s", cache)
    return df


# ── Phase 2: features per snapshot ────────────────────────────────────────────
def add_features(klines: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Compute rolling features on the 1m grid; later we'll sample snapshots."""
    df = klines.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Realized vol (annualized): stdev of 1m log-returns × sqrt(525600)
    ann = np.sqrt(60 * 24 * 365)
    df["rv_1h"]  = df["log_return"].rolling(60).std()  * ann
    df["rv_4h"]  = df["log_return"].rolling(240).std() * ann
    df["rv_24h"] = df["log_return"].rolling(1440).std() * ann
    df["rv_7d"]  = df["log_return"].rolling(10080).std() * ann

    # Cumulative log returns (last N minutes)
    df["ret_1h"]  = np.log(df["close"] / df["close"].shift(60))
    df["ret_4h"]  = np.log(df["close"] / df["close"].shift(240))
    df["ret_24h"] = np.log(df["close"] / df["close"].shift(1440))

    # Distance from 24h high/low
    df["high_24h"] = df["high"].rolling(1440).max()
    df["low_24h"]  = df["low"].rolling(1440).min()
    df["dist_from_high_24h"] = (df["close"] - df["high_24h"]) / df["close"]
    df["dist_from_low_24h"]  = (df["close"] - df["low_24h"])  / df["close"]

    # Volume regime (last 1h vs trailing 24h)
    vol_1h = df["volume"].rolling(60).sum()
    vol_24h = df["volume"].rolling(1440).sum()
    df["vol_ratio_1h_24h"] = vol_1h / (vol_24h / 24)

    # Vol regime (short vs medium)
    df["vol_regime_4h_24h"] = df["rv_4h"] / df["rv_24h"]

    log.info("Features computed; rows=%d, finite-rows=%d",
             len(df), df["rv_24h"].notna().sum())
    return df


# ── Phase 3: build training set ───────────────────────────────────────────────
def build_training_set(features: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Sample snapshots every SNAPSHOT_INTERVAL_MIN, generate labels at each
    (strike_offset, horizon) combo. Skips rows whose horizon would extend past
    the data window or whose features have NaN (warmup).
    """
    df = features.dropna(subset=["rv_24h"]).reset_index(drop=True)
    df = df.set_index("open_time", drop=False)

    # Pre-extract ALL columns as numpy arrays once to avoid pandas iloc overhead
    # (was the bottleneck — 1.7M df.iloc accesses per symbol).
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    times = df["open_time"].values  # ms
    rv_1h = df["rv_1h"].values.astype(np.float64)
    rv_4h = df["rv_4h"].values.astype(np.float64)
    rv_24h_a = df["rv_24h"].values.astype(np.float64)
    rv_7d_a = df["rv_7d"].values.astype(np.float64)
    vol_regime = df["vol_regime_4h_24h"].values.astype(np.float64)
    ret_1h_a = df["ret_1h"].values.astype(np.float64)
    ret_4h_a = df["ret_4h"].values.astype(np.float64)
    ret_24h_a = df["ret_24h"].values.astype(np.float64)
    dist_high = df["dist_from_high_24h"].values.astype(np.float64)
    dist_low = df["dist_from_low_24h"].values.astype(np.float64)
    vol_ratio = df["vol_ratio_1h_24h"].values.astype(np.float64)
    sym_val = df["symbol"].values if "symbol" in df.columns else None

    rows = []
    snap_indices = list(range(0, len(df), SNAPSHOT_INTERVAL_MIN))
    log.info("Building labels over %d snapshots × %d offsets × %d horizons "
             "= %d candidate rows",
             len(snap_indices), len(STRIKE_OFFSETS), len(HORIZONS_H),
             len(snap_indices) * len(STRIKE_OFFSETS) * len(HORIZONS_H))

    n = len(df)
    for i in snap_indices:
        t0 = times[i]
        cur = closes[i]
        if not np.isfinite(cur):
            continue
        sym_i = sym_val[i] if sym_val is not None else ""
        # Snapshot features (read once per snapshot, not per row)
        f_rv_1h = float(rv_1h[i])
        f_rv_4h = float(rv_4h[i])
        f_rv_24h = float(rv_24h_a[i])
        f_rv_7d = float(rv_7d_a[i])
        f_vol_regime = float(vol_regime[i])
        f_ret_1h = float(ret_1h_a[i])
        f_ret_4h = float(ret_4h_a[i])
        f_ret_24h = float(ret_24h_a[i])
        f_dist_high = float(dist_high[i])
        f_dist_low = float(dist_low[i])
        f_vol_ratio = float(vol_ratio[i])

        for horizon_h in HORIZONS_H:
            horizon_min = horizon_h * 60
            j = i + horizon_min
            if j >= n:
                continue
            window_high = float(highs[i:j+1].max())
            window_low = float(lows[i:j+1].min())

            for offset in STRIKE_OFFSETS:
                strike = cur * (1.0 + offset)
                if offset > 0:
                    direction = "above"
                    label = 1 if window_high >= strike else 0
                else:
                    direction = "below"
                    label = 1 if window_low <= strike else 0

                rows.append({
                    "ts": t0,
                    "symbol": sym_i,
                    "current": cur,
                    "strike": strike,
                    "strike_offset": offset,
                    "direction": direction,
                    "hours_to_close": float(horizon_h),
                    "label": label,
                    "log_strike_current": float(np.log(strike / cur)),
                    "rv_1h": f_rv_1h,
                    "rv_4h": f_rv_4h,
                    "rv_24h": f_rv_24h,
                    "rv_7d": f_rv_7d,
                    "vol_regime_4h_24h": f_vol_regime,
                    "ret_1h": f_ret_1h,
                    "ret_4h": f_ret_4h,
                    "ret_24h": f_ret_24h,
                    "dist_from_high_24h": f_dist_high,
                    "dist_from_low_24h": f_dist_low,
                    "vol_ratio_1h_24h": f_vol_ratio,
                })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        log.warning("No training rows generated; data window too short for "
                    "configured horizons (max horizon=%dh, valid rows=%d)",
                    max(HORIZONS_H), len(df))
        return out
    log.info("Generated %d training rows; label balance:\n%s",
             len(out),
             out.groupby(["direction", "hours_to_close"])["label"]
                .agg(["count", "mean"]).round(3))
    return out


# ── CLI entry ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--quick", action="store_true",
                    help="Smaller sweep for testing (fewer offsets/horizons)")
    args = ap.parse_args()

    if args.quick:
        global STRIKE_OFFSETS, HORIZONS_H, SNAPSHOT_INTERVAL_MIN
        STRIKE_OFFSETS = [-0.10, -0.05, -0.02, +0.02, +0.05, +0.10]
        HORIZONS_H = [4, 24]
        SNAPSHOT_INTERVAL_MIN = 240  # every 4h to limit row count

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("strike_training")

    klines = fetch_klines(args.symbol, args.days, log)
    feats = add_features(klines, log)
    feats["symbol"] = args.symbol
    train = build_training_set(feats, log)
    train["symbol"] = args.symbol

    out_path = DATA_DIR / f"{args.symbol}_{args.days}d.parquet"
    train.to_parquet(out_path)
    log.info("Wrote training set: %s (rows=%d, cols=%d)",
             out_path, len(train), len(train.columns))


if __name__ == "__main__":
    main()
