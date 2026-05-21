"""
Bootstrap the crypto prediction model from 90 days of Binance 1-minute candle data.

Strategy
--------
- Fetch 90 days of 1-minute BTC + ETH candles from Binance (no auth required)
- For every 5-minute interval, compute technical indicators on the preceding 70 candles
- Label: 1 (UP) if close price is higher 5 minutes later, 0 (DOWN) otherwise
- Train LogisticRegression (upgrades to LightGBM at >= 100 samples)
- Save to models/crypto_model.pkl

Sentiment features (fear_greed, llm_sentiment, news_count, poly_imbalance) are
unavailable for historical periods — neutral defaults are used. The model learns
primarily from technical indicators; sentiment acts as a real-time adjustment in
live prediction where it is always computed fresh.

Run once:
    python scripts/bootstrap_crypto.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from crypto.indicators import build_indicators
from crypto.model import FEATURE_DEFAULTS, FEATURE_NAMES, CryptoModel
from crypto.price_feed import Candle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("bootstrap_crypto")

BINANCE_BASE = "https://api.binance.com/api/v3"
SYMBOLS      = ["BTCUSDT", "ETHUSDT"]
DAYS         = 90    # days of history to pull
MIN_HISTORY  = 70    # candles needed before making a prediction point
HORIZON      = 5     # label = did price rise N minutes later?
STEP         = 5     # generate one sample per N minutes (avoids autocorrelation)
MODEL_PATH   = _ROOT / "models" / "crypto_model.pkl"

_CLIENT = httpx.Client(timeout=20.0)


def _fetch_candles(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    """Paginate Binance klines, 1000 candles per request."""
    candles: list[Candle] = []
    cursor = start_ms
    batch  = 0

    while cursor < end_ms:
        resp = _CLIENT.get(
            f"{BINANCE_BASE}/klines",
            params={
                "symbol":    symbol,
                "interval":  "1m",
                "startTime": cursor,
                "endTime":   end_ms,
                "limit":     1000,
            },
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break

        for row in rows:
            candles.append(Candle(
                open_time    = row[0],
                open         = float(row[1]),
                high         = float(row[2]),
                low          = float(row[3]),
                close        = float(row[4]),
                volume       = float(row[5]),
                quote_volume = float(row[7]),
            ))

        cursor  = rows[-1][0] + 60_000  # advance past last returned candle
        batch  += 1
        if batch % 25 == 0:
            log.info("  %s: %d candles fetched ...", symbol, len(candles))
        time.sleep(0.05)

    return candles


def _generate_samples(candles: list[Candle]) -> tuple[list[list[float]], list[int]]:
    """Slide a window and produce labeled feature rows."""
    X: list[list[float]] = []
    y: list[int]         = []
    n = len(candles)

    for i in range(MIN_HISTORY, n - HORIZON, STEP):
        label  = 1 if candles[i + HORIZON].close > candles[i].close else 0
        window = candles[i - MIN_HISTORY : i + 1]
        indic  = build_indicators(window)

        row = [
            float(indic[k]) if (k in indic and indic[k] is not None)
            else float(FEATURE_DEFAULTS[k])
            for k in FEATURE_NAMES
        ]
        X.append(row)
        y.append(label)

    return X, y


def main() -> None:
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - DAYS * 24 * 60 * 60 * 1000

    all_X: list[list[float]] = []
    all_y: list[int]         = []

    for symbol in SYMBOLS:
        log.info("Fetching %d days of %s 1m candles from Binance ...", DAYS, symbol)
        candles = _fetch_candles(symbol, start_ms, now_ms)
        log.info("Got %d candles for %s", len(candles), symbol)

        X, y = _generate_samples(candles)
        n_up = sum(y)
        log.info(
            "Samples from %s: %d  (UP=%d %.1f%% | DOWN=%d %.1f%%)",
            symbol, len(y), n_up, n_up / len(y) * 100 if y else 0,
            len(y) - n_up, (len(y) - n_up) / len(y) * 100 if y else 0,
        )
        all_X.extend(X)
        all_y.extend(y)

    X_arr = np.array(all_X, dtype=float)
    y_arr = np.array(all_y, dtype=int)

    total_up = int(y_arr.sum())
    log.info(
        "Total dataset: %d samples | UP=%d (%.1f%%) | DOWN=%d (%.1f%%)",
        len(y_arr), total_up, total_up / len(y_arr) * 100,
        len(y_arr) - total_up, (len(y_arr) - total_up) / len(y_arr) * 100,
    )

    model = CryptoModel(MODEL_PATH)
    model.train(X_arr, y_arr)
    log.info("Bootstrap complete. Model saved to %s  (n=%d)", MODEL_PATH, model.n_samples)


if __name__ == "__main__":
    main()
