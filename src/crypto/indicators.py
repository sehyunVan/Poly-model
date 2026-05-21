"""
Technical indicators computed from Binance candle data.
All inputs are lists of floats (close prices or volumes).
"""
from __future__ import annotations

import math
from typing import Optional

from .price_feed import Candle


# ── Core helpers ──────────────────────────────────────────────────────────────

def _returns(closes: list[float]) -> list[float]:
    """Log returns between consecutive closes."""
    out = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev > 0:
            out.append(math.log(closes[i] / prev))
        else:
            out.append(0.0)
    return out


def _sma(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


# ── Indicators ────────────────────────────────────────────────────────────────

def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """
    Wilder RSI. Returns 0–100.
    Returns None if not enough data.
    """
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def momentum_return(closes: list[float], lookback: int) -> Optional[float]:
    """Simple price return over N candles: (close[-1] / close[-1-lookback]) - 1."""
    if len(closes) < lookback + 1:
        return None
    prev = closes[-(lookback + 1)]
    if prev == 0:
        return None
    return (closes[-1] / prev) - 1.0


def bollinger_position(closes: list[float], period: int = 20) -> Optional[float]:
    """
    Where the current price sits within the Bollinger Band.
    Returns 0.0 (at lower band) to 1.0 (at upper band), 0.5 = midpoint.
    Returns None if not enough data or bandwidth is zero.
    """
    if len(closes) < period:
        return None
    window = closes[-period:]
    mean = sum(window) / period
    std = _stdev(window)
    if std == 0:
        return 0.5
    upper = mean + 2 * std
    lower = mean - 2 * std
    band_width = upper - lower
    if band_width == 0:
        return 0.5
    return max(0.0, min(1.0, (closes[-1] - lower) / band_width))


def volume_surge(volumes: list[float], lookback: int = 20) -> Optional[float]:
    """
    Ratio of current volume to the N-period average.
    >1.0 means above-average volume (momentum confirmation).
    """
    if len(volumes) < lookback + 1:
        return None
    avg = sum(volumes[-lookback - 1:-1]) / lookback
    if avg == 0:
        return 1.0
    return volumes[-1] / avg


def ema(values: list[float], period: int) -> list[float]:
    """
    Exponential moving average series.
    Uses the standard multiplier k = 2/(period+1).
    Initialises with a simple average of the first `period` values.
    """
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    seed = sum(values[:period]) / period
    result: list[float] = [seed]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def macd_histogram(
    closes: list[float],
    fast: int = 3,
    slow: int = 15,
    signal: int = 3,
) -> Optional[float]:
    """
    MACD histogram  =  (fast_EMA − slow_EMA) − EMA(signal, fast_EMA − slow_EMA)
    Positive = bullish momentum (buyers in control).
    Negative = bearish momentum.
    Returns None if not enough candles.

    Default parameters (3, 15, 3) are tuned for 1-minute Polymarket 5-min windows:
    fast = 3 min EMA   slow = 15 min EMA   signal = 3 min EMA of MACD line.
    """
    needed = slow + signal - 1   # minimum closes required
    if len(closes) < needed:
        return None

    fast_ema   = ema(closes, fast)
    slow_ema   = ema(closes, slow)
    # Align: fast_ema is longer (or equal) — trim to match slow_ema length
    offset     = len(fast_ema) - len(slow_ema)
    macd_line  = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]
    sig_line   = ema(macd_line, signal)
    if not sig_line:
        return None
    return macd_line[-1] - sig_line[-1]


def macd_score_normalized(
    closes: list[float],
    fast: int = 3,
    slow: int = 15,
    signal: int = 3,
    scale_pct: float = 0.0003,   # histogram of 0.03% of price → ±1.0
) -> Optional[float]:
    """
    MACD histogram normalised to [-1, +1] relative to current price.
    scale_pct = 0.0003 means:
      BTC $80k → scale $24    (histogram > $24 saturates at +1.0)
      ETH $1800 → scale $0.54
    Returns None if not enough data.
    """
    h = macd_histogram(closes, fast, slow, signal)
    if h is None or not closes or closes[-1] <= 0:
        return None
    scale = closes[-1] * scale_pct
    if scale == 0:
        return None
    return max(-1.0, min(1.0, h / scale))


def build_indicators(candles: list[Candle]) -> dict[str, Optional[float]]:
    """
    Given a list of 1-minute candles (oldest first), compute all indicators.
    Returns a dict of feature_name -> value (None if insufficient data).
    """
    closes  = [c.close  for c in candles]
    volumes = [c.quote_volume for c in candles]  # USDT volume

    return {
        "rsi_14":         rsi(closes, 14),
        "momentum_1m":    momentum_return(closes, 1),
        "momentum_5m":    momentum_return(closes, 5),
        "momentum_15m":   momentum_return(closes, 15),
        "momentum_1h":    momentum_return(closes, 60),
        "bb_position_20": bollinger_position(closes, 20),
        "volume_surge":   volume_surge(volumes, 20),
        "volatility_1h":  _stdev(_returns(closes[-61:])) if len(closes) >= 62 else None,
    }
