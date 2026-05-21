"""
hyperliquid/breakout.py — 15-minute breakout/breakdown scanner for Hyperliquid perps.

Inspired by moondevonyt's breakout scanner approach:
  - 7-day lookback on 15-minute candles
  - Bollinger Band breach (price statistically extended)
  - Donchian channel breach (price at 7-day high/low)
  - Volume spike confirmation (current vol > 2× 20-bar average)
  - RSI momentum component

Returns a score in [-1, +1]:
  > 0  → bullish breakout (aligns with LONG HL trade)
  < 0  → bearish breakdown (aligns with SHORT HL trade)
  None → insufficient data

Scoring weights:
  BB breach:       40%   (primary — statistically extended)
  Donchian breach: 35%   (primary — actual 7-day high/low)
  Volume spike:    15%   (amplifier — confirms conviction)
  Rate of change:  10%   (momentum tilt)

Usage in hl/loop.py:
  from hyperliquid.breakout import compute_breakout_score
  bk = compute_breakout_score("BTC")
  # Block trade if breakout strongly opposes signal direction
  if bk is not None and bk * signal_sign < -0.4:
      skip()
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

from hyperliquid.feed import get_candles, HLCandle  # type: ignore

_log = logging.getLogger("hyperliquid.breakout")

# ── Config ─────────────────────────────────────────────────────────────────────
BB_PERIOD    = 20        # Bollinger Band lookback (bars)
BB_STDDEV    = 2.0       # Standard deviation multiplier
DON_PERIOD   = 672       # Donchian lookback: 7 days × 24h × 4 bars/h = 672 15-min bars
VOL_PERIOD   = 20        # Volume spike lookback
ROC_PERIOD   = 4         # Rate of change: 4 bars = 1 hour on 15m chart

W_BB  = 0.40
W_DON = 0.35
W_VOL = 0.15
W_ROC = 0.10

# Cache breakout results — 15-min candles change slowly; recalculate every 10 min
_CACHE: dict[str, tuple[Optional[float], float]] = {}   # coin → (score, mono_ts)
_CACHE_TTL = 600.0   # 10 minutes


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_breakout_score(coin: str) -> Optional[float]:
    """
    Return a breakout score in [-1, +1] for coin on 15-minute timeframe (7-day lookback).

    Positive = bullish breakout above BB/Donchian upper band.
    Negative = bearish breakdown below BB/Donchian lower band.
    None if fewer than BB_PERIOD candles available.

    Result is cached for 10 minutes (15-min data doesn't change faster).
    """
    now = time.monotonic()
    cached = _CACHE.get(coin)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0]

    candles = _fetch_candles(coin)
    if len(candles) < BB_PERIOD + 1:
        _log.debug("breakout(%s): only %d candles, need %d", coin, len(candles), BB_PERIOD + 1)
        _CACHE[coin] = (None, now)
        return None

    score = _compute(candles)
    _log.debug("breakout(%s): score=%.3f from %d candles", coin, score, len(candles))
    _CACHE[coin] = (score, now)
    return score


def invalidate_cache(coin: str) -> None:
    """Force re-computation on next call (e.g. after new candle completes)."""
    _CACHE.pop(coin, None)


# ── Internals ─────────────────────────────────────────────────────────────────

def _fetch_candles(coin: str) -> list[HLCandle]:
    """Fetch 7 days of 15-minute candles from Hyperliquid."""
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - 7 * 24 * 3600 * 1000   # 7 days back
    try:
        candles = get_candles(coin, "15m", start_ms, end_ms)
        return candles
    except Exception as exc:
        _log.debug("_fetch_candles(%s) failed: %s", coin, exc)
        return []


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _compute(candles: list[HLCandle]) -> float:
    """
    Compute combined breakout score from the full candle list.
    Uses the last candle as current price vs historical window.
    """
    closes  = [c.close  for c in candles]
    highs   = [c.high   for c in candles]
    lows    = [c.low    for c in candles]
    volumes = [c.volume for c in candles]

    current_close = closes[-1]
    current_vol   = volumes[-1]

    # ── Bollinger Bands (last BB_PERIOD closes) ───────────────────────────────
    bb_window  = closes[-BB_PERIOD - 1 : -1]   # exclude current bar
    bb_mean    = sum(bb_window) / len(bb_window)
    bb_var     = sum((x - bb_mean) ** 2 for x in bb_window) / len(bb_window)
    bb_std     = math.sqrt(bb_var) if bb_var > 0 else 1e-9
    bb_upper   = bb_mean + BB_STDDEV * bb_std
    bb_lower   = bb_mean - BB_STDDEV * bb_std

    # How many standard deviations is current close beyond the band?
    # +1.0 when exactly at upper band, higher for further breach
    if current_close > bb_upper:
        bb_score = _clamp((current_close - bb_upper) / bb_std + 1.0)
    elif current_close < bb_lower:
        bb_score = _clamp((current_close - bb_lower) / bb_std - 1.0)
    else:
        # Inside bands: partial score proportional to position within band
        bb_score = (current_close - bb_mean) / (bb_upper - bb_mean) * 0.5

    # ── Donchian Channel (7-day high/low on all bars except current) ──────────
    don_len    = min(DON_PERIOD, len(candles) - 1)
    don_highs  = highs[-don_len - 1 : -1]
    don_lows   = lows[-don_len - 1 : -1]
    don_high   = max(don_highs)
    don_low    = min(don_lows)
    don_range  = don_high - don_low if don_high != don_low else 1e-9
    don_mid    = (don_high + don_low) / 2

    if current_close >= don_high:
        don_score = 1.0   # at or above 7-day high — strong breakout
    elif current_close <= don_low:
        don_score = -1.0  # at or below 7-day low — strong breakdown
    else:
        # Normalised position within channel: +1 at top, -1 at bottom
        don_score = _clamp((current_close - don_mid) / (don_range / 2))

    # ── Volume spike amplifier ────────────────────────────────────────────────
    vol_window  = volumes[-VOL_PERIOD - 1 : -1]
    avg_vol     = sum(vol_window) / len(vol_window) if vol_window else 1e-9
    vol_ratio   = current_vol / avg_vol if avg_vol > 0 else 1.0
    # +1 when vol is 2× average, 0 at 1×, negative if below average
    vol_score   = _clamp((vol_ratio - 1.0))   # maps 2× → +1, 0.5× → -0.5

    # ── Rate of Change (momentum over last ROC_PERIOD bars) ───────────────────
    if len(closes) > ROC_PERIOD:
        roc_base  = closes[-ROC_PERIOD - 1]
        roc_pct   = (current_close - roc_base) / roc_base if roc_base > 0 else 0.0
        # Normalise: ±0.5% on 15-min → ±1.0 score
        roc_score = _clamp(roc_pct / 0.005)
    else:
        roc_score = 0.0

    score = (W_BB * bb_score + W_DON * don_score
             + W_VOL * vol_score + W_ROC * roc_score)

    _log.debug(
        "breakout detail: bb=%.3f don=%.3f vol=%.3f roc=%.3f → score=%.3f",
        bb_score, don_score, vol_score, roc_score, score,
    )
    return _clamp(score)
