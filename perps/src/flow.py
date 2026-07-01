"""
perps/src/flow.py — order-flow signal calculator for Hyperliquid perps.

Direct port of `src/crypto/flow.py` (Polymarket up/down) onto HL perp data.

Key differences from the Polymarket flow:
  - "UP token vs DOWN token" → simply LONG vs SHORT the perp.
  - "price_drift from window-open price" → drift from N seconds ago (rolling).
  - "cross_ob_imbalance across UP+DOWN orderbooks" → single-book imbalance at
    levels within ±N basis points of mid.
  - oracle_lag uses HL's own mark vs oracle (computed inside HLFeed).
  - cvd_score uses HL's own trade-tape CVD (computed inside HLFeed).
  - No CLOB-CVD signal (Polymarket-specific). Replaced by HL CVD.

Score in [-1, +1]:
    score = sum(w_i * signal_i) where missing signals' weight redistributes to drift.

Direction:
    score > +threshold  → LONG
    score < -threshold  → SHORT
    else                → NO_TRADE
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("perps.flow")

# Signal weights — must sum to 1.0 when all signals available.
# Starting weights mirror what's currently profitable on polymarket
# (drift + clob_cvd + oracle_lag + mom_30s), adapted to HL signals.
# Recalibrate after Phase 0.5.
W_DRIFT       = 0.30   # 60s price drift normalized by DRIFT_SCALE
W_OB          = 0.15   # near-mid book imbalance
W_CVD         = 0.25   # taker tape CVD (HL trades)
W_ORACLE_LAG  = 0.15   # mark vs oracle gap
W_FUNDING     = 0.10   # funding rate as contrarian/momentum signal
W_MOM_30S     = 0.05   # last 30s drift — short-horizon momentum

DRIFT_WINDOW_SEC = 60.0    # rolling window for drift
DRIFT_SCALE      = 0.0010  # 0.10% move maps to ±1.0 drift score
MOM_30S_WINDOW   = 30.0
MOM_30S_SCALE    = 0.0005  # 0.05% move in 30s maps to ±1.0

OB_NEAR_BPS      = 5.0     # consider levels within ±5 bps of mid (typical for $0.1 ticks on $77k BTC)
FUNDING_SCALE    = 0.0001  # 1bp/hr funding maps to ±1.0 (HL is hourly funding)


@dataclass
class PerpSignal:
    direction:        str          # "LONG", "SHORT", "NO_TRADE"
    score:            float        # weighted sum in [-1, +1]
    alpha:            float        # |score| - threshold (positive means trade)
    mid:              float        # current mid price
    drift_score:      float
    ob_score:         float
    cvd_score:        float
    oracle_lag_score: float
    funding_score:    float
    mom_30s_score:    float
    components_used:  int          # how many of the 6 signals were available


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── Component signals ──────────────────────────────────────────────────────

def _drift_score(mid_history: list[tuple[float, float]],
                 window_sec: float,
                 scale: float) -> Optional[float]:
    """
    Drift = (latest_mid - mid_at_window_start) / mid_at_window_start / scale.
    Needs at least 5 ticks in the window.
    """
    if len(mid_history) < 5:
        return None
    now = mid_history[-1][0]
    cutoff = now - window_sec
    older = [m for ts, m in mid_history if ts <= cutoff]
    if not older:
        # window not full yet — use oldest available point
        oldest_mid = mid_history[0][1]
    else:
        oldest_mid = older[-1]
    latest_mid = mid_history[-1][1]
    if oldest_mid <= 0:
        return None
    return _clamp((latest_mid - oldest_mid) / oldest_mid / scale)


def _ob_score(bids: list[dict], asks: list[dict],
              near_bps: float = OB_NEAR_BPS) -> Optional[float]:
    """
    Imbalance of bid vs ask depth within ±near_bps of mid.
    Returns [-1, +1] where +1 = all depth is bids (buy pressure), -1 = all asks.
    """
    if not bids or not asks:
        return None
    try:
        bb = float(bids[0]["px"])
        ba = float(asks[0]["px"])
    except Exception:
        return None
    mid = (bb + ba) / 2.0
    if mid <= 0:
        return None
    band = mid * (near_bps / 1e4)
    bid_lo = mid - band
    ask_hi = mid + band

    bid_depth = sum(
        float(l["sz"]) * float(l["px"])
        for l in bids if float(l["px"]) >= bid_lo
    )
    ask_depth = sum(
        float(l["sz"]) * float(l["px"])
        for l in asks if float(l["px"]) <= ask_hi
    )
    total = bid_depth + ask_depth
    if total < 1000.0:  # too little near-mid depth to read
        return None
    return _clamp((bid_depth - ask_depth) / total)


def _funding_score(funding_per_period: Optional[float],
                    scale: float = FUNDING_SCALE) -> Optional[float]:
    """
    Funding rate as a directional signal. Interpretation:
      Positive funding (longs pay shorts) = market crowded long = contrarian SHORT signal.
      Negative funding (shorts pay longs) = market crowded short = contrarian LONG signal.

    This is the standard funding-fade thesis. May or may not work — Phase 0 will tell us.
    Sign: positive funding → negative score (signal goes against the crowd).
    """
    if funding_per_period is None:
        return None
    return _clamp(-funding_per_period / scale)


# ── Combined signal ────────────────────────────────────────────────────────

def compute_signal(
    *,
    mid_history: list[tuple[float, float]],
    bids: list[dict],
    asks: list[dict],
    cvd_score: Optional[float],
    oracle_lag_score: Optional[float],
    funding_rate: Optional[float],
    signal_threshold: float,
) -> PerpSignal:
    """
    Combine all components into a weighted score. Missing components redistribute
    their weight to drift (the most fundamental signal).
    """
    drift   = _drift_score(mid_history, DRIFT_WINDOW_SEC, DRIFT_SCALE)
    mom30   = _drift_score(mid_history, MOM_30S_WINDOW,   MOM_30S_SCALE)
    obs     = _ob_score(bids, asks)
    cvd     = cvd_score
    lag     = oracle_lag_score
    fund    = _funding_score(funding_rate)

    # Tally available signals and redistribute missing weights to drift
    parts: list[tuple[float, float]] = []  # (weight, value)
    if drift is not None:
        parts.append((W_DRIFT, drift))
    drift_extra = 0.0
    for w, v in [
        (W_OB,         obs),
        (W_CVD,        cvd),
        (W_ORACLE_LAG, lag),
        (W_FUNDING,    fund),
        (W_MOM_30S,    mom30),
    ]:
        if v is None:
            drift_extra += w
        else:
            parts.append((w, v))
    # add the recovered weight onto drift if it exists
    if drift is not None and drift_extra > 0:
        parts = [
            (w + drift_extra if i == 0 else w, v)
            for i, (w, v) in enumerate(parts)
        ]
    components_used = sum(1 for v in (drift, obs, cvd, lag, fund, mom30) if v is not None)

    if not parts or (drift is None and components_used < 2):
        return PerpSignal(
            direction="NO_TRADE", score=0.0, alpha=-signal_threshold,
            mid=(mid_history[-1][1] if mid_history else 0.0),
            drift_score=0.0, ob_score=0.0, cvd_score=0.0,
            oracle_lag_score=0.0, funding_score=0.0, mom_30s_score=0.0,
            components_used=components_used,
        )

    score = sum(w * v for w, v in parts)
    alpha = abs(score) - signal_threshold
    if alpha <= 0:
        direction = "NO_TRADE"
    else:
        direction = "LONG" if score > 0 else "SHORT"

    return PerpSignal(
        direction        = direction,
        score            = score,
        alpha            = alpha,
        mid              = mid_history[-1][1],
        drift_score      = drift or 0.0,
        ob_score         = obs or 0.0,
        cvd_score        = cvd or 0.0,
        oracle_lag_score = lag or 0.0,
        funding_score    = fund or 0.0,
        mom_30s_score    = mom30 or 0.0,
        components_used  = components_used,
    )
