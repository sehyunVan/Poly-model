"""
crypto/pairs.py — Cross-asset pairs signal for Polymarket crypto markets.

Strategy: In correlated crypto markets (BTC/ETH/SOL), a strong move in one
asset predicts a follow-through move in lagging assets within minutes.

When BTC surges +0.4% in 60s but ETH has only moved +0.05%, ETH is the
"lagger" — it tends to catch up. This predicts UP on ETH markets.
Conversely, if ETH spiked and BTC hasn't followed, expect UP on BTC.

Signal:
    divergence = leader_60s_return - target_60s_return
    confidence = min(1.0, abs(divergence) / SCALE)
    direction  = "UP" if divergence > threshold (target lagging leader UP)
               = "DOWN" if divergence < -threshold (target leading, likely reverting)
               = "NEUTRAL" otherwise

Integration (in loop.py):
    When pairs.direction AGREES with flow signal → boost bet by pairs_agree_boost
    When pairs.direction DISAGREES with flow signal → skip (conflicting signals)

Data source: BinanceLiveFeed.get_recent_return() — already running in the loop,
zero extra API calls needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from crypto.price_feed import BinanceLiveFeed


# Which asset leads which: for each target asset, the primary comparison asset.
# ETH and SOL both beta off BTC. SOL also has high ETH correlation.
_LEADER_MAP: dict[str, list[str]] = {
    "ETH": ["BTC"],
    "SOL": ["BTC", "ETH"],
    "BTC": ["ETH"],   # ETH occasionally leads BTC (stETH flows, L2 activity)
}

_BINANCE_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Minimum divergence to emit a non-NEUTRAL signal.
# 0.002 = 0.20% difference in 60s returns — meaningful in 5m prediction context.
_MIN_DIVERGENCE = 0.0020

# Scale: this divergence value maps to confidence = 1.0.
# 0.006 = 0.60% gap in 60s returns → maximum confidence.
_CONFIDENCE_SCALE = 0.0060

# Window (seconds) for comparing recent returns.
_WINDOW_SEC = 60.0


@dataclass
class PairsSignal:
    direction:  str    # "UP" | "DOWN" | "NEUTRAL"
    confidence: float  # 0.0 – 1.0
    divergence: float  # leader_return - target_return (signed)
    leader:     str    # which asset moved first
    target:     str    # which asset is lagging


def get_pairs_signal(
    target_symbol: str,
    live_feed: "BinanceLiveFeed",
    window_sec: float = _WINDOW_SEC,
    min_divergence: float = _MIN_DIVERGENCE,
) -> PairsSignal:
    """
    Compute cross-asset pairs signal for target_symbol.

    Args:
        target_symbol: "BTC" | "ETH" | "SOL" — the asset we are considering trading
        live_feed: BinanceLiveFeed instance (already running in loop.py)
        window_sec: lookback window in seconds for return comparison
        min_divergence: minimum divergence to emit a directional signal

    Returns:
        PairsSignal with direction, confidence, and debug fields.
        direction "UP"  → target likely to follow leader upward (buy UP token).
        direction "DOWN"→ target over-extended vs leader, fade back down.
        direction "NEUTRAL" → no significant cross-asset divergence detected.
    """
    leaders = _LEADER_MAP.get(target_symbol.upper(), [])
    if not leaders:
        return PairsSignal("NEUTRAL", 0.0, 0.0, "", target_symbol)

    target_binance = _BINANCE_MAP.get(target_symbol.upper())
    if target_binance is None:
        return PairsSignal("NEUTRAL", 0.0, 0.0, "", target_symbol)

    target_ret = live_feed.get_recent_return(target_binance, window_sec)
    if target_ret is None:
        return PairsSignal("NEUTRAL", 0.0, 0.0, "", target_symbol)

    # Compare against each leader; use the one with the highest absolute divergence.
    best_div  = 0.0
    best_lead = leaders[0]
    for lead in leaders:
        lead_binance = _BINANCE_MAP.get(lead)
        if lead_binance is None:
            continue
        lead_ret = live_feed.get_recent_return(lead_binance, window_sec)
        if lead_ret is None:
            continue
        # divergence > 0 → leader moved UP more than target → target is lagging → expect UP
        div = lead_ret - target_ret
        if abs(div) > abs(best_div):
            best_div  = div
            best_lead = lead

    if abs(best_div) < min_divergence:
        return PairsSignal("NEUTRAL", 0.0, best_div, best_lead, target_symbol)

    confidence = min(1.0, abs(best_div) / _CONFIDENCE_SCALE)

    if best_div > 0:
        direction = "UP"    # target lagging leader's UP move → target likely to follow
    else:
        direction = "DOWN"  # target ran ahead of leader → mean-reversion downward

    return PairsSignal(direction, confidence, best_div, best_lead, target_symbol)


# ── Cache to avoid repeated calls within one loop cycle ──────────────────────

_pairs_cache: dict[str, tuple["PairsSignal", float]] = {}
_CACHE_TTL = 15.0  # seconds — refresh each loop cycle (20s interval)


def get_pairs_signal_cached(
    target_symbol: str,
    live_feed: "BinanceLiveFeed",
    window_sec: float = _WINDOW_SEC,
    min_divergence: float = _MIN_DIVERGENCE,
) -> PairsSignal:
    """
    Cached version of get_pairs_signal. Returns the same result for all
    markets of the same symbol within one 15s window (avoids redundant
    calls when BTC markets appear multiple times per cycle).
    """
    now = time.monotonic()
    cached = _pairs_cache.get(target_symbol)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0]
    sig = get_pairs_signal(target_symbol, live_feed, window_sec, min_divergence)
    _pairs_cache[target_symbol] = (sig, now)
    return sig
