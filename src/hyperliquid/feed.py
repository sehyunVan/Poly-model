"""
hyperliquid/feed.py — Hyperliquid public REST data layer.

All queries go to the single POST endpoint https://api.hyperliquid.xyz/info.
No authentication is required for any function in this module.

Provides:
  get_mid_price()            — current mid price for BTC or ETH
  get_funding()              — current funding rate + OI
  get_recent_liquidations()  — long/short liquidation notional in last N seconds
  get_liq_imbalance_score()  — [-1,+1] derived from liquidation imbalance
  get_funding_score()        — [-1,+1] contrarian fade of extreme funding
  get_candles()              — historical OHLCV for backtesting
  get_sz_decimals()          — per-coin contract size precision

Results for funding and liquidations are cached in memory to avoid duplicate
API calls when loop.py and flow.py both need data in the same cycle.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

_log = logging.getLogger("hyperliquid.feed")

_HL_BASE   = "https://api.hyperliquid.xyz/info"
_HL_CLIENT = httpx.Client(timeout=8.0)

# ── In-memory caches ──────────────────────────────────────────────────────────
# Each cache entry: (data, monotonic_timestamp)
_FUNDING_CACHE: dict[str, tuple] = {}   # coin → (HLFundingSnapshot, float)
_LIQ_CACHE:    dict[str, tuple] = {}   # coin → (HLLiqSnapshot, float)
_SZ_DEC_CACHE: Optional[dict[str, int]] = None
_SZ_DEC_TS:    float = 0.0

_FUNDING_TTL = 30.0   # seconds — funding rate changes every 8h, 30s is fine
_LIQ_TTL     = 15.0   # seconds — liquidations are time-sensitive
_SZ_DEC_TTL  = 3600.0 # seconds — contract metadata rarely changes


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class HLCandle:
    open_time: int    # ms epoch (window start)
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float  # base asset (BTC or ETH)


@dataclass
class HLFundingSnapshot:
    coin:         str
    funding_rate: float   # per-8h rate (e.g. 0.0001 = 0.01% per 8h)
    open_interest: float  # USD open interest
    mid_price:    float


@dataclass
class HLLiqSnapshot:
    coin:           str
    long_liq_usd:   float   # USD notional of long positions liquidated in window
    short_liq_usd:  float   # USD notional of short positions liquidated in window
    window_seconds: int
    total_usd:      float = field(init=False)

    def __post_init__(self):
        self.total_usd = self.long_liq_usd + self.short_liq_usd


# ── Core HTTP helper ──────────────────────────────────────────────────────────

def _hl_post(body: dict) -> object:
    """POST to HL info endpoint. Returns parsed JSON. Raises on HTTP error."""
    try:
        r = _HL_CLIENT.post(_HL_BASE, json=body)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        _log.debug("HL API HTTP error: %s  body=%s", exc, body.get("type"))
        raise
    except Exception as exc:
        _log.debug("HL API error: %s  body=%s", exc, body.get("type"))
        raise


# ── Price ─────────────────────────────────────────────────────────────────────

def get_mid_price(coin: str) -> Optional[float]:
    """
    Return current mid price for coin ("BTC" or "ETH").
    Uses allMids endpoint — single call returns all coins.
    """
    try:
        data = _hl_post({"type": "allMids"})
        if isinstance(data, dict):
            raw = data.get(coin)
            if raw is not None:
                return float(raw)
    except Exception as exc:
        _log.debug("get_mid_price(%s) failed: %s", coin, exc)
    return None


# ── Contract metadata ─────────────────────────────────────────────────────────

def get_sz_decimals() -> dict[str, int]:
    """
    Return {coin: szDecimals} for all Hyperliquid perp coins.
    Cached for 1 hour (metadata never changes between restarts).
    BTC = 5 (min 0.00001 BTC), ETH = 4 (min 0.0001 ETH).
    """
    global _SZ_DEC_CACHE, _SZ_DEC_TS
    now = time.monotonic()
    if _SZ_DEC_CACHE is not None and (now - _SZ_DEC_TS) < _SZ_DEC_TTL:
        return _SZ_DEC_CACHE
    try:
        data = _hl_post({"type": "meta"})
        universe = data.get("universe", [])
        result = {asset["name"]: int(asset["szDecimals"]) for asset in universe}
        _SZ_DEC_CACHE = result
        _SZ_DEC_TS    = now
        return result
    except Exception as exc:
        _log.warning("get_sz_decimals failed: %s", exc)
        # Hardcoded fallback — safe defaults
        return {"BTC": 5, "ETH": 4}


# ── Funding rate ──────────────────────────────────────────────────────────────

def get_funding(coin: str) -> Optional[HLFundingSnapshot]:
    """
    Return current funding rate and OI for coin.
    Cached for _FUNDING_TTL seconds.

    Funding rate is per-8h (e.g. 0.0001 = 0.01% per 8h).
    Positive funding → longs pay shorts (crowded long side).
    Negative funding → shorts pay longs (crowded short side).
    """
    now = time.monotonic()
    cached = _FUNDING_CACHE.get(coin)
    if cached and (now - cached[1]) < _FUNDING_TTL:
        return cached[0]

    try:
        data = _hl_post({"type": "metaAndAssetCtxs"})
        # data is [meta_dict, [assetCtx, ...]] where assetCtx order matches meta universe
        if not isinstance(data, list) or len(data) < 2:
            return None
        meta    = data[0]
        ctxs    = data[1]
        coins   = [a["name"] for a in meta.get("universe", [])]
        if coin not in coins:
            return None
        idx = coins.index(coin)
        ctx = ctxs[idx]
        snap = HLFundingSnapshot(
            coin=coin,
            funding_rate=float(ctx.get("funding",       0.0)),
            open_interest=float(ctx.get("openInterest", 0.0)),
            mid_price=float(ctx.get("midPx",            0.0)),
        )
        _FUNDING_CACHE[coin] = (snap, now)
        return snap
    except Exception as exc:
        _log.debug("get_funding(%s) failed: %s", coin, exc)
        return None


def get_funding_score(coin: str, fade_threshold: float = 0.0005) -> Optional[float]:
    """
    Contrarian funding score in [-1, +1].

    High positive funding → longs are crowded → fade longs → return negative score.
    High negative funding → shorts are crowded → fade shorts → return positive score.

    Returns None if |funding_rate| < fade_threshold / 2 (noise — no signal).

    Formula: score = -clamp(funding_rate / fade_threshold, -1, 1)
    The negation converts "high positive funding" (bearish contrarian) → negative score.
    """
    snap = get_funding(coin)
    if snap is None:
        return None
    fr = snap.funding_rate
    if abs(fr) < fade_threshold / 2:
        return None   # funding too small to carry information
    raw   = fr / fade_threshold
    raw   = max(-1.0, min(1.0, raw))
    return -raw       # negate: high long crowding → negative score (short signal)


# ── Liquidation events ────────────────────────────────────────────────────────

def get_recent_liquidations(coin: str, lookback_seconds: int = 60) -> HLLiqSnapshot:
    """
    Fetch recent trades for coin, filter for liquidation events in the last
    lookback_seconds, sum long and short liquidation notional.

    Liquidation trades appear in recentTrades with a "liquidation" key = True.
    The "px" and "sz" fields give fill price and size (coin units).

    Returns HLLiqSnapshot. long_liq_usd and short_liq_usd are 0 if no data.

    Cache: TTL = _LIQ_TTL (15s) — time-sensitive for signal freshness.
    """
    now_mono = time.monotonic()
    cached   = _LIQ_CACHE.get(coin)
    if cached and (now_mono - cached[1]) < _LIQ_TTL:
        return cached[0]

    snap = HLLiqSnapshot(coin=coin, long_liq_usd=0.0, short_liq_usd=0.0,
                         window_seconds=lookback_seconds)
    try:
        data = _hl_post({"type": "recentTrades", "coin": coin})
        if not isinstance(data, list):
            return snap

        cutoff_ms = int(time.time() * 1000) - lookback_seconds * 1000
        long_liq  = 0.0
        short_liq = 0.0

        for trade in data:
            # Liquidation flag — HL marks these with "liquidation": true
            if not trade.get("liquidation", False):
                continue
            ts_ms = int(trade.get("time", 0))
            if ts_ms < cutoff_ms:
                continue
            px    = float(trade.get("px",   0))
            sz    = float(trade.get("sz",   0))
            side  = str(trade.get("side",  "")).upper()
            notional = px * sz
            # "B" = buyer (liquidated short gets bought out) → short liquidation
            # "A" = seller/ask (liquidated long gets sold)   → long liquidation
            if side == "A":
                long_liq  += notional
            else:
                short_liq += notional

        snap = HLLiqSnapshot(coin=coin, long_liq_usd=long_liq,
                             short_liq_usd=short_liq,
                             window_seconds=lookback_seconds)
    except Exception as exc:
        _log.debug("get_recent_liquidations(%s) failed: %s", coin, exc)

    _LIQ_CACHE[coin] = (snap, now_mono)
    return snap


def get_liq_imbalance_score(
    coin: str,
    lookback_seconds: int = 60,
    min_notional: float = 10_000.0,
) -> Optional[float]:
    """
    Liquidation imbalance score in [-1, +1].

    Positive = more shorts being liquidated → bullish (short squeeze signal).
    Negative = more longs being liquidated → bearish (long liquidation cascade).

    Returns None when total notional < min_notional (not enough activity to signal).

    Formula: (short_liq - long_liq) / (short_liq + long_liq)
    """
    snap = get_recent_liquidations(coin, lookback_seconds)
    if snap.total_usd < min_notional:
        return None
    if snap.total_usd == 0:
        return None
    return (snap.short_liq_usd - snap.long_liq_usd) / snap.total_usd


# ── Historical candles (for backtest) ─────────────────────────────────────────

def get_candles(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[HLCandle]:
    """
    Fetch historical OHLCV candles from Hyperliquid.

    coin     : "BTC" or "ETH"
    interval : "1m", "5m", "15m", "1h", "4h", "1d"
    start_ms : window start in milliseconds (inclusive)
    end_ms   : window end in milliseconds (exclusive)

    HL returns up to 5000 candles per request. Paginates automatically
    if the requested range exceeds one batch.

    IMPORTANT: interval strings match Hyperliquid's exact format ("5m" not "5min").
    """
    candles: list[HLCandle] = []
    batch_start = start_ms
    interval_ms = _interval_to_ms(interval)

    while batch_start < end_ms:
        try:
            data = _hl_post({
                "type": "candleSnapshot",
                "req": {
                    "coin":      coin,
                    "interval":  interval,
                    "startTime": batch_start,
                    "endTime":   end_ms,
                },
            })
            if not data:
                break
            for row in data:
                # HL candle format: {"t": ms, "o": "84000", "h": "84100",
                #                    "l": "83900", "c": "84050", "v": "1.23"}
                candles.append(HLCandle(
                    open_time=int(row["t"]),
                    open=float(row["o"]),
                    high=float(row["h"]),
                    low=float(row["l"]),
                    close=float(row["c"]),
                    volume=float(row["v"]),
                ))
            if len(data) < 5000:
                break   # less than full batch → we have all data
            # Advance past last candle for next batch
            last_t     = int(data[-1]["t"])
            batch_start = last_t + interval_ms
        except Exception as exc:
            _log.warning("get_candles(%s,%s) failed at %d: %s", coin, interval, batch_start, exc)
            break

    return candles


def _interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds. E.g. '5m' → 300_000."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    try:
        num  = int(interval[:-1])
        unit = interval[-1]
        return num * units[unit]
    except Exception:
        return 300_000   # default 5m
