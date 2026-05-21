"""
Market Conviction Paper Trading — Early Skew Detection
=====================================================

Strategy: Enter when market shows early directional skew (e.g., 55/45 or 45/55),
BEFORE crowd pushes it to extremes (95/5).

Better entry prices = higher payouts = lower breakeven WR needed.
Example: at 55% entry need 54% WR to break even, vs 95% WR at late entry.

Config: min_skew / max_skew determines entry window (default 0.45-0.55 = neutral)
"""

import logging
import requests
import json
from typing import Optional

log = logging.getLogger("crypto.market_conviction")

GAMMA_BASE = "https://gamma-api.polymarket.com"
_HTTP = requests.Session()
_HTTP.headers["User-Agent"] = "poly-conviction/1.0"


def fetch_market(slug: str) -> Optional[dict]:
    """Fetch a market by slug from Gamma API."""
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=8)
        r.raise_for_status()
        events = r.json()
        if not events:
            return None
        ev = events[0]
        markets = ev.get("markets", [])
        if not markets:
            return None
        m = markets[0]
        m["_symbol"] = slug.split("-")[0].upper()
        m["_timeframe"] = slug.split("-")[2] if len(slug.split("-")) > 2 else "5m"
        return m
    except Exception as e:
        log.debug(f"fetch_market({slug}) failed: {e}")


def fetch_market_by_id(market_id: str, symbol: str = "UNKNOWN", timeframe: str = "5m") -> Optional[dict]:
    """Fetch a market directly by its condition/market ID — used to settle stale open positions."""
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/markets/{market_id}", timeout=8)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        m = r.json()
        if not m:
            return None
        m["_symbol"] = symbol
        m["_timeframe"] = timeframe
        return m
    except Exception as e:
        log.debug(f"fetch_market_by_id({market_id}) failed: {e}")
        return None
        return None


def compute_conviction_signal(market: dict) -> Optional[dict]:
    """
    Read market price -> compute conviction.

    If market prices YES at 0.30 and NO at 0.70:
      - Market believes: 30% UP, 70% DOWN
      - Conviction = max(0.30, 0.70) = 0.70 (70% certain)
      - Direction = DOWN

    Only trade if conviction > 60%.
    Size bet by conviction: 60% -> $2, 95% -> $10.
    """
    try:
        prices_raw = market.get("outcomePrices", "[0.5, 0.5]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        up_price = float(prices[0])  # YES token price
        down_price = float(prices[1])  # NO token price

        # Skip already-resolved or near-resolved markets (one outcome at 1.0, other at 0.0)
        # or markets priced extremely far (>.98 or <.02) indicating resolution soon
        if (up_price >= 0.98 or down_price >= 0.98 or up_price <= 0.02 or down_price <= 0.02):
            return None

        conviction = max(up_price, down_price)

        # Skip if market is too uncertain
        if conviction < 0.60:
            return None

        direction = "UP" if up_price > down_price else "DOWN"
        entry_price = up_price if direction == "UP" else down_price

        # Size: 60% -> $2, 95% -> $10
        if conviction <= 0.60:
            bet_size = 2.0
        elif conviction >= 0.95:
            bet_size = 10.0
        else:
            bet_size = 2.0 + (conviction - 0.60) / (0.95 - 0.60) * 8.0

        return {
            "direction": direction,
            "conviction": conviction,
            "bet_size": min(bet_size, 10.0),
            "entry_price": entry_price,
            "symbol": market.get("_symbol", "UNKNOWN"),
            "timeframe": market.get("_timeframe", "5m"),
            "market_id": market.get("id", ""),
            "slug": market.get("slug", ""),
        }
    except Exception as e:
        log.debug(f"compute_conviction_signal failed: {e}")
        return None


def compute_market_skew(market: dict) -> Optional[dict]:
    """
    Detect market skew (market certainty on one direction).

    Returns dict with:
      - up_skew: YES price as % (0.0 to 1.0)
      - direction: "UP" if up_skew > 0.5, "DOWN" if < 0.5
      - should_enter: True if skew is in [min_skew, max_skew] range

    Example: YES @ 0.55, NO @ 0.45 → up_skew=0.55, direction="UP", should_enter=(if 0.45-0.55 configured)
    """
    try:
        prices_raw = market.get("outcomePrices", "[0.5, 0.5]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        yes_price = float(prices[0])
        no_price = float(prices[1])

        # Skip near-resolved
        if yes_price >= 0.98 or yes_price <= 0.02 or no_price >= 0.98 or no_price <= 0.02:
            return None

        # Skew: YES price as percentage
        total = yes_price + no_price
        if total <= 0:
            return None

        up_skew = yes_price / total
        direction = "UP" if up_skew > 0.5 else "DOWN"

        return {
            "up_skew": up_skew,
            "down_skew": 1.0 - up_skew,
            "direction": direction,
            "yes_price": yes_price,
            "no_price": no_price,
            "total_cost": total,
            "symbol": market.get("_symbol", "UNKNOWN"),
            "timeframe": market.get("_timeframe", "5m"),
            "market_id": market.get("id", ""),
            "slug": market.get("slug", ""),
        }
    except Exception as e:
        log.debug(f"compute_market_skew failed: {e}")
        return None


def check_arb_opportunity(market: dict, min_margin: float = 0.005) -> dict:
    """
    Detect when YES + NO prices sum to < $1.00 (mispricing).
    Return arbitrage signal if margin >= min_margin after accounting for fees.

    Example: YES @ 0.42, NO @ 0.55 = $0.97 total → buy both → guaranteed $1.00 payout → profit $0.03
    """
    try:
        prices_raw = market.get("outcomePrices", "[0.5, 0.5]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        yes_price = float(prices[0])
        no_price = float(prices[1])

        # Skip already-resolved or near-resolved markets
        if yes_price >= 0.98 or no_price >= 0.98:
            return None

        total_cost = yes_price + no_price
        gross_margin = 1.0 - total_cost

        if gross_margin < min_margin:
            return None

        # Polymarket 2% fee on the payout (~$0.02 per $1 won)
        fee_cost = 0.02
        net_margin = gross_margin - fee_cost

        # Only trigger if net is positive (fee doesn't erase profit)
        # Note: at 0.5% gross (0.995 total_cost), net = -1.5% (fee-negative in live)
        # But we keep it for paper testing to see how often arb appears

        return {
            "type": "ARB",
            "yes_price": yes_price,
            "no_price": no_price,
            "total_cost": total_cost,
            "gross_margin": gross_margin,
            "net_margin": net_margin,
            "margin_pct": (net_margin / total_cost) * 100 if total_cost > 0 else 0,
            "symbol": market.get("_symbol", "UNKNOWN"),
            "timeframe": market.get("_timeframe", "5m"),
        }
    except Exception as e:
        log.debug(f"check_arb_opportunity failed: {e}")
        return None
