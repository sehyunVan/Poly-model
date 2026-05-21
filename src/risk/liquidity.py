"""
Slippage estimation and liquidity scoring.

estimate_slippage() walks the orderbook to compute the expected average
fill price for a given USDC order size, then expresses the deviation from
the best quote as a fraction.

check_liquidity() produces a single 0–1 score combining orderbook depth
and 24-hour volume.  The execution layer uses this to reject orders on
markets that are too thin.
"""

from __future__ import annotations

import math

from data.schemas import OrderBookSnapshot, OrderBookLevel


def estimate_slippage(
    orderbook: OrderBookSnapshot,
    desired_size: float,
    side: str = "BUY",
) -> float:
    """
    Estimate the average slippage of filling desired_size USDC on one side.

    For a BUY order the asks are consumed top-to-bottom (lowest ask first).
    For a SELL order the bids are consumed top-to-bottom (highest bid first).

    Slippage = |weighted_avg_fill_price - best_quote| / best_quote

    If the book is too thin to fill the full order, slippage is measured
    against the last available level (conservative estimate).

    Args:
        orderbook:    Current OrderBookSnapshot.
        desired_size: Desired fill size in USDC.
        side:         "BUY" or "SELL".

    Returns:
        Slippage fraction in [0, 1).  Returns 1.0 when the book is empty.
    """
    if desired_size <= 0:
        return 0.0

    levels: list[OrderBookLevel] = (
        orderbook.asks if side.upper() == "BUY" else orderbook.bids
    )
    if not levels:
        return 1.0

    best_quote = levels[0].price
    if best_quote <= 0:
        return 1.0

    filled_usdc    = 0.0
    weighted_price = 0.0

    for level in levels:
        price      = level.price
        # Each level's USDC capacity = price * size_in_shares.
        # In Polymarket, size is already quoted in USDC equivalent.
        level_usdc = level.size

        take = min(level_usdc, desired_size - filled_usdc)
        weighted_price += price * take
        filled_usdc    += take

        if filled_usdc >= desired_size:
            break

    if filled_usdc <= 0:
        return 1.0

    avg_fill  = weighted_price / filled_usdc
    slippage  = abs(avg_fill - best_quote) / best_quote
    return round(min(slippage, 1.0), 6)


def check_liquidity(
    orderbook: OrderBookSnapshot,
    volume_24h: float,
    min_depth_usdc: float = 500.0,
    min_volume_24h: float = 1000.0,
) -> float:
    """
    Compute a composite liquidity score in [0, 1].

    Score is the geometric mean of:
        depth_score  = min(total_10_level_depth / min_depth_usdc,  1.0)
        volume_score = min(volume_24h           / min_volume_24h,  1.0)

    A score of 1.0 means both thresholds are fully met or exceeded.
    A score of 0.0 means the book is completely empty or volume is zero.

    Args:
        orderbook:      Current OrderBookSnapshot.
        volume_24h:     24-hour traded volume in USDC.
        min_depth_usdc: Depth considered "sufficient" (10-level total bid+ask).
        min_volume_24h: Volume considered "sufficient".

    Returns:
        Liquidity score in [0.0, 1.0].
    """
    # Depth score: sum of top-10 bid and ask sizes
    depth = sum(l.size for l in orderbook.bids[:10]) + \
            sum(l.size for l in orderbook.asks[:10])
    depth_score  = min(depth    / max(min_depth_usdc, 1e-9), 1.0)
    volume_score = min(volume_24h / max(min_volume_24h, 1e-9), 1.0)

    # Geometric mean — penalises imbalance between depth and volume
    score = math.sqrt(depth_score * volume_score)
    return round(score, 4)
