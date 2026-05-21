"""
Pre-trade orderbook and liquidity filters.

Every filter must pass before generate_signals() is called.
Failing filters produce a NO_TRADE signal immediately without
wasting a prediction cycle on an illiquid market.
"""

from __future__ import annotations

from data.schemas import OrderBookSnapshot


def passes_pre_trade_filters(
    orderbook: OrderBookSnapshot,
    volume_24h: float,
    min_volume_24h: float = 1000.0,
    max_volume_24h: float = 50000.0,
    max_spread: float = 0.03,
    min_book_depth_usdc: float = 100.0,
    min_market_price: float = 0.08,
    max_market_price: float = 0.92,
) -> tuple[bool, str]:
    """
    Run all pre-trade checks against the current orderbook snapshot.

    Checks (in order):
        1. Best bid and ask levels must exist.
        2. Bid-ask spread <= max_spread.
        3. Combined depth on the best bid + ask level >= min_book_depth_usdc.
        4. 24-hour traded volume in [min_volume_24h, max_volume_24h].
           Upper bound skips efficient markets dominated by professional traders
           where the LLM/model edge is too small to overcome market pricing.
        5. Mid-price within [min_market_price, max_market_price].

    Args:
        orderbook:           Current OrderBookSnapshot for the market.
        volume_24h:          24-hour traded volume in USDC.
        min_volume_24h:      Minimum acceptable 24h volume.
        max_spread:          Maximum acceptable bid-ask spread.
        min_book_depth_usdc: Minimum combined liquidity on best bid + ask.
        min_market_price:    Skip markets where YES mid-price is below this.
        max_market_price:    Skip markets where YES mid-price is above this.

    Returns:
        (passed: bool, reason: str)
        reason is an empty string when passed=True.
    """
    # 1. Book must have at least one level on each side
    if not orderbook.bids or not orderbook.asks:
        return False, "empty orderbook: no bids or asks"

    # 2. Spread check
    spread = orderbook.spread
    if spread is None:
        return False, "cannot compute spread: missing best bid or ask"
    if spread > max_spread:
        return False, (
            f"spread {spread:.4f} exceeds max {max_spread:.4f}"
        )

    # 3. Best-level depth check
    best_bid_size = orderbook.bids[0].size
    best_ask_size = orderbook.asks[0].size
    best_depth = best_bid_size + best_ask_size
    if best_depth < min_book_depth_usdc:
        return False, (
            f"best-level depth {best_depth:.2f} USDC below minimum {min_book_depth_usdc:.2f}"
        )

    # 4. Volume check (min + max)
    if volume_24h < min_volume_24h:
        return False, (
            f"24h volume {volume_24h:.2f} USDC below minimum {min_volume_24h:.2f}"
        )
    if max_volume_24h > 0 and volume_24h > max_volume_24h:
        return False, (
            f"24h volume {volume_24h:.2f} USDC above maximum {max_volume_24h:.2f} "
            f"(market too efficient — professional traders dominate)"
        )

    # 5. Market price bounds — block extreme long-shots and near-certainties.
    #    The contaminated bootstrap model generates spuriously large alpha on these.
    mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2.0
    if mid_price < min_market_price:
        return False, (
            f"market price {mid_price:.3f} below minimum {min_market_price:.3f} (long-shot filter)"
        )
    if mid_price > max_market_price:
        return False, (
            f"market price {mid_price:.3f} above maximum {max_market_price:.3f} (near-certainty filter)"
        )

    return True, ""
