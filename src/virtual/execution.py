"""
Virtual order execution — simulates fills from the live orderbook without
placing real orders on the Polymarket CLOB.

simulate_fill() mirrors the real execute_signal() interface so that
main.py can call it as a drop-in replacement in virtual mode.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.schemas import OrderBookSnapshot          # type: ignore
from execution.schemas import OrderResult           # type: ignore
from risk.liquidity import estimate_slippage        # type: ignore
from signal_layer.schemas import TradeSignal        # type: ignore
from virtual.portfolio import VirtualPortfolio, VirtualPosition  # type: ignore


def simulate_fill(
    signal: TradeSignal,
    final_size: float,
    orderbook: OrderBookSnapshot,
    slippage_cap: float = 0.01,
) -> OrderResult:
    """
    Simulate an order fill using the current live orderbook.

    Fill price is the best ask (for YES buys) or (1 - best bid) for NO buys,
    adjusted for the estimated slippage of the desired size.

    If estimated slippage exceeds slippage_cap, the size is scaled down
    proportionally (same behaviour as real execute_signal()).

    Returns an OrderResult with status="FILLED" on success, or "NO_FILL"
    if the orderbook is empty or size is zero.
    """
    _base = dict(
        market_id=signal.market_id,
        token_id="",
        side="BUY",
        requested_size=final_size,
        attempts=1,
        timestamp=datetime.now(timezone.utc),
    )

    if final_size <= 0:
        return OrderResult(
            status="NO_FILL",
            reason="final_size is zero or negative",
            **_base,
        )

    # ── Determine reference price and slippage ────────────────────────────────
    direction = signal.direction   # "BUY_YES" | "BUY_NO"
    is_yes    = direction == "BUY_YES"

    if is_yes:
        if not orderbook.asks:
            return OrderResult(status="NO_FILL", reason="empty ask side", **_base)
        best_quote = orderbook.asks[0].price
        slippage   = estimate_slippage(orderbook, final_size, side="BUY")
    else:
        # For NO tokens the effective price is 1 - YES_bid
        if not orderbook.bids:
            return OrderResult(status="NO_FILL", reason="empty bid side", **_base)
        best_quote = 1.0 - orderbook.bids[0].price
        # Slippage is symmetric; we estimate against the bid side
        slippage   = estimate_slippage(orderbook, final_size, side="SELL")

    # ── Scale down if slippage exceeds cap ────────────────────────────────────
    adjusted_size = final_size
    if slippage > slippage_cap and slippage > 0:
        scale         = slippage_cap / slippage
        adjusted_size = max(final_size * scale, 0.0)

    if adjusted_size <= 0:
        return OrderResult(
            status="NO_FILL",
            reason=f"size reduced to zero after slippage scaling (slippage={slippage:.4f})",
            **_base,
        )

    # ── Compute simulated fill price ──────────────────────────────────────────
    # Approximate fill price = best_quote + (slippage * best_quote)
    # Clamped to [0.001, 0.999] to stay within prediction-market bounds.
    fill_price = best_quote * (1.0 + slippage)
    fill_price = max(0.001, min(0.999, fill_price))

    return OrderResult(
        status="FILLED",
        filled_size=round(adjusted_size, 6),
        avg_fill_price=round(fill_price, 6),
        reason=f"virtual fill | direction={direction} | slippage={slippage:.4f}",
        **_base,
    )


def simulate_position_exit(
    vp: VirtualPortfolio,
    pos: VirtualPosition,
    current_yes_price: float,
) -> dict:
    """
    Force-close a virtual position at the current market price.

    Used for stop-loss exits.  The position is moved to closed_positions with
    outcome=None (distinguishing it from natural settlement).

    Returns:
        {"exit_value": float, "realized_pnl": float}
    """
    p_fill = max(0.001, min(0.999, pos.fill_price))
    p_curr = max(0.001, min(0.999, current_yes_price))

    if pos.direction == "YES":
        # Held YES tokens: count = size/p_fill; sell at p_curr each
        exit_value = pos.size_usdc * (p_curr / p_fill)
    else:
        # Held NO tokens: count = size/(1-p_fill); sell at (1-p_curr) each
        exit_value = pos.size_usdc * ((1.0 - p_curr) / (1.0 - p_fill))

    exit_value   = round(max(0.0, exit_value), 6)
    realized_pnl = round(exit_value - pos.size_usdc, 6)

    pos.realized_pnl = realized_pnl
    pos.outcome      = None   # force-exit, not natural settlement

    vp.available_usdc  = round(vp.available_usdc + exit_value, 6)
    vp.positions       = [p for p in vp.positions if p is not pos]
    vp.closed_positions.append(pos)
    vp.mark_updated()

    return {"exit_value": exit_value, "realized_pnl": realized_pnl}
