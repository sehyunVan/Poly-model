"""
Order execution: translate approved TradeSignals into live CLOB orders.

execute_signal() is the primary entry point:
    1. Resolves the YES/NO token from the market registry.
    2. Fetches the current orderbook to determine the best quote.
    3. Estimates fill slippage; scales size down if slippage_cap is exceeded.
    4. Places a GTC limit order at the best quote.
    5. Polls for fill for up to 60 seconds.
    6. On non-fill: cancels the stale order and re-quotes (up to 3 attempts).
    7. Returns NO_FILL after 3 failed attempts.

close_position() liquidates an existing position:
    "limit"  — GTC limit order at the current best quote on the closing side.
    "market" — FOK order for immediate execution at any available price.

Note: This module calls the py_clob_client ClobClient directly, which is the
same underlying library used by polymarket-mcp-main/server.py's place-order
and cancel-order tools.  Using the client directly avoids an MCP stdio round-trip
and allows synchronous blocking I/O with full error handling.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from risk.liquidity import estimate_slippage
from risk.schemas import Position
from signal_layer.schemas import TradeSignal

from .schemas import OrderRequest, OrderResult

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_log = logging.getLogger("execution")
if not _log.handlers:
    from logging.handlers import RotatingFileHandler as _RFH
    _fh = _RFH(
        _LOG_DIR / "execution.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))
    _log.addHandler(_fh)
    _log.addHandler(logging.StreamHandler())
    _log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_SEARCH_PATHS = [
    _HERE.parents[3],
    _HERE.parents[3] / "polymarket-mcp-main" / "polymarket-mcp-main",
]
for _p in _SEARCH_PATHS:
    _env = _p / ".env"
    if _env.exists():
        load_dotenv(_env)
        break

CLOB_HOST = os.getenv("CLOB_HOST", "https://clob.polymarket.com")

# ---------------------------------------------------------------------------
# Execution parameters
# ---------------------------------------------------------------------------

_MAX_REQUOTES      = 3    # Maximum re-quote attempts before giving up
_FILL_TIMEOUT_SEC  = 60   # Seconds to wait for a fill before cancelling
_POLL_INTERVAL_SEC = 5    # Seconds between fill-status polls

# ---------------------------------------------------------------------------
# ClobClient helper
# ---------------------------------------------------------------------------


_clob_unavailable_logged = False   # suppress repeated warnings once noted


def _get_clob_client():
    """Return an authenticated ClobClient, or None when credentials are absent."""
    global _clob_unavailable_logged
    try:
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.constants import POLYGON

        key    = os.getenv("KEY")
        funder = os.getenv("FUNDER")

        if not key or not funder:
            if not _clob_unavailable_logged:
                _log.warning("KEY/FUNDER not set - order execution unavailable")
                _clob_unavailable_logged = True
            return None

        client = ClobClient(
            CLOB_HOST,
            key=key,
            chain_id=POLYGON,
            funder=funder,
            signature_type=0,
        )
        client.set_api_creds(client.create_or_derive_api_key())
        return client
    except ImportError:
        if not _clob_unavailable_logged:
            _log.warning("py_clob_client not installed - order execution unavailable")
            _clob_unavailable_logged = True
        return None


# ---------------------------------------------------------------------------
# Low-level order primitives
# ---------------------------------------------------------------------------


def _place_order(
    client,
    token_id: str,
    price: float,
    size: float,
    side: str,
    order_type_str: str = "GTC",
) -> Optional[str]:
    """
    Submit a single order via ClobClient.

    Returns:
        Exchange-assigned order_id string on success, None on failure.
    """
    try:
        from py_clob_client_v2.clob_types import OrderArgs, OrderType

        order_type = OrderType.GTC if order_type_str.upper() == "GTC" else OrderType.FOK
        order_args = OrderArgs(
            token_id=token_id,
            price=round(max(0.01, min(0.99, price)), 4),
            size=round(size, 2),
            side=side,
        )
        signed = client.create_order(order_args)
        result = client.post_order(signed, order_type)

        if isinstance(result, dict):
            order_id = result.get("orderID") or result.get("order_id")
            if order_id:
                return str(order_id)

        _log.error("_place_order: unexpected result format: %s", result)
        return None
    except Exception as exc:
        _log.error("_place_order failed: %s", exc)
        return None


def _cancel_order(client, order_id: str) -> bool:
    """Cancel an order by ID.  Returns True if the request was accepted."""
    try:
        result = client.cancel_order(order_id)
        _log.info("_cancel_order %s → %s", order_id, result)
        return True
    except Exception as exc:
        _log.error("_cancel_order %s failed: %s", order_id, exc)
        return False


def _get_fill_status(client, order_id: str) -> dict:
    """
    Query the CLOB API for the current fill status of an order.

    Returns:
        {
          "status":       "MATCHED" | "LIVE" | "CANCELLED" | "UNKNOWN",
          "filled_size":  float,
          "avg_price":    float,
        }
    """
    try:
        order = client.get_order(order_id)
        if not order:
            return {"status": "UNKNOWN", "filled_size": 0.0, "avg_price": 0.0}

        raw = order if isinstance(order, dict) else (
            order.__dict__ if hasattr(order, "__dict__") else {}
        )
        status      = str(raw.get("status", "UNKNOWN")).upper()
        filled_size = float(raw.get("size_matched", raw.get("filled", 0)) or 0)
        avg_price   = float(raw.get("average_price", raw.get("avg_price", 0)) or 0)

        return {"status": status, "filled_size": filled_size, "avg_price": avg_price}
    except Exception as exc:
        _log.warning("_get_fill_status %s: %s", order_id, exc)
        return {"status": "UNKNOWN", "filled_size": 0.0, "avg_price": 0.0}


def _wait_for_fill(client, order_id: str, timeout_sec: int = _FILL_TIMEOUT_SEC) -> dict:
    """
    Block until the order is filled or the timeout expires.

    Returns:
        Final fill-status dict from _get_fill_status().
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        info   = _get_fill_status(client, order_id)
        status = info.get("status", "UNKNOWN")
        if status in ("MATCHED", "CANCELLED"):
            return info
        time.sleep(_POLL_INTERVAL_SEC)
    return _get_fill_status(client, order_id)   # final check after timeout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_signal(
    signal: TradeSignal,
    final_size: float,
    slippage_cap: float = 0.01,
) -> OrderResult:
    """
    Convert an approved TradeSignal into a live CLOB order.

    Execution procedure:
        1. Resolve YES/NO token_id from the market registry.
        2. Fetch the current orderbook for the best quote.
        3. Estimate slippage; if it exceeds slippage_cap, scale size down.
        4. Place a GTC limit order at the best quote.
        5. Poll for fill for up to 60 seconds.
        6. On non-fill: cancel the stale order and re-quote (up to 3 attempts).
        7. Return NO_FILL after all attempts are exhausted.

    Args:
        signal:       Approved TradeSignal from signal_layer.
        final_size:   Risk-approved USDC order size (>= 0).
        slippage_cap: Maximum acceptable slippage fraction (default 1 %).

    Returns:
        OrderResult describing the final execution outcome.
    """
    if final_size <= 0:
        return OrderResult(
            status="NO_FILL",
            market_id=signal.market_id,
            reason="final_size must be > 0",
        )

    if signal.direction == "NO_TRADE":
        return OrderResult(
            status="NO_FILL",
            market_id=signal.market_id,
            reason="signal direction is NO_TRADE",
        )

    # Resolve YES/NO token_id and CLOB side
    from data.market import get_market, get_orderbook

    market = get_market(signal.market_id)
    if market is None:
        _log.error("execute_signal: market %s not found", signal.market_id)
        return OrderResult(
            status="ERROR",
            market_id=signal.market_id,
            reason=f"market {signal.market_id} not found",
        )

    if signal.direction == "BUY_YES":
        token_id  = market.yes_token_id
        clob_side = "BUY"
    else:  # BUY_NO
        token_id  = market.no_token_id
        clob_side = "BUY"

    if not token_id:
        return OrderResult(
            status="ERROR",
            market_id=signal.market_id,
            reason=f"token_id for direction {signal.direction} is empty",
        )

    client = _get_clob_client()
    if client is None:
        return OrderResult(
            status="NO_FILL",
            market_id=signal.market_id,
            token_id=token_id,
            side=clob_side,
            requested_size=final_size,
            reason="CLOB client unavailable (missing KEY/FUNDER credentials)",
        )

    # Balance preflight: refuse to submit if available USDC is insufficient
    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams
        balance_data = client.get_balance_allowance(BalanceAllowanceParams(asset_type=None))
        if isinstance(balance_data, dict):
            usdc_balance = float(balance_data.get("balance", balance_data.get("USDC", 0)) or 0)
            if usdc_balance < final_size:
                _log.error(
                    "execute_signal: insufficient balance %.2f for order size %.2f",
                    usdc_balance, final_size,
                )
                return OrderResult(
                    status="NO_FILL",
                    market_id=signal.market_id,
                    token_id=token_id,
                    side=clob_side,
                    requested_size=final_size,
                    reason=f"insufficient balance: ${usdc_balance:.2f} < ${final_size:.2f}",
                )
    except Exception as exc:
        _log.warning("execute_signal: balance check failed (%s) — proceeding", exc)

    size    = final_size
    attempt = 0

    while attempt < _MAX_REQUOTES:
        attempt += 1

        # Refresh orderbook on every attempt to get the current best quote
        ob = get_orderbook(signal.market_id)
        if clob_side == "BUY":
            if not ob.asks:
                _log.warning(
                    "execute_signal attempt %d: no asks for %s",
                    attempt, signal.market_id,
                )
                break
            price = ob.asks[0].price
        else:
            if not ob.bids:
                _log.warning(
                    "execute_signal attempt %d: no bids for %s",
                    attempt, signal.market_id,
                )
                break
            price = ob.bids[0].price

        # Estimate impact; reduce size if slippage exceeds cap
        slip = estimate_slippage(ob, size, clob_side)
        if slip > slippage_cap:
            old_size = size
            # Scale proportionally so expected slippage ≈ slippage_cap
            size = max(size * (slippage_cap / max(slip, 1e-9)), 1.0)
            _log.info(
                "execute_signal: slippage %.4f > cap %.4f; size %.2f → %.2f",
                slip, slippage_cap, old_size, size,
            )

        _log.info(
            "execute_signal attempt %d/%d: %s %s @ %.4f × $%.2f  slip=%.4f",
            attempt, _MAX_REQUOTES,
            signal.market_id, clob_side, price, size, slip,
        )

        order_id = _place_order(client, token_id, price, size, clob_side)
        if order_id is None:
            _log.error(
                "execute_signal: place_order failed on attempt %d/%d",
                attempt, _MAX_REQUOTES,
            )
            continue  # Retry with a fresh price on the next iteration

        fill_info = _wait_for_fill(client, order_id, _FILL_TIMEOUT_SEC)
        status    = fill_info.get("status", "UNKNOWN")

        if status == "MATCHED":
            filled = fill_info.get("filled_size") or size
            avg    = fill_info.get("avg_price") or price
            _log.info(
                "execute_signal: FILLED order %s  filled=%.2f  avg_price=%.4f",
                order_id, filled, avg,
            )
            return OrderResult(
                order_id=order_id,
                status="FILLED",
                filled_size=filled,
                avg_fill_price=avg,
                market_id=signal.market_id,
                token_id=token_id,
                side=clob_side,
                requested_size=final_size,
                attempts=attempt,
                reason="order filled",
            )

        # Not filled — cancel the stale order before re-quoting
        _cancel_order(client, order_id)
        _log.info(
            "execute_signal: attempt %d unfilled (status=%s), order %s cancelled",
            attempt, status, order_id,
        )

    # All attempts exhausted without a fill
    _log.warning(
        "execute_signal: NO_FILL after %d attempt(s) for %s",
        attempt, signal.market_id,
    )
    return OrderResult(
        status="NO_FILL",
        market_id=signal.market_id,
        token_id=token_id,
        side=clob_side,
        requested_size=final_size,
        attempts=attempt,
        reason=f"order unfilled after {attempt} attempt(s)",
    )


def close_position(
    position: Position,
    mode: str = "limit",
) -> OrderResult:
    """
    Liquidate an existing position.

    Called by the risk module in response to 'close_position' or
    'reduce_position' actions from evaluate_portfolio_risk().

    Args:
        position: Current Position to close (size is in USDC).
        mode:     "limit"  → GTC limit order at the current best bid.
                  "market" → FOK order for immediate execution.

    Returns:
        OrderResult describing the liquidation outcome.
    """
    from data.market import get_orderbook

    # Closing a BUY position means we SELL the token back
    token_id  = position.token_id
    clob_side = "SELL"
    size      = position.size

    client = _get_clob_client()
    if client is None:
        return OrderResult(
            status="NO_FILL",
            market_id=position.market_id,
            token_id=token_id,
            side=clob_side,
            requested_size=size,
            reason="CLOB client unavailable",
        )

    ob = get_orderbook(position.market_id)

    # Slippage guard: warn loudly if closing into a thin market.
    # For "market" (FOK) mode we proceed anyway to guarantee the position is closed,
    # but we log a warning so the operator can see the cost.
    slip = estimate_slippage(ob, size, "SELL")
    if slip > 0.05:
        _log.warning(
            "close_position: HIGH SLIPPAGE %.2f%% for %s size=%.2f — "
            "market is thin; proceeding because position must be closed",
            slip * 100, position.market_id, size,
        )

    if mode == "market":
        # FOK: execute immediately at any available price
        price      = ob.bids[0].price if ob.bids else position.current_price
        order_type = "FOK"
    else:
        # GTC: limit order at best bid
        price      = ob.bids[0].price if ob.bids else position.current_price
        order_type = "GTC"

    _log.info(
        "close_position: %s token=%s mode=%s price=%.4f size=%.2f slip=%.4f",
        position.market_id, token_id, mode, price, size, slip,
    )

    order_id = _place_order(client, token_id, price, size, clob_side, order_type)
    if order_id is None:
        return OrderResult(
            status="ERROR",
            market_id=position.market_id,
            token_id=token_id,
            side=clob_side,
            requested_size=size,
            reason="place_order failed during close_position",
        )

    if mode == "market":
        # FOK result is immediate — just query once
        fill_info = _get_fill_status(client, order_id)
        filled    = "FILLED" if fill_info.get("status") == "MATCHED" else "NO_FILL"
        return OrderResult(
            order_id=order_id,
            status=filled,
            filled_size=fill_info.get("filled_size", 0.0),
            avg_fill_price=fill_info.get("avg_price", price),
            market_id=position.market_id,
            token_id=token_id,
            side=clob_side,
            requested_size=size,
            attempts=1,
            reason=f"close_position FOK: {filled}",
        )

    # GTC: wait up to the standard timeout
    fill_info = _wait_for_fill(client, order_id, _FILL_TIMEOUT_SEC)
    status    = "FILLED" if fill_info.get("status") == "MATCHED" else "PARTIALLY_FILLED"
    _log.info(
        "close_position: order %s  filled=%.2f  status=%s",
        order_id, fill_info.get("filled_size", 0.0), status,
    )
    return OrderResult(
        order_id=order_id,
        status=status,
        filled_size=fill_info.get("filled_size", 0.0),
        avg_fill_price=fill_info.get("avg_price", price),
        market_id=position.market_id,
        token_id=token_id,
        side=clob_side,
        requested_size=size,
        attempts=1,
        reason=f"close_position GTC: {status}",
    )
