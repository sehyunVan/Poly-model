"""
crypto/execution.py — real CLOB order placement for the crypto loop.

place_crypto_order() is the live counterpart to virtual simulate_fill().
Called by loop.py when VIRTUAL_MODE=false.

Fixes applied vs initial version:
  H1 — all py_clob_client calls wrapped in _clob_timeout() (ThreadPoolExecutor)
       to prevent API hangs from blocking the loop indefinitely.
  H2 — balance preflight check before placing any order; returns NO_FILL with
       a clear "insufficient balance" reason instead of a cryptic API error.
  M1 — accepts price_hint (pre-fetched ask from loop.py's _get_clob_spread)
       on the first attempt to avoid a redundant duplicate CLOB book fetch.
  MM — maker mode: tries a limit bid just above best bid before falling back
       to taker. Saves 2% taker fee + improves average fill by ~1-2 cents.
       fill_type="MAKER" or "TAKER" in return dict for performance tracking.
"""
from __future__ import annotations

import concurrent.futures  # for TimeoutError catch clauses only; dispatch via call_with_timeout()
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

_SRC_INFRA = Path(__file__).resolve().parent.parent / "infra"
if str(_SRC_INFRA.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_INFRA.parent))

from infra.types import ClobTokenId, ConditionId        # noqa: E402
from infra.http_client import call_with_timeout, CLOB_TIMEOUT  # noqa: E402

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from execution.order import (       # type: ignore
    _get_clob_client,
    _place_order,
    _cancel_order,
    _get_fill_status,
)

_HTTP = httpx.Client(timeout=6.0)

_FILL_TIMEOUT_SEC       = 15   # taker fill poll timeout
_POLL_INTERVAL_SEC      = 3
_MAX_REQUOTES           = 2
_MAKER_FILL_TIMEOUT_SEC = 45   # wait up to 45s for maker fill before taker fallback
_MAKER_BID_OFFSET       = 0.010 # place bid this many cents above best bid (aggressive maker)

# H1: _clob_timeout is now an alias for the canonical call_with_timeout().
# New call sites should import from infra.http_client directly.
_clob_timeout = call_with_timeout


# ── Balance query (used by loop.py for C1 startup sync) ───────────────────────

def get_clob_balance(log: logging.Logger) -> Optional[float]:  # noqa: D103
    """
    Query actual USDC balance from the Polymarket CLOB wallet.

    Called by loop.py at startup in real mode to sync vp.available_usdc
    with the real funded amount (C1).  Returns None if unavailable.
    """
    client = _get_clob_client()
    if client is None:
        return None
    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        data = _clob_timeout(
            client.get_balance_allowance,
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )
        if isinstance(data, dict):
            raw     = float(data.get("balance", data.get("USDC", 0)) or 0)
            # USDC on Polygon uses 6 decimal places; API returns micro-USDC
            balance = raw / 1_000_000 if raw > 1_000 else raw
            log.info("CLOB wallet USDC balance: $%.2f  (raw=%s)", balance, raw)
            return balance
    except concurrent.futures.TimeoutError:
        log.warning("get_clob_balance: timed out")
    except Exception as exc:
        log.warning("get_clob_balance failed: %s", exc)
    return None


# ── CLOB book fetch ───────────────────────────────────────────────────────────

def _fetch_book(token_id: ClobTokenId) -> tuple[Optional[float], Optional[float]]:
    """Return (best_ask, best_bid) for the token, or (None, None) on failure."""
    try:
        r = _HTTP.get(
            "https://clob.polymarket.com/book",
            params={"token_id": token_id},
        )
        r.raise_for_status()
        book = r.json()
        asks = book.get("asks", [])
        bids = book.get("bids", [])
        best_ask = min(float(a["price"]) for a in asks) if asks else None
        best_bid = max(float(b["price"]) for b in bids) if bids else None
        return best_ask, best_bid
    except Exception:
        pass
    return None, None


# ── Order placement ───────────────────────────────────────────────────────────

def place_crypto_order(
    token_id: ClobTokenId,
    bet_size_usdc: float,
    market_id: ConditionId,
    log: logging.Logger,
    price_hint: Optional[float] = None,
    band_min: float = 0.0,
    band_max: float = 1.0,
    maker_mode: bool = True,
) -> dict:
    """
    Place a real BUY order on the Polymarket CLOB for a crypto market token.

    Procedure:
      1. H2: Check wallet balance — abort early with clear message if underfunded.
      MM: Maker attempt (when maker_mode=True):
          - Fetch live book to get best bid.
          - Place limit buy at best_bid + _MAKER_BID_OFFSET (aggressive maker).
          - Poll for fill up to _MAKER_FILL_TIMEOUT_SEC.
          - If filled: return with fill_type="MAKER" (0% fee, better price).
          - If not: cancel and fall through to taker.
      2. M1: Use price_hint (pre-fetched ask from loop.py) on first taker attempt
             to avoid a redundant CLOB book fetch.
      3. B1: On re-quotes (attempt 2+), re-check fresh price against band_min/max.
             If the price has drifted out of band, abort — do not place at a bad price.
      4. Convert: token_count = bet_size_usdc / price.
      5. H1: Place a GTC BUY order via _clob_timeout() (prevents hangs).
      6. Poll for fill up to _FILL_TIMEOUT_SEC seconds.
      7. If unfilled, cancel and re-quote up to _MAX_REQUOTES times total.

    Args:
        token_id      : CLOB token to buy (up_token or down_token from loop.py)
        bet_size_usdc : USDC amount to spend
        market_id     : for logging only
        log           : logger from loop.py
        price_hint    : pre-fetched min(ask) from _get_clob_spread(); used on
                        first taker attempt to skip a duplicate CLOB book fetch (M1)
        band_min      : lower bound of acceptable fill price (from MIN_CLOB_PRICE)
        band_max      : upper bound of acceptable fill price (from MAX_CLOB_PRICE)
        maker_mode    : if True (default), attempt maker limit order first before
                        falling back to taker. Saves 2% fee + better avg fill.

    Returns dict with keys:
        status       : "FILLED" | "NO_FILL"
        fill_price   : float   — avg fill price per token (0.0 if not filled)
        filled_usdc  : float   — USDC cost of filled tokens (0.0 if not filled)
        order_id     : str | None
        fill_type    : "MAKER" | "TAKER" | "NONE"
        reason       : str
    """
    def _no_fill(reason: str) -> dict:
        return {
            "status":      "NO_FILL",
            "fill_price":  0.0,
            "filled_usdc": 0.0,
            "order_id":    None,
            "fill_type":   "NONE",
            "reason":      reason,
        }

    client = _get_clob_client()
    if client is None:
        return _no_fill("CLOB client unavailable (missing KEY/FUNDER)")

    # ── H2: balance preflight ─────────────────────────────────────────────────
    # Fail fast with a readable message instead of a cryptic API error later.
    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        bal_data = _clob_timeout(
            client.get_balance_allowance,
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )
        if isinstance(bal_data, dict):
            raw_bal     = float(bal_data.get("balance", bal_data.get("USDC", 0)) or 0)
            wallet_usdc = raw_bal / 1_000_000 if raw_bal > 1_000 else raw_bal
            if wallet_usdc < bet_size_usdc:
                log.error(
                    "place_crypto_order: WALLET EMPTY  balance=$%.2f  bet=$%.2f",
                    wallet_usdc, bet_size_usdc,
                )
                return _no_fill(
                    f"insufficient wallet balance ${wallet_usdc:.2f} < bet ${bet_size_usdc:.2f}"
                )
    except concurrent.futures.TimeoutError:
        # Balance check timed out — skip trade rather than risk overdraft
        return _no_fill("balance check timed out — skipping trade for safety")
    except Exception as exc:
        log.warning("place_crypto_order: balance check failed (%s) — proceeding", exc)

    # ── MM: maker mode — try limit bid before lifting the ask ─────────────────
    # Place a GTC buy just above the current best bid. If a seller hits our order
    # we get filled at a better price AND pay 0% maker fee (vs 2% taker fee).
    # Falls through to the taker loop below if not filled within _MAKER_FILL_TIMEOUT_SEC.
    if maker_mode:
        best_ask, best_bid = _fetch_book(token_id)
        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            maker_price = round(best_bid + _MAKER_BID_OFFSET, 4)
            # Stay strictly below the ask so the order rests as a maker bid, not a taker.
            maker_price = min(maker_price, round(best_ask - 0.001, 4))
            maker_price = round(maker_price, 4)
            in_band = (band_max <= band_min) or (band_min <= maker_price <= band_max)
            if in_band:
                maker_tokens = round(bet_size_usdc / maker_price, 2)
                if maker_tokens >= 5.0:
                    log.info(
                        "place_crypto_order MAKER: bid=%.4f ask=%.4f → limit=%.4f  "
                        "tokens=%.2f  usdc=$%.2f",
                        best_bid, best_ask, maker_price, maker_tokens, bet_size_usdc,
                    )
                    maker_oid = None
                    try:
                        maker_oid = _clob_timeout(
                            _place_order, client, token_id, maker_price, maker_tokens, "BUY"
                        )
                    except concurrent.futures.TimeoutError:
                        log.warning("place_crypto_order: maker order timed out — falling back to taker")
                    except Exception as exc:
                        log.warning("place_crypto_order: maker order failed (%s) — falling back to taker", exc)

                    if maker_oid:
                        maker_deadline = time.monotonic() + _MAKER_FILL_TIMEOUT_SEC
                        while time.monotonic() < maker_deadline:
                            try:
                                info = _clob_timeout(_get_fill_status, client, maker_oid)
                            except Exception:
                                time.sleep(_POLL_INTERVAL_SEC)
                                continue
                            status = info.get("status", "UNKNOWN")
                            if status == "MATCHED":
                                avg_price   = float(info.get("avg_price") or maker_price)
                                filled_toks = float(info.get("filled_size") or maker_tokens)
                                filled_usdc = round(filled_toks * avg_price, 4)
                                saved       = round((best_ask - avg_price) * filled_toks, 4)
                                log.info(
                                    "place_crypto_order: MAKER FILL  order=%s  avg=%.4f  "
                                    "tokens=%.2f  usdc=$%.4f  saved=$%.4f vs ask",
                                    maker_oid, avg_price, filled_toks, filled_usdc, saved,
                                )
                                return {
                                    "status":      "FILLED",
                                    "fill_price":  round(avg_price, 6),
                                    "filled_usdc": filled_usdc,
                                    "order_id":    maker_oid,
                                    "fill_type":   "MAKER",
                                    "reason":      "maker fill",
                                }
                            if status == "CANCELLED":
                                break
                            time.sleep(_POLL_INTERVAL_SEC)

                        # Not filled in time — cancel and fall through to taker
                        try:
                            _clob_timeout(_cancel_order, client, maker_oid)
                        except Exception:
                            pass
                        log.info(
                            "place_crypto_order: MAKER unfilled after %ds — "
                            "cancelled %s, falling back to taker",
                            _MAKER_FILL_TIMEOUT_SEC, maker_oid,
                        )

    # ── Taker mode: lift the ask (existing logic) ─────────────────────────────
    attempt       = 0
    last_order_id = None

    while attempt < _MAX_REQUOTES:
        attempt += 1

        # ── M1: use pre-fetched ask on first attempt; re-fetch on requotes ────
        if attempt == 1 and price_hint is not None:
            price = price_hint
        else:
            price, _ = _fetch_book(token_id)

        if price is None:
            log.warning(
                "place_crypto_order: no asks for token %s (attempt %d)",
                token_id[:16], attempt,
            )
            continue

        # ── B1: check band on all attempts (not just requotes) ─────────────────
        if band_max > band_min and not (band_min <= price <= band_max):
            log.warning(
                "place_crypto_order: re-quote price %.4f outside band [%.2f–%.2f]"
                " — aborting (B1)",
                price, band_min, band_max,
            )
            return _no_fill(
                f"re-quote price {price:.4f} outside band [{band_min:.2f}–{band_max:.2f}]"
            )

        token_count = round(bet_size_usdc / price, 2)
        # Polymarket CLOB enforces a minimum of 5 tokens per order.
        if token_count < 5.0:
            adjusted = round(5.0 * price, 4)
            from crypto.loop import MAX_BET_ABS as _MAX_BET_ABS  # type: ignore
            if adjusted > _MAX_BET_ABS:
                return _no_fill(
                    f"5-token minimum ${adjusted:.2f} exceeds max_bet_abs "
                    f"${_MAX_BET_ABS:.2f} — skipping"
                )
            log.info(
                "place_crypto_order: adjusting bet $%.2f→$%.2f for 5-token minimum "
                "at price %.4f",
                bet_size_usdc, adjusted, price,
            )
            bet_size_usdc = adjusted
            token_count   = 5.0
        if token_count < 1.0:
            return _no_fill(
                f"token_count {token_count:.2f} < 1 at price {price:.4f}"
            )

        log.info(
            "place_crypto_order TAKER attempt %d/%d: price=%.4f  tokens=%.2f  usdc=$%.2f",
            attempt, _MAX_REQUOTES, price, token_count, bet_size_usdc,
        )

        # ── H1: place order with timeout ──────────────────────────────────────
        try:
            order_id = _clob_timeout(
                _place_order, client, token_id, price, token_count, "BUY"
            )
        except concurrent.futures.TimeoutError:
            log.error(
                "place_crypto_order: _place_order timed out after %ds (attempt %d)",
                CLOB_TIMEOUT, attempt,
            )
            continue
        except Exception as exc:
            log.error(
                "place_crypto_order: _place_order error: %s (attempt %d)", exc, attempt
            )
            continue

        if order_id is None:
            log.error(
                "place_crypto_order: _place_order returned None (attempt %d)", attempt
            )
            continue

        last_order_id = order_id
        deadline      = time.monotonic() + _FILL_TIMEOUT_SEC

        while time.monotonic() < deadline:
            # ── H1: fill status poll with timeout ─────────────────────────────
            try:
                info = _clob_timeout(_get_fill_status, client, order_id)
            except concurrent.futures.TimeoutError:
                log.warning("place_crypto_order: fill poll timed out — retrying")
                time.sleep(_POLL_INTERVAL_SEC)
                continue
            except Exception as exc:
                log.warning("place_crypto_order: fill poll error: %s", exc)
                time.sleep(_POLL_INTERVAL_SEC)
                continue

            status = info.get("status", "UNKNOWN")

            if status == "MATCHED":
                avg_price   = float(info.get("avg_price") or price)
                filled_toks = float(info.get("filled_size") or token_count)
                filled_usdc = round(filled_toks * avg_price, 4)
                log.info(
                    "place_crypto_order: TAKER FILL  order=%s  avg=%.4f  "
                    "tokens=%.2f  usdc=$%.4f",
                    order_id, avg_price, filled_toks, filled_usdc,
                )
                return {
                    "status":      "FILLED",
                    "fill_price":  round(avg_price, 6),
                    "filled_usdc": filled_usdc,
                    "order_id":    order_id,
                    "fill_type":   "TAKER",
                    "reason":      f"taker fill on attempt {attempt}",
                }

            if status == "CANCELLED":
                break

            time.sleep(_POLL_INTERVAL_SEC)

        # Cancel stale order before re-quoting (best-effort)
        try:
            _clob_timeout(_cancel_order, client, order_id)
        except Exception:
            pass
        log.info(
            "place_crypto_order: attempt %d unfilled — cancelled %s, re-quoting",
            attempt, order_id,
        )

    log.warning(
        "place_crypto_order: NO_FILL after %d attempt(s)  token=%s",
        attempt, token_id[:16],
    )
    return {
        "status":      "NO_FILL",
        "fill_price":  0.0,
        "filled_usdc": 0.0,
        "order_id":    last_order_id,
        "fill_type":   "NONE",
        "reason":      f"no fill after {attempt} attempt(s)",
    }


# ── Mid-window exit: SELL owned tokens ───────────────────────────────────────

def sell_crypto_position(
    token_id: ClobTokenId,
    token_count: float,
    market_id: ConditionId,
    log: logging.Logger,
) -> dict:
    """
    Place a GTC SELL order for token_count tokens of the owned position.
    Used by the mid-window profit-taking exit in loop.py.

    Returns dict with keys:
        status       : "FILLED" | "NO_BID" | "BID_FAIL" | "NO_CLIENT" | "ORDER_FAIL"
        fill_price   : float — bid price achieved (0.0 on failure)
        proceeds_usdc: float — token_count × fill_price (0.0 on failure)
    """
    def _no_sell(status: str) -> dict:
        return {"status": status, "fill_price": 0.0, "proceeds_usdc": 0.0}

    client = _get_clob_client()
    if client is None:
        return _no_sell("NO_CLIENT")

    # Fetch best bid — the price we'd receive by placing a limit sell.
    try:
        r = _HTTP.get(
            "https://clob.polymarket.com/book",
            params={"token_id": str(token_id)},
            timeout=6.0,
        )
        r.raise_for_status()
        bids = r.json().get("bids", [])
        best_bid = max(float(b["price"]) for b in bids) if bids else None
    except Exception as exc:
        log.warning("sell_crypto_position: bid fetch failed for %s: %s", str(token_id)[:12], exc)
        return _no_sell("BID_FAIL")

    if best_bid is None or best_bid < 0.50:
        log.info(
            "sell_crypto_position: no viable bid (%.3f) — skip  token=%s",
            best_bid or 0.0, str(token_id)[:12],
        )
        return _no_sell("NO_BID")

    size = round(token_count, 4)
    try:
        from py_clob_client_v2.clob_types import OrderArgs
        args = OrderArgs(
            token_id=str(token_id),
            price=best_bid,
            size=size,
            side="SELL",
        )
        _clob_timeout(client.create_and_post_order, args)
        proceeds = round(best_bid * token_count, 6)
        log.info(
            "sell_crypto_position: SELL %.4f tokens @ %.4f → $%.4f  market=%s",
            size, best_bid, proceeds, str(market_id)[:12],
        )
        return {"status": "FILLED", "fill_price": best_bid, "proceeds_usdc": proceeds}
    except Exception as exc:
        log.warning("sell_crypto_position: order failed for %s: %s", str(market_id)[:12], exc)
        return _no_sell("ORDER_FAIL")
