"""
hyperliquid/execution.py — real order placement on Hyperliquid perpetuals.

Mirrors the structure of crypto/execution.py:
  _hl_timeout()      — ThreadPoolExecutor wrapper preventing SDK call hangs
  get_hl_balance()   — USDC account balance query (for startup C1 sync)
  place_hl_order()   — enter a leveraged perp position with TP + SL orders
  close_hl_position()— market-close an open position (max_hold_time exit)
  cancel_hl_order()  — cancel a single bracket order by ID

Dependencies (add to requirements.txt / pyproject.toml before going live):
  hyperliquid-python-sdk
  eth_account   (usually installed with web3)

Environment variables (set in .env):
  HL_ADDRESS  — wallet address (0x...)
  HL_KEY      — private key for signing orders

VIRTUAL_MODE guard: this module is only imported by loop.py when
HL_VIRTUAL_MODE=false. Never import unconditionally.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperliquid.feed import get_mid_price, get_sz_decimals  # type: ignore

_HL_CALL_TIMEOUT   = 10   # hard timeout per SDK call (seconds) — mirrors _CLOB_CALL_TIMEOUT
_ORDER_TIMEOUT_SEC = 15   # how long to wait for a market order to fill
_POLL_INTERVAL_SEC = 2

# Per-coin leverage currently set on the exchange (cached to avoid redundant calls)
_leverage_set: dict[str, int] = {}

# ── SDK loader (bypasses src/hyperliquid/ name conflict) ──────────────────────
_SDK: dict | None = None

def _load_sdk() -> dict:
    """
    Load hyperliquid SDK classes once, bypassing the local src/hyperliquid/ package
    that shadows the installed hyperliquid-python-sdk on sys.path.
    Caches result so path manipulation only happens on first call.
    """
    global _SDK
    if _SDK is not None:
        return _SDK

    # Temporarily remove src/ so Python finds the installed SDK, not local package
    removed: list[tuple[int, str]] = []
    i = 0
    while i < len(sys.path):
        if sys.path[i] == str(_SRC):
            removed.append((i, sys.path.pop(i)))
        else:
            i += 1

    # Clear any cached local hyperliquid entries from sys.modules
    old_mods: dict = {}
    for k in list(sys.modules):
        if k == "hyperliquid" or k.startswith("hyperliquid."):
            old_mods[k] = sys.modules.pop(k)

    try:
        from hyperliquid.info     import Info      # type: ignore
        from hyperliquid.exchange import Exchange   # type: ignore
        from hyperliquid.utils    import constants  # type: ignore
        from eth_account          import Account    # type: ignore
        _SDK = {"Info": Info, "Exchange": Exchange, "constants": constants, "Account": Account}
    except ImportError as exc:
        raise RuntimeError(
            "hyperliquid-python-sdk or eth_account not installed. "
            "Run: pip install hyperliquid-python-sdk eth_account"
        ) from exc
    finally:
        # Restore src/ to sys.path
        for idx, p in sorted(removed):
            sys.path.insert(idx, p)
        # Restore local hyperliquid modules; keep SDK submodules already loaded
        for k, v in old_mods.items():
            if k not in sys.modules:
                sys.modules[k] = v

    return _SDK


# ── Timeout wrapper ───────────────────────────────────────────────────────────

def _hl_timeout(fn, *args, timeout: int = _HL_CALL_TIMEOUT):
    """
    Run fn(*args) in a thread with a hard timeout.
    Prevents Hyperliquid SDK TCP hangs from blocking the loop indefinitely.
    Same pattern as _clob_timeout() in crypto/execution.py.
    Raises concurrent.futures.TimeoutError on timeout.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args).result(timeout=timeout)


# ── Client factory ────────────────────────────────────────────────────────────

def _get_hl_clients():
    """
    Build and return (info, exchange) clients.
    Returns (None, None) if HL_ADDRESS or HL_KEY are missing.

    info     — hyperliquid.info.Info (read-only, no auth)
    exchange — hyperliquid.exchange.Exchange (signing, requires HL_KEY)
    """
    sdk = _load_sdk()
    Info      = sdk["Info"]
    Exchange  = sdk["Exchange"]
    constants = sdk["constants"]
    Account   = sdk["Account"]

    address = os.getenv("HL_ADDRESS")
    key     = os.getenv("HL_KEY")
    if not address or not key:
        return None, None

    try:
        wallet   = Account.from_key(key)
        info     = Info(constants.MAINNET_API_URL, skip_ws=True)
        exchange = Exchange(wallet, constants.MAINNET_API_URL)
        return info, exchange
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise HL clients: {exc}") from exc


# ── Balance query ─────────────────────────────────────────────────────────────

def get_hl_balance(log: logging.Logger) -> Optional[float]:
    """
    Query USDC balance from Hyperliquid.
    In unified account mode, spot USDC serves as perp collateral — accountValue
    shows $0 until a position is opened. We add the spot USDC balance to cover
    this case so the C1 check and trade sizing both work correctly.
    Returns float USDC or None on failure.
    """
    import httpx as _httpx
    address = os.getenv("HL_ADDRESS", "")
    if not address:
        return None
    try:
        # Perp account value
        r1 = _httpx.post("https://api.hyperliquid.xyz/info",
            json={"type": "clearinghouseState", "user": address}, timeout=10)
        perp_val = float(r1.json().get("marginSummary", {}).get("accountValue", 0))

        # Spot USDC (in unified mode this IS the available collateral)
        r2 = _httpx.post("https://api.hyperliquid.xyz/info",
            json={"type": "spotClearinghouseState", "user": address}, timeout=10)
        spot_usdc = 0.0
        for b in r2.json().get("balances", []):
            if b.get("coin") == "USDC":
                spot_usdc = float(b.get("total", 0))
                break

        total = round(perp_val + spot_usdc, 4)
        log.info("HL balance — perp: $%.2f  spot USDC: $%.2f  total: $%.2f",
                 perp_val, spot_usdc, total)
        return total
    except Exception as exc:
        log.warning("get_hl_balance failed: %s", exc)
    return None


# ── Leverage management ───────────────────────────────────────────────────────

def _ensure_leverage(exchange, coin: str, leverage: int) -> None:
    """
    Set cross-margin leverage for coin if not already at the desired level.
    Cached per session — avoids a redundant SDK call on every trade.
    """
    if _leverage_set.get(coin) == leverage:
        return
    _hl_timeout(exchange.update_leverage, leverage, coin, True)
    _leverage_set[coin] = leverage


# ── Order placement ───────────────────────────────────────────────────────────

def place_hl_order(
    coin: str,
    side: str,                       # "long" or "short"
    size_usdc: float,                # notional USD (pre-leverage)
    leverage: int,
    tp_pct: float,
    sl_pct: float,
    log: logging.Logger,
    mid_price_hint: Optional[float] = None,
) -> dict:
    """
    Open a leveraged perp position on Hyperliquid with TP + SL bracket orders.

    Procedure:
      1. Balance preflight — abort if accountValue < size_usdc.
      2. Set cross-margin leverage (cached — no-op if already set).
      3. Compute contract size: sz = size_usdc * leverage / mid_price.
         Rounded to szDecimals for the coin (BTC=5, ETH=4).
      4. Place market order (exchange.market_open).
      5. Extract fill price from response.
      6. Place TP limit order (reduce_only=True, Gtc).
      7. Place SL trigger order (reduce_only=True, isMarket=True).

    Returns dict:
        status          : "FILLED" | "NO_FILL"
        fill_price      : float
        size_contracts  : float
        size_usd        : float  (actual notional at fill)
        tp_order_id     : int | None
        sl_order_id     : int | None
        tp_price        : float
        sl_price        : float
        reason          : str
    """
    def _no_fill(reason: str) -> dict:
        return {
            "status": "NO_FILL", "fill_price": 0.0, "size_contracts": 0.0,
            "size_usd": 0.0, "tp_order_id": None, "sl_order_id": None,
            "tp_price": 0.0, "sl_price": 0.0, "reason": reason,
        }

    info, exchange = _get_hl_clients()
    if info is None or exchange is None:
        return _no_fill("HL clients unavailable (missing HL_ADDRESS or HL_KEY)")

    address = os.getenv("HL_ADDRESS", "")
    is_buy  = (side == "long")

    # ── Balance preflight ─────────────────────────────────────────────────────
    try:
        state   = _hl_timeout(info.user_state, address)
        balance = float(state["marginSummary"]["accountValue"])
        if balance < size_usdc:
            log.error(
                "place_hl_order: WALLET INSUFFICIENT  balance=$%.2f  needed=$%.2f",
                balance, size_usdc,
            )
            return _no_fill(f"insufficient balance ${balance:.2f} < ${size_usdc:.2f}")
    except concurrent.futures.TimeoutError:
        return _no_fill("balance preflight timed out")
    except Exception as exc:
        log.warning("place_hl_order: balance preflight failed (%s) — proceeding", exc)

    # ── Mid price ─────────────────────────────────────────────────────────────
    mid_price = mid_price_hint or get_mid_price(coin)
    if mid_price is None or mid_price <= 0:
        return _no_fill(f"could not get mid price for {coin}")

    # ── Contract size ─────────────────────────────────────────────────────────
    sz_decimals = get_sz_decimals()
    n_decimals  = sz_decimals.get(coin, 5)
    sz          = round(size_usdc * leverage / mid_price, n_decimals)
    if sz <= 0:
        return _no_fill(f"computed sz={sz} <= 0 at price {mid_price}")

    log.info(
        "place_hl_order: %s %s  sz=%.6f  price_hint=%.2f  leverage=%dx  usdc=$%.2f",
        side.upper(), coin, sz, mid_price, leverage, size_usdc,
    )

    # ── Set leverage ──────────────────────────────────────────────────────────
    try:
        _ensure_leverage(exchange, coin, leverage)
    except Exception as exc:
        log.warning("place_hl_order: update_leverage failed (%s) — proceeding", exc)

    # ── Market entry order ────────────────────────────────────────────────────
    try:
        result = _hl_timeout(
            exchange.market_open, coin, is_buy, sz, None, 0.01
        )
    except concurrent.futures.TimeoutError:
        return _no_fill("market_open timed out")
    except Exception as exc:
        log.error("place_hl_order: market_open failed: %s", exc)
        return _no_fill(f"market_open error: {exc}")

    # Parse fill from response
    fill_price     = 0.0
    size_contracts = sz
    try:
        statuses = result["response"]["data"]["statuses"]
        filled   = statuses[0].get("filled", {})
        if not filled:
            return _no_fill(f"order not filled: {statuses[0]}")
        fill_price     = float(filled.get("avgPx", mid_price))
        size_contracts = float(filled.get("totalSz", sz))
    except Exception as exc:
        log.warning("place_hl_order: could not parse fill response (%s)", exc)
        fill_price = mid_price   # best guess

    size_usd = round(size_contracts * fill_price, 2)
    log.info(
        "place_hl_order: FILLED %s %s  fill=%.4f  sz=%.6f  usd=$%.2f",
        side.upper(), coin, fill_price, size_contracts, size_usd,
    )

    # ── TP / SL bracket orders ────────────────────────────────────────────────
    if is_buy:
        tp_price = round(fill_price * (1 + tp_pct), 2)
        sl_price = round(fill_price * (1 - sl_pct), 2)
    else:
        tp_price = round(fill_price * (1 - tp_pct), 2)
        sl_price = round(fill_price * (1 + sl_pct), 2)

    tp_order_id = _place_tp_order(exchange, coin, is_buy, size_contracts, tp_price, log)
    sl_order_id = _place_sl_order(exchange, coin, is_buy, size_contracts, sl_price, log)

    return {
        "status":         "FILLED",
        "fill_price":     fill_price,
        "size_contracts": size_contracts,
        "size_usd":       size_usd,
        "tp_order_id":    tp_order_id,
        "sl_order_id":    sl_order_id,
        "tp_price":       tp_price,
        "sl_price":       sl_price,
        "reason":         "market fill",
    }


def _place_tp_order(
    exchange,
    coin: str,
    entry_is_buy: bool,
    sz: float,
    tp_price: float,
    log: logging.Logger,
) -> Optional[int]:
    """Place a limit reduce_only order to take profit. Returns order ID or None."""
    is_buy = not entry_is_buy   # TP closes the position: buy → sell TP, short → buy TP
    try:
        result = _hl_timeout(
            exchange.order,
            coin, is_buy, sz, tp_price,
            {"limit": {"tif": "Gtc"}},
            None,    # cloid
            True,    # reduce_only
        )
        oid = result["response"]["data"]["statuses"][0].get("resting", {}).get("oid")
        log.info("TP order placed: oid=%s  price=%.4f", oid, tp_price)
        return oid
    except Exception as exc:
        log.warning("_place_tp_order failed: %s", exc)
        return None


def _place_sl_order(
    exchange,
    coin: str,
    entry_is_buy: bool,
    sz: float,
    sl_price: float,
    log: logging.Logger,
) -> Optional[int]:
    """Place a trigger (stop-market) reduce_only order for stop loss. Returns order ID or None."""
    is_buy = not entry_is_buy
    try:
        result = _hl_timeout(
            exchange.order,
            coin, is_buy, sz, sl_price,
            {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}},
            None,    # cloid
            True,    # reduce_only
        )
        oid = result["response"]["data"]["statuses"][0].get("resting", {}).get("oid")
        log.info("SL order placed: oid=%s  price=%.4f", oid, sl_price)
        return oid
    except Exception as exc:
        log.warning("_place_sl_order failed: %s", exc)
        return None


# ── Position close ────────────────────────────────────────────────────────────

def close_hl_position(
    coin: str,
    side: str,               # the side we are IN ("long" or "short")
    size_contracts: float,
    tp_order_id: Optional[int],
    sl_order_id: Optional[int],
    log: logging.Logger,
) -> dict:
    """
    Market-close an open HL position immediately (max_hold_time expiry).
    Also cancels any outstanding TP/SL bracket orders.

    Returns dict with keys: status, fill_price, reason.
    """
    _, exchange = _get_hl_clients()
    if exchange is None:
        return {"status": "ERROR", "fill_price": 0.0, "reason": "no exchange client"}

    is_buy = (side == "short")   # closing a long → sell; closing a short → buy

    # Cancel bracket orders first (best-effort)
    for oid in [tp_order_id, sl_order_id]:
        if oid is not None:
            cancel_hl_order(coin, oid, log)

    try:
        result = _hl_timeout(exchange.market_open, coin, is_buy, size_contracts, None, 0.01, True)
        fill_price = 0.0
        try:
            filled     = result["response"]["data"]["statuses"][0].get("filled", {})
            fill_price = float(filled.get("avgPx", 0))
        except Exception:
            pass
        log.info("close_hl_position: %s %s closed at %.4f", side, coin, fill_price)
        return {"status": "CLOSED", "fill_price": fill_price, "reason": "max_hold_time"}
    except concurrent.futures.TimeoutError:
        return {"status": "TIMEOUT", "fill_price": 0.0, "reason": "market_close timed out"}
    except Exception as exc:
        log.error("close_hl_position failed: %s", exc)
        return {"status": "ERROR", "fill_price": 0.0, "reason": str(exc)}


# ── Order cancellation ────────────────────────────────────────────────────────

def cancel_hl_order(coin: str, order_id: int, log: logging.Logger) -> bool:
    """Cancel a single HL order by ID. Returns True on success."""
    _, exchange = _get_hl_clients()
    if exchange is None:
        return False
    try:
        _hl_timeout(exchange.cancel, coin, order_id)
        log.debug("Cancelled HL order %s for %s", order_id, coin)
        return True
    except Exception as exc:
        log.debug("cancel_hl_order(%s, %s) failed: %s", coin, order_id, exc)
        return False


# ── Position state check ──────────────────────────────────────────────────────

def get_open_position(coin: str, log: logging.Logger) -> Optional[dict]:
    """
    Return the current HL open position for coin, or None if flat.
    Used by loop.py to detect TP/SL fills without an explicit callback.

    Returns dict with keys: size (contracts), side ("long"/"short"), entry_px, unrealized_pnl.
    """
    info, _ = _get_hl_clients()
    if info is None:
        return None
    address = os.getenv("HL_ADDRESS", "")
    try:
        state     = _hl_timeout(info.user_state, address)
        positions = state.get("assetPositions", [])
        for item in positions:
            pos = item.get("position", {})
            if pos.get("coin") == coin:
                sz = float(pos.get("szi", 0))
                if sz == 0:
                    return None
                return {
                    "size":            abs(sz),
                    "side":            "long" if sz > 0 else "short",
                    "entry_px":        float(pos.get("entryPx",       0)),
                    "unrealized_pnl":  float(pos.get("unrealizedPnl", 0)),
                }
    except Exception as exc:
        log.debug("get_open_position(%s) failed: %s", coin, exc)
    return None
