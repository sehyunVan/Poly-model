"""
bot/execution.py — Real CLOB order placement for AI swarm picks.

Uses py_clob_client (same as the crypto loop) for authenticated order placement.
The bot must run in the server venv where py_clob_client is installed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

CLOB_HOST           = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
EXEC_MIN_SCORE      = 0.60
EXEC_MIN_UNANIMOUS  = False  # ★ relaxed True→False (2026-05-14): paper-only NO with 1 YES dissenter = 17 trades, 94.1% WR, +$30 real PnL; dissent is not a quality warning
EXEC_FLAT_BET       = 100.0 # target bet — raised $80→$100 on 2026-06-26 (user). NOTE: still capped at balance×0.35 below, so $100 only BINDS while balance ≥ ~$286; at the current ~$200-280 liquid the effective bet is the 0.35 cap (~$70-98), so this mostly matters once the wallet grows. Revert to 80.0 if WR drops below ~78% over 50+ trades.
EXEC_MIN_BET        = 10.0  # floor — don't trade below $10 (too small vs API cost)
EXEC_MIN_TOKENS     = 5.0   # Polymarket CLOB minimum token count per order
_CLOB_TIMEOUT_S     = 15

# ── Payout-proportional sizing (2026-06-11) — CURRENTLY INACTIVE ──────────────
# Superseded same day by the engine-zone-only gate in execute_pick() (we now
# restrict to [0.70, 0.78) and bet flat max, rather than trading all bands at
# scaled size). Function kept for one-line reversal if the band is ever widened.
# 14d audit (159 settled NO trades, +$432, 86.8% WR) showed profit is wildly
# concentrated by NO ask = payout size:
#   [0.70-0.75) 93.5% WR +34.4% ROI  -> 62% of all profit from 19% of trades
#   [0.75-0.85) ~0% ROI (dead capital)
#   [0.85-0.95) 100% WR but +5-14% ROI, thin payout, fragile small-sample
# Flat $40 sizing bet the same on a fat 0.72 NO as a thin 0.95 NO. Kelly says
# the opposite: at ask>=0.92 net odds are <0.09, so optimal stake collapses even
# at high WR (and one loss wipes >11 wins). We now scale bet by net-odds ratio
# vs an anchor at 0.78 (the top of the engine zone), clamped to [floor, 1.0].
# This is ~quarter-Kelly given the roughly-flat ~85% WR across bands, and a
# monotone-decreasing-in-ask curve faithful to both observed hypotheses
# (cheap NO = fat edge = bet big; expensive NO = thin profit = bet small).
# Reversal: set _SIZE_MULT_FLOOR = 1.0 to restore flat sizing.
_SIZE_ANCHOR_ASK    = 0.78   # asks <= this get full size
_SIZE_MULT_FLOOR    = 0.25   # smallest multiplier (× base $40 = $10 min bet)


def _payout_size_mult(ask: float) -> float:
    """Bet multiplier in [_SIZE_MULT_FLOOR, 1.0]: full size for fat-payout cheap
    NO, shrinking as ask rises and the NO payout thins out."""
    net = (1.0 - ask) / ask
    ref = (1.0 - _SIZE_ANCHOR_ASK) / _SIZE_ANCHOR_ASK
    return max(_SIZE_MULT_FLOOR, min(1.0, net / ref))

# Maker mode: try a limit order inside the spread before lifting the ask.
# Swarm markets are 24h+ so there is no time pressure — we can wait 5 minutes.
_SWARM_MAKER_OFFSET  = 0.012  # bid this many cents inside the ask (maker placement)
_SWARM_MAKER_TIMEOUT = 60     # ★ LOWERED 300 → 60 (2026-05-07): on-chain audit showed
                              # only ~20% of maker attempts filled in 5 min before
                              # price moved away. 60s gives taker fallback a chance
                              # while signal is still fresh.
_SWARM_POLL_INTERVAL = 15     # seconds between fill status polls

# Import wallet share — swarm bot uses only this fraction of the total CLOB balance.
try:
    from bot.config import SWARM_WALLET_SHARE  # type: ignore
except ImportError:
    SWARM_WALLET_SHARE = 0.50

# Add src/ to path so we can import the existing CLOB helpers
_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Timeout wrapper ───────────────────────────────────────────────────────────

def _clob_call(fn, *args, timeout: int = _CLOB_TIMEOUT_S):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args).result(timeout=timeout)


# ── CLOB client (lazy, cached) ────────────────────────────────────────────────

_client_cache = None

def _get_client():
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    try:
        from py_clob_client_v2.client import ClobClient  # type: ignore
        from py_clob_client_v2.constants import POLYGON   # type: ignore
        key    = os.getenv("KEY")
        funder = os.getenv("FUNDER")
        host   = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
        if not key or not funder:
            raise RuntimeError("KEY or FUNDER not set in .env")
        client = ClobClient(host, key=key, chain_id=POLYGON,
                            funder=funder, signature_type=0)
        client.set_api_creds(client.create_or_derive_api_key())
        _client_cache = client
        return client
    except ImportError:
        raise RuntimeError("py_clob_client not installed")


# ── Balance ───────────────────────────────────────────────────────────────────

def _get_balance_sync() -> float:
    from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType  # type: ignore
    client = _get_client()
    resp   = _clob_call(lambda: client.get_balance_allowance(
        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    ))
    # resp is a dict: {"balance": "22960247", "allowances": {...}}
    return float(resp["balance"]) / 1e6


# ── Best ask ──────────────────────────────────────────────────────────────────

async def _get_best_ask(token_id: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=8) as h:
            r = await h.get(f"{CLOB_HOST}/book", params={"token_id": token_id})
            r.raise_for_status()
            asks = r.json().get("asks", [])
            return min(float(a["price"]) for a in asks) if asks else None
    except Exception as e:
        log.warning("Ask fetch failed for %s: %s", token_id[:12], e)
        return None


# ── Order placement ───────────────────────────────────────────────────────────

def _place_order_sync(token_id: str, price: float, size: float) -> bool:
    """Taker order — place at ask price, return True on success."""
    from py_clob_client_v2.clob_types import OrderArgs  # type: ignore
    client = _get_client()
    args   = OrderArgs(token_id=token_id, price=price, size=size, side="BUY")
    resp   = _clob_call(lambda: client.create_and_post_order(args))
    log.info("Order response: %s", resp)
    return True


def _place_maker_order_sync(token_id: str, price: float, size: float) -> Optional[str]:
    """
    Place a maker limit BUY order and return the order_id (or None on failure).
    At 0% maker fee this saves ~2% vs taker. Returns None on any error.
    """
    from py_clob_client_v2.clob_types import OrderArgs  # type: ignore
    try:
        client = _get_client()
        args   = OrderArgs(token_id=token_id, price=price, size=size, side="BUY")
        resp   = _clob_call(lambda: client.create_and_post_order(args))
        log.info("Maker order response: %s", resp)
        if isinstance(resp, dict):
            oid = resp.get("orderID") or resp.get("order_id") or resp.get("id")
            return str(oid) if oid else None
    except Exception as exc:
        log.warning("_place_maker_order_sync failed: %s", exc)
    return None


def _get_order_status_sync(order_id: str) -> str:
    """Return the current status of an order: MATCHED, OPEN, CANCELLED, or UNKNOWN."""
    try:
        client = _get_client()
        resp   = _clob_call(lambda: client.get_order(order_id))
        if isinstance(resp, dict):
            return resp.get("status", "UNKNOWN")
    except Exception as exc:
        log.warning("_get_order_status_sync(%s): %s", order_id[:12], exc)
    return "UNKNOWN"


def _cancel_order_sync(order_id: str) -> bool:
    """
    Cancel an open order. Returns True if confirmed cancelled, False on error.
    Caller MUST check the return value before placing a taker fallback —
    a failed cancel leaves a GTC order in the book that could double-fill.

    ★ 2026-05-14: switched from cancel_order(str) to cancel_orders([str]).
    In py_clob_client_v2, cancel_order() expects an OrderPayload object — passing
    a string raised AttributeError on every cancel, causing 2h cooldowns on every
    unfilled maker. cancel_orders([id]) is the working batch API and returns
    {"canceled": [...], "not_canceled": {...}}.
    """
    try:
        client = _get_client()
        resp = _clob_call(lambda: client.cancel_orders([order_id]))
        canceled = resp.get("canceled", []) if isinstance(resp, dict) else []
        if order_id in canceled:
            log.info("Maker order cancelled: %s", order_id[:12])
            return True
        # Order not in canceled list — could be already filled, expired, or not found.
        not_canceled = resp.get("not_canceled", {}) if isinstance(resp, dict) else {}
        log.warning(
            "_cancel_order_sync(%s) returned without confirmation — resp=%s",
            order_id[:12], str(resp)[:200],
        )
        # Treat "not found" / "already filled" as success (no live order to worry about).
        reason = str(not_canceled.get(order_id, "")).lower() if isinstance(not_canceled, dict) else ""
        if "not found" in reason or "filled" in reason or "matched" in reason:
            return True
        return False
    except Exception as exc:
        log.warning("_cancel_order_sync(%s) FAILED — taker fallback blocked: %s", order_id[:12], exc)
        return False


async def _poll_maker_fill(order_id: str, timeout: int, interval: int) -> bool:
    """
    Poll for a maker order fill up to `timeout` seconds.
    Returns True if MATCHED, False if timeout or CANCELLED.
    """
    loop     = asyncio.get_event_loop()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await loop.run_in_executor(None, _get_order_status_sync, order_id)
        if status == "MATCHED":
            return True
        if status == "CANCELLED":
            return False
        await asyncio.sleep(interval)
    return False


# ── Public: execute one pick ──────────────────────────────────────────────────

async def execute_pick(pick, reentry_info: Optional[dict] = None) -> Optional[dict]:
    """
    Place a real CLOB order for the given pick.
    Mirrors the paper tracker: flat $10 bet, no ask floors/ceilings, no direction flips.
    Returns {"ask": float, "bet": float, "fill_type": str, "direction": str} on success.
    """
    from bot.swarm.models import Decision

    market    = pick.market
    direction = pick.direction
    token_id  = market.no_token_id if direction == Decision.NO else market.yes_token_id

    # YES direction block (re-instated 2026-05-09).
    # All-time YES: 39.0% WR / -$109 (n=82). Last 14d YES: 42% WR / -$70.
    # The 2026-04-24 unblock was based on weak post-blind-vote evidence and
    # has not paid off. NO is the only structurally profitable direction
    # (75% WR / +$48 all-time). Re-block until a stronger YES edge is found.
    if direction == Decision.YES:
        log.info("SKIP YES direction — blocked  %s", market.question[:60])
        return None

    if not token_id:
        log.warning("No token_id for %s on %r", direction.value, market.question[:50])
        return None

    ask = await _get_best_ask(token_id)
    if ask is None:
        log.warning("No ask price for token %s", token_id[:12])
        return None

    # ── ENGINE-ZONE gate (floor 0.65 since 06-14; ceiling 0.92→0.85 on 2026-06-26) ──
    # Active band: [0.65, 0.85).
    # CEILING history: 0.95→0.78 (06-11) → 0.78→0.92 (06-15, for volume) → 0.92→0.85
    # (06-26, TRIPWIRE FIRED). The 06-15 re-open assumed [0.85,0.92) was the strongest
    # zone (+$9.02 EV/trade in that window), but it was fragile survivorship and reverted:
    # over 06-16→06-26 (n=43) [0.85,0.92) ran 86% WR / −$36 / −$0.84 EV — breaching the
    # pre-committed "WR<88% → pull ceiling" rule. At ask 0.85–0.92 breakeven WR ≈ ask, so
    # 86% tips net-negative and a cluster of favorite-upsets drained it fast.
    # Pulled to 0.85, NOT 0.78, because the fresh data shows [0.78,0.85) is now one of the
    # BEST zones (n=33, 88% WR, +$175, +$5.31 EV) — keep it. Zones kept (since 06-16):
    #   [0.65,0.78) 77% WR +$386 +$6.90 EV  ·  [0.78,0.85) 88% WR +$175 +$5.31 EV
    # Zone cut: [0.85,0.92) −$36 −$0.84 EV.
    # Reversal: ceiling 0.92 (re-open high-ask) or 0.78 (engine-core only).
    if ask < 0.65:
        log.info("SKIP engine-floor: ask=%.3f < 0.65 (below engine zone)  %s", ask, market.question[:55])
        ghost.record_ghost(pick, ask)   # ghost-track: does it rise into the zone?
        return None
    if ask >= 0.85:
        log.info("SKIP engine-ceiling: ask=%.3f >= 0.85 (payout too thin / weak cushion)  %s", ask, market.question[:55])
        ghost.record_ghost(pick, ask)   # ghost-track: does it fall into the zone? (adverse)
        return None

    # Safety: CLOB/Gamma divergence guard for YES bets only.
    # Catches stale Gamma price or token ID swap (e.g. ask=0.90 vs yes_price=0.30).
    if direction == Decision.YES and ask > market.yes_price * 1.5:
        log.info(
            "SKIP divergence: YES ask=%.3f >> yes_price=%.3f (stale Gamma or token mismatch)  %s",
            ask, market.yes_price, market.question[:50],
        )
        return None

    # Re-entry: only proceed if ask has improved ≥ 4% since the initial entry.
    if reentry_info is not None:
        initial_ask = reentry_info.get("ask", 1.0)
        if ask > initial_ask * 0.96:
            log.info(
                "SKIP RE-ENTRY ask=%.3f not improved vs initial=%.3f  %s",
                ask, initial_ask, market.question[:50],
            )
            return None
        log.info("RE-ENTRY APPROVED ask=%.3f improved from initial=%.3f  %s", ask, initial_ask, market.question[:50])

    loop = asyncio.get_event_loop()
    try:
        raw_balance = await loop.run_in_executor(None, _get_balance_sync)
    except Exception as e:
        log.error("Balance check failed: %s", e)
        return None

    balance = round(raw_balance * SWARM_WALLET_SHARE, 4)
    if balance < EXEC_MIN_BET:
        log.warning("Balance $%.2f below min bet $%.2f", balance, EXEC_MIN_BET)
        return None

    # Flat max bet inside the engine zone (capped at 35% of swarm budget). With
    # the band already gated to [0.70, 0.78), payout is uniformly fat — no need
    # to scale by ask, so we bet max on every qualifying pick. (_payout_size_mult
    # is retained but unused; re-enable it if the band is ever widened again.)
    bet  = min(EXEC_FLAT_BET, balance * 0.35)
    bet  = max(EXEC_MIN_BET, bet)
    size = math.ceil(bet / ask * 10000) / 10000

    if size < EXEC_MIN_TOKENS:
        log.info(
            "SKIP min-tokens: size=%.4f < %.1f required (ask=%.3f bet=$%.2f)  %s",
            size, EXEC_MIN_TOKENS, ask, bet, market.question[:50],
        )
        return None

    action = "RE-ENTRY" if reentry_info else "EXEC"
    log.info("%s %s  ask=%.3f  bet=$%.2f  tokens=%.4f  %s",
             action, direction.value, ask, bet, size, market.question[:55])

    # ── Maker attempt: post limit bid inside the spread, wait 5 min ──────────
    maker_price = round(ask - _SWARM_MAKER_OFFSET, 4)
    maker_size  = math.ceil(bet / maker_price * 10000) / 10000
    if maker_price > 0 and maker_size >= EXEC_MIN_TOKENS:
        log.info("MAKER attempt: ask=%.4f → limit=%.4f  tokens=%.4f  %s",
                 ask, maker_price, maker_size, market.question[:50])
        order_id = await loop.run_in_executor(
            None, _place_maker_order_sync, token_id, maker_price, maker_size
        )
        if order_id:
            filled = await _poll_maker_fill(
                order_id, timeout=_SWARM_MAKER_TIMEOUT, interval=_SWARM_POLL_INTERVAL
            )
            if filled:
                log.info("MAKER FILL %s @ %.4f  bet=$%.2f  %s",
                         order_id[:12], maker_price, bet, market.question[:50])
                return {"ask": maker_price, "bet": bet, "fill_type": "MAKER",
                        "direction": direction.value, "is_reentry": reentry_info is not None}
            cancelled = await loop.run_in_executor(None, _cancel_order_sync, order_id)
            if not cancelled:
                log.warning("MAKER cancel failed — skipping taker fallback  %s", market.question[:50])
                return None
            log.info("MAKER unfilled after %ds — falling back to taker  %s",
                     _SWARM_MAKER_TIMEOUT, market.question[:50])

    # ── Taker fallback ────────────────────────────────────────────────────────
    try:
        ok = await loop.run_in_executor(None, _place_order_sync, token_id, ask, size)
        return {"ask": ask, "bet": bet, "fill_type": "TAKER",
                "direction": direction.value, "is_reentry": reentry_info is not None} if ok else None
    except Exception as e:
        log.error("Order placement failed: %s", e)
        return None


# ── Execution gate ────────────────────────────────────────────────────────────

def should_execute(pick, active_model_count: int) -> bool:
    if pick.score < EXEC_MIN_SCORE:
        return False
    if EXEC_MIN_UNANIMOUS:
        dissent = pick.yes_votes if pick.direction.value == "NO" else pick.no_votes
        if dissent > 0:
            return False
        agreeing = pick.yes_votes + pick.no_votes
        if agreeing < max(2, active_model_count // 2):
            return False
    return True
