"""
crypto/clob_feed.py — Polymarket CLOB WebSocket feed.

Maintains a live orderbook cache per token_id, replacing per-cycle REST calls to
https://clob.polymarket.com/book in flow.py and loop.py.

Architecture mirrors BinanceLiveFeed in price_feed.py:
  - Background daemon thread with its own asyncio event loop
  - Thread-safe reads via Lock
  - Auto-reconnect on disconnect (3s backoff)
  - Heartbeat PING every 10s (server requirement)
  - subscribe() is safe to call before or after start()
  - Graceful fallback: returns ([], []) when stale — callers fall back to REST

Integration points:
  - flow.py::_cross_imbalance()  → replaces 2 REST /book fetches per signal
  - loop.py::_get_clob_spread()  → replaces 1 REST /book fetch before order entry
  Both save ~200-400ms per cycle and give sub-second book freshness.

WebSocket endpoints (Polymarket docs):
  Market channel: wss://ws-subscriptions-clob.polymarket.com/ws/market
  Subscription:   {"assets_ids": [...], "type": "market"}
  Heartbeat:      send "PING" every 10s; server replies "PONG"
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

_log = logging.getLogger("crypto.clob_feed")

CLOB_WS_URL      = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BOOK_STALE_SEC   = 5.0    # seconds before cached book is considered stale
PING_INTERVAL    = 10.0   # Polymarket requires PING every 10s
TRADE_WINDOW_SEC = 300.0  # rolling window for CLOB CVD accumulation


class CLOBFeed:
    """
    Persistent WebSocket connection to the Polymarket CLOB market channel.

    Subscribes to token_id streams and maintains a live book (bids + asks)
    per token. The book is incrementally updated via price_change events.
    """

    def __init__(self) -> None:
        # token_id → {"bids": {price_str: size_float}, "asks": {...}, "ts": float}
        self._books:         dict[str, dict]                      = {}
        self._lock           = threading.Lock()
        self._pending:       set[str]                             = set()
        self._subscribed:    set[str]                             = set()
        self._loop:          Optional[asyncio.AbstractEventLoop]  = None
        self._thread:        Optional[threading.Thread]           = None
        self._sub_event:     Optional[asyncio.Event]              = None
        self._running        = False
        # token_id → list of (timestamp_float, notional_float, is_buy_bool)
        self._trade_history: dict[str, list]                      = {}
        # token_id → list of (timestamp_float, price_float) for tick velocity
        self._price_history: dict[str, list]                      = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def subscribe(self, token_ids: list[str]) -> None:
        """
        Register token_ids for WS subscription.
        Safe to call before start() — tokens are queued and sent on connect.
        Safe to call after start() — tokens are sent to the live connection.
        """
        new = set(token_ids) - self._subscribed
        if not new:
            return
        with self._lock:
            self._pending.update(new)
        if self._running and self._loop is not None and self._sub_event is not None:
            self._loop.call_soon_threadsafe(self._sub_event.set)

    def start(self) -> None:
        """Start the background WebSocket thread. Safe to call multiple times."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="CLOBFeed"
        )
        self._thread.start()
        _log.info("CLOBFeed started (url=%s)", CLOB_WS_URL)

    def stop(self) -> None:
        self._running = False

    def get_book(self, token_id: str) -> tuple[list[dict], list[dict]]:
        """
        Return (bids, asks) as lists of {"price": str, "size": str} dicts.
        Bids sorted descending; asks ascending. Matches _fetch_book() REST format.
        Returns ([], []) when stale or unavailable — caller falls back to REST.
        """
        with self._lock:
            entry = self._books.get(token_id)
        if entry is None or (time.time() - entry["ts"]) > BOOK_STALE_SEC:
            return [], []
        bids = sorted(
            [{"price": p, "size": str(s)} for p, s in entry["bids"].items() if s > 0],
            key=lambda x: float(x["price"]),
            reverse=True,
        )
        asks = sorted(
            [{"price": p, "size": str(s)} for p, s in entry["asks"].items() if s > 0],
            key=lambda x: float(x["price"]),
        )
        return bids, asks

    def get_fill_price(self, token_id: str) -> tuple[Optional[float], Optional[float]]:
        """
        Return (min_ask, bid_ask_spread). Matches _get_clob_spread() semantics.
        Returns (None, None) when stale or no asks in book.
        """
        bids, asks = self.get_book(token_id)
        if not asks:
            return None, None
        fill = min(float(a["price"]) for a in asks)
        spd  = (fill - max(float(b["price"]) for b in bids)) if bids else None
        return fill, spd

    def is_live(self, token_id: str) -> bool:
        """True if we have a fresh book for this token."""
        with self._lock:
            entry = self._books.get(token_id)
        return entry is not None and (time.time() - entry["ts"]) <= BOOK_STALE_SEC

    def seed_book(self, token_id: str,
                  bids: list[dict], asks: list[dict]) -> None:
        """
        Seed the cache from a REST book response so price_change events can apply.

        Polymarket WS sends only incremental price_change deltas — no book snapshot
        on subscription. _handle_event drops price_change events when _books[token]
        is absent (entry is None). This seeds the entry so deltas apply immediately.

        Called by flow.py and loop.py after each REST fallback fetch.
        Only writes when the WS book is absent or stale; never overwrites a fresh
        WS book with potentially-lagged REST data.
        """
        bids_d = {b["price"]: float(b["size"]) for b in bids}
        asks_d = {a["price"]: float(a["size"]) for a in asks}
        with self._lock:
            entry = self._books.get(token_id)
            if entry is None or (time.time() - entry["ts"]) > BOOK_STALE_SEC:
                self._books[token_id] = {
                    "bids": bids_d, "asks": asks_d, "ts": time.time()
                }

    def record_price(self, token_id: str, price: float) -> None:
        """Record a Gamma consensus price for this token (used for tick velocity)."""
        now = time.time()
        with self._lock:
            hist = self._price_history.setdefault(token_id, [])
            hist.append((now, price))
            cutoff = now - 120.0
            while hist and hist[0][0] < cutoff:
                hist.pop(0)

    def get_tick_velocity(self, token_id: str, window_seconds: float = 60.0,
                          scale: float = 0.015) -> Optional[float]:
        """
        Price velocity of this token over the last window_seconds, normalised to [-1, +1].
        1.5% move in 60s maps to ±1.0.  Returns None if fewer than 5 ticks in window.
        Positive = token price rising (crowd buying UP); negative = price falling.
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            hist = self._price_history.get(token_id, [])
            recent = [(ts, px) for ts, px in hist if ts >= cutoff]
        if len(recent) < 5:
            return None
        oldest_px = recent[0][1]
        latest_px = recent[-1][1]
        if oldest_px <= 0:
            return None
        ret = (latest_px - oldest_px) / oldest_px
        return max(-1.0, min(1.0, ret / scale))

    def get_clob_cvd_score(self, token_id: str,
                            window_sec: float = TRADE_WINDOW_SEC) -> Optional[float]:
        """
        Normalised CLOB CVD for token_id in [-1, +1].
        CVD = (buy_notional - sell_notional) / (buy_notional + sell_notional).
        Inferred from price_change ask-size shrinks (taker buys) and bid-size shrinks
        (taker sells) accumulated over the rolling window_sec window.
        Returns None when total notional < $5 (insufficient data).
        Positive = net aggressive buying; negative = net aggressive selling.
        """
        cutoff = time.time() - window_sec
        with self._lock:
            hist     = self._trade_history.get(token_id, [])
            buy_vol  = sum(n for ts, n, b in hist if b     and ts >= cutoff)
            sell_vol = sum(n for ts, n, b in hist if not b and ts >= cutoff)
        total = buy_vol + sell_vol
        if total < 5.0:
            return None
        return (buy_vol - sell_vol) / total

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        finally:
            self._loop.close()

    async def _connect_loop(self) -> None:
        import websockets
        self._sub_event = asyncio.Event()
        while self._running:
            try:
                _log.info("CLOBFeed: connecting...")
                async with websockets.connect(
                    CLOB_WS_URL,
                    # Disable websockets library's own ping — we handle it manually
                    # because Polymarket expects plain-text "PING", not WS protocol pings.
                    ping_interval=None,
                    open_timeout=15,
                ) as ws:
                    _log.info("CLOBFeed: connected")
                    # Re-queue all previously subscribed tokens so they are
                    # re-sent on reconnect (server forgets subscriptions on disconnect).
                    with self._lock:
                        self._pending.update(self._subscribed)
                        self._subscribed.clear()
                    await self._flush_pending(ws)
                    await asyncio.gather(
                        self._recv_loop(ws),
                        self._sub_watcher(ws),
                        self._heartbeat(ws),
                        return_exceptions=True,
                    )
            except Exception as exc:
                if self._running:
                    _log.warning("CLOBFeed: disconnected (%s) — retry in 3s", exc)
                await asyncio.sleep(3)

    async def _heartbeat(self, ws) -> None:
        """Send PING every PING_INTERVAL seconds. Polymarket closes idle connections."""
        while self._running:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _flush_pending(self, ws) -> None:
        """Send subscription message for all pending tokens."""
        with self._lock:
            to_sub = list(self._pending)
            self._pending.clear()
        if not to_sub:
            return
        self._subscribed.update(to_sub)
        await ws.send(json.dumps({
            "assets_ids": to_sub,
            "type":       "market",
        }))
        _log.info("CLOBFeed: subscribed to %d token(s)", len(to_sub))

    async def _sub_watcher(self, ws) -> None:
        """Picks up tokens registered via subscribe() while already connected."""
        while self._running:
            await self._sub_event.wait()
            self._sub_event.clear()
            await self._flush_pending(ws)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if not self._running:
                break
            if raw == "PONG":
                continue
            self._handle(raw)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _handle(self, raw: str) -> None:
        try:
            events = json.loads(raw)
            if isinstance(events, dict):
                events = [events]
            for e in events:
                self._handle_event(e)
        except Exception:
            pass

    def _handle_event(self, e: dict) -> None:
        etype = e.get("event_type")
        token = e.get("asset_id")
        if not token:
            return
        now = time.time()

        if etype == "book":
            bids = {b["price"]: float(b["size"]) for b in e.get("bids", [])}
            asks = {a["price"]: float(a["size"]) for a in e.get("asks", [])}
            with self._lock:
                self._books[token] = {"bids": bids, "asks": asks, "ts": now}
            _log.debug("CLOBFeed snapshot: %s…  bids=%d asks=%d",
                       token[:12], len(bids), len(asks))

        elif etype == "price_change":
            with self._lock:
                entry = self._books.get(token)
                if entry is None:
                    return
                hist = self._trade_history.setdefault(token, [])
                for change in e.get("changes", []):
                    price    = change.get("price", "")
                    size     = float(change.get("size", 0))
                    side     = change.get("side", "").upper()
                    is_bid   = side in ("BUY", "BID")
                    book     = entry["bids"] if is_bid else entry["asks"]
                    old_size = book.get(price, 0.0)
                    # Only record complete level clears (size → 0) as inferred fills.
                    # Partial size decreases are ambiguous (could be MM partial cancel).
                    # Level-clearing events are almost always real taker aggression.
                    if size == 0 and old_size > 0 and price:
                        hist.append((now, old_size * float(price), not is_bid))
                    if size == 0:
                        book.pop(price, None)
                    else:
                        book[price] = size
                # Trim history older than the CVD window
                cutoff = now - TRADE_WINDOW_SEC
                while hist and hist[0][0] < cutoff:
                    hist.pop(0)
                entry["ts"] = now


# Module-level singleton — started once in loop.py alongside BinanceLiveFeed.
clob_feed = CLOBFeed()
