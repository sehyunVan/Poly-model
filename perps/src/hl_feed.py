"""
perps/src/hl_feed.py — Hyperliquid market-data feed (raw WebSocket).

Mirrors `src/crypto/clob_feed.py` (proven pattern: daemon thread, asyncio loop,
auto-reconnect with re-subscribe, threading.Lock for thread-safe reads).

Previous version used `hyperliquid-python-sdk`'s WebsocketManager, which silently
exits on WS disconnect (its run_forever() has no reconnect logic). That cost us
35h of data in the first Phase-0 deployment. This version owns the WS connection
directly and re-subscribes all coins on every reconnect.

WebSocket API:
  URL          wss://api.hyperliquid.xyz/ws
  Subscribe    {"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}}
  Ping         {"method": "ping"}  → server replies channel="pong"
  L2 update    {"channel": "l2Book", "data": {"coin": ..., "levels": [[bids],[asks]], "time": ...}}
  Trade        {"channel": "trades", "data": [{"coin", "side": "A"|"B", "px", "sz", ...}, ...]}
  Ctx update   {"channel": "activeAssetCtx", "data": {"coin": ..., "ctx": {...}}}

Side semantics on HL trades:
  "A" = aggressor sold (taker hit a bid) → SELL pressure
  "B" = aggressor bought (taker lifted an offer) → BUY pressure

Public API (unchanged from previous version):
    feed = HLFeed(); feed.subscribe_coin("BTC")
    feed.get_book("BTC") -> (bids, asks)   ([], []) when stale (>5s)
    feed.get_mid("BTC")  -> float | None
    feed.get_cvd_score("BTC") -> float | None
    feed.get_oracle_lag_score("BTC") -> float | None
    feed.get_funding_rate("BTC") -> float | None
    feed.get_mid_history("BTC", window_sec=120) -> [(ts, mid), ...]
    feed.is_live("BTC") -> bool
    feed.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

_log = logging.getLogger("perps.hl_feed")

HL_WS_URL        = "wss://api.hyperliquid.xyz/ws"
BOOK_STALE_SEC   = 5.0
TRADE_WINDOW_SEC = 300.0
MID_HISTORY_SEC  = 120.0
PING_INTERVAL    = 30.0    # HL closes idle conns ~60s; ping at 30s
RECONNECT_BACKOFF = 3.0
# If we receive no messages for this long while connected, assume the server
# has silently dropped our subscription and force-reconnect. Observed failure
# mode on 2026-05-28: connection stays "open" but stops delivering data.
WATCHDOG_IDLE_SEC = 30.0


class HLFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # coin → {"bids": [{px, sz}], "asks": [{px, sz}], "ts": float}
        self._books: dict[str, dict] = {}
        # coin → list of (ts, notional_usd, is_buy_bool)
        self._trades: dict[str, list] = {}
        # coin → list of (ts, mid_price)
        self._mids: dict[str, list] = {}
        # coin → most recent activeAssetCtx payload
        self._ctx: dict[str, dict] = {}
        # coins we want subscribed (re-sent on every reconnect)
        self._wanted: set[str] = set()
        # asyncio loop running in background thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._sub_event: Optional[asyncio.Event] = None
        self._running = False
        self._connected_at: float = 0.0   # for staleness alerting upstream
        self._last_msg_ts: float = 0.0    # any inbound message — for watchdog

    # ── Public API ──────────────────────────────────────────────────────────

    def subscribe_coin(self, coin: str) -> None:
        coin = coin.upper()
        with self._lock:
            new = coin not in self._wanted
            if new:
                self._wanted.add(coin)
        if not new:
            return
        if not self._running:
            self.start()
        # notify the loop to (re)subscribe
        if self._loop is not None and self._sub_event is not None:
            self._loop.call_soon_threadsafe(self._sub_event.set)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="HLFeed")
        self._thread.start()
        _log.info("HLFeed started (url=%s)", HL_WS_URL)

    def stop(self) -> None:
        self._running = False

    def get_book(self, coin: str) -> tuple[list[dict], list[dict]]:
        coin = coin.upper()
        with self._lock:
            entry = self._books.get(coin)
        if entry is None or (time.time() - entry["ts"]) > BOOK_STALE_SEC:
            return [], []
        return list(entry["bids"]), list(entry["asks"])

    def get_mid(self, coin: str) -> Optional[float]:
        bids, asks = self.get_book(coin)
        if not bids or not asks:
            return None
        return (float(bids[0]["px"]) + float(asks[0]["px"])) / 2.0

    def get_spread_bps(self, coin: str) -> Optional[float]:
        bids, asks = self.get_book(coin)
        if not bids or not asks:
            return None
        bb, ba = float(bids[0]["px"]), float(asks[0]["px"])
        if bb <= 0:
            return None
        return (ba - bb) / bb * 1e4

    def get_cvd_score(self, coin: str,
                       window_sec: float = TRADE_WINDOW_SEC) -> Optional[float]:
        coin = coin.upper()
        cutoff = time.time() - window_sec
        with self._lock:
            hist = list(self._trades.get(coin, []))
        buy = sum(n for ts, n, b in hist if b and ts >= cutoff)
        sell = sum(n for ts, n, b in hist if not b and ts >= cutoff)
        total = buy + sell
        if total < 1000.0:
            return None
        return _clamp((buy - sell) / total)

    def get_mid_history(self, coin: str,
                         window_sec: float = MID_HISTORY_SEC) -> list[tuple[float, float]]:
        coin = coin.upper()
        cutoff = time.time() - window_sec
        with self._lock:
            hist = list(self._mids.get(coin, []))
        return [(ts, m) for ts, m in hist if ts >= cutoff]

    def get_funding_rate(self, coin: str) -> Optional[float]:
        coin = coin.upper()
        with self._lock:
            ctx = self._ctx.get(coin)
        if not ctx:
            return None
        try:
            return float(ctx.get("ctx", {}).get("funding", 0.0))
        except Exception:
            return None

    def get_oracle_lag_score(self, coin: str, scale: float = 0.001) -> Optional[float]:
        coin = coin.upper()
        with self._lock:
            ctx = self._ctx.get(coin)
        if not ctx:
            return None
        try:
            inner = ctx.get("ctx", {})
            mark = float(inner.get("markPx", 0.0))
            oracle = float(inner.get("oraclePx", 0.0))
        except Exception:
            return None
        if oracle <= 0:
            return None
        return _clamp((mark - oracle) / oracle / scale)

    def is_live(self, coin: str) -> bool:
        coin = coin.upper()
        with self._lock:
            entry = self._books.get(coin)
        return entry is not None and (time.time() - entry["ts"]) <= BOOK_STALE_SEC

    # ── Background thread ──────────────────────────────────────────────────

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
                _log.info("HLFeed: connecting...")
                async with websockets.connect(
                    HL_WS_URL,
                    ping_interval=None,   # we send our own pings
                    open_timeout=15,
                ) as ws:
                    self._connected_at = time.time()
                    self._last_msg_ts = time.time()
                    _log.info("HLFeed: connected")
                    # Re-subscribe all wanted coins after every reconnect.
                    # HL server forgets subscriptions on disconnect.
                    await self._send_subscriptions(ws)
                    # Wait for whichever task ends first (any of them ending means
                    # connection is dead — fall through to reconnect).
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(self._recv_loop(ws)),
                            asyncio.create_task(self._sub_watcher(ws)),
                            asyncio.create_task(self._heartbeat(ws)),
                            asyncio.create_task(self._watchdog(ws)),
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    # close socket so the next iteration starts clean
                    try:
                        await ws.close()
                    except Exception:
                        pass
            except Exception as exc:
                if self._running:
                    _log.warning("HLFeed: disconnected (%s) — retry in %ds",
                                  exc, RECONNECT_BACKOFF)
                await asyncio.sleep(RECONNECT_BACKOFF)

    async def _heartbeat(self, ws) -> None:
        while self._running:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send(json.dumps({"method": "ping"}))
            except Exception as exc:
                _log.warning("HLFeed: heartbeat failed (%s) — triggering reconnect", exc)
                return   # exits gather → connect_loop reconnects

    async def _watchdog(self, ws) -> None:
        """If no message received in WATCHDOG_IDLE_SEC, server has silently
        dropped us — force reconnect by exiting (which triggers gather completion)."""
        while self._running:
            await asyncio.sleep(5)
            idle = time.time() - self._last_msg_ts
            if idle > WATCHDOG_IDLE_SEC:
                _log.warning(
                    "HLFeed: watchdog tripped (idle %.0fs > %.0fs) — forcing reconnect",
                    idle, WATCHDOG_IDLE_SEC,
                )
                return

    async def _send_subscriptions(self, ws) -> None:
        with self._lock:
            coins = list(self._wanted)
        if not coins:
            return
        for coin in coins:
            for sub_type in ("l2Book", "trades", "activeAssetCtx"):
                msg = {"method": "subscribe",
                       "subscription": {"type": sub_type, "coin": coin}}
                await ws.send(json.dumps(msg))
        _log.info("HLFeed: subscribed coins=%s (l2Book + trades + activeAssetCtx)", coins)

    async def _sub_watcher(self, ws) -> None:
        """Re-sends subscriptions when subscribe_coin() is called while connected."""
        while self._running:
            await self._sub_event.wait()
            self._sub_event.clear()
            await self._send_subscriptions(ws)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if not self._running:
                break
            self._last_msg_ts = time.time()
            self._handle(raw)

    # ── Event handlers ──────────────────────────────────────────────────────

    def _handle(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        ch = msg.get("channel")
        if ch == "pong" or ch == "subscriptionResponse":
            return
        if ch == "l2Book":
            self._on_l2book(msg.get("data", {}))
        elif ch == "trades":
            self._on_trades(msg.get("data", []))
        elif ch in ("activeAssetCtx", "activeSpotAssetCtx"):
            self._on_ctx(msg.get("data", {}))

    def _on_l2book(self, data: dict) -> None:
        try:
            coin = data.get("coin", "").upper()
            levels = data.get("levels", [[], []])
            bids_raw, asks_raw = levels[0], levels[1]
            bids = [{"px": l["px"], "sz": l["sz"]} for l in bids_raw]
            asks = [{"px": l["px"], "sz": l["sz"]} for l in asks_raw]
            now = time.time()
            with self._lock:
                self._books[coin] = {"bids": bids, "asks": asks, "ts": now}
                if bids and asks:
                    mid = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2.0
                    hist = self._mids.setdefault(coin, [])
                    hist.append((now, mid))
                    cutoff = now - MID_HISTORY_SEC
                    while hist and hist[0][0] < cutoff:
                        hist.pop(0)
        except Exception as exc:
            _log.debug("l2book parse fail: %s", exc)

    def _on_trades(self, trades: list) -> None:
        try:
            now = time.time()
            with self._lock:
                for t in trades:
                    coin = t.get("coin", "").upper()
                    px = float(t.get("px", 0.0))
                    sz = float(t.get("sz", 0.0))
                    side = t.get("side", "")  # "A"=taker sold, "B"=taker bought
                    is_buy = (side == "B")
                    notional = px * sz
                    hist = self._trades.setdefault(coin, [])
                    hist.append((now, notional, is_buy))
                    cutoff = now - TRADE_WINDOW_SEC
                    while hist and hist[0][0] < cutoff:
                        hist.pop(0)
        except Exception as exc:
            _log.debug("trades parse fail: %s", exc)

    def _on_ctx(self, data: dict) -> None:
        try:
            coin = data.get("coin", "").upper()
            with self._lock:
                self._ctx[coin] = data
        except Exception as exc:
            _log.debug("ctx parse fail: %s", exc)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
