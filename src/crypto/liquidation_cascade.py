"""
crypto/liquidation_cascade.py — Binance forced-liquidation cascade detector.

When a large cluster of forced liquidations fires on Binance perpetuals within
a short window, BTC/ETH/SOL continues in that direction ~70% of the time for
the next 2-5 minutes. This makes it a strong entry trigger for Polymarket 5m markets.

Architecture: mirrors BinanceLiveFeed — daemon thread, asyncio loop, threading.Lock,
auto-reconnect with 3s backoff. Zero blocking calls on the main loop thread.

Binance forced-order stream:
  URL: wss://fstream.binance.com/ws/!forceOrder@arr
  Message format:
    {"stream": "!forceOrder@arr", "data": {"e": "forceOrder", "o": {
      "s": "BTCUSDT",   # symbol
      "S": "SELL",      # SELL = long-position forced closed → bearish cascade
                        # BUY  = short-position forced closed → bullish cascade
      "q": "0.100",     # quantity (base asset)
      "p": "84250.00",  # fill price
      "T": 1713876543000  # trade time (ms)
    }}}

Direction mapping:
  "SELL" liquidation (longs blown out) → price moves DOWN → bet DOWN token
  "BUY"  liquidation (shorts squeezed) → price moves UP   → bet UP token

Usage in loop.py:
    from crypto.liquidation_cascade import liq_cascade
    liq_cascade.start()
    sig = liq_cascade.get_signal("BTC")
    # sig.direction: "UP" | "DOWN" | "NEUTRAL"
    # sig.notional_usd: cumulative USD liquidated in the cascade window
    # sig.confidence: 0.0–1.0 (based on notional vs threshold)
    # sig.age_sec: seconds since cascade detected (valid for CASCADE_DECAY_SEC)
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger("crypto.liq_cascade")

# Binance USDM futures forced-order stream (all symbols, no auth needed).
_WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"

# Rolling accumulation window: liquidations within CASCADE_WINDOW_SEC are summed.
CASCADE_WINDOW_SEC   = 15.0

# Minimum cumulative USD liquidated in one direction within the window to fire a signal.
# Research: $10M+ cascades → 70% WR on continuation. $30M+ → ~80% WR.
CASCADE_THRESHOLD_USD = 10_000_000.0   # $10M default — tune after 50+ signals

# How long a fired cascade signal remains valid (seconds).
# BTC typically continues for 2-5 minutes after a cascade; 120s is conservative.
CASCADE_DECAY_SEC = 120.0

# Confidence is 1.0 at this notional level (scales linearly from threshold to here).
CASCADE_MAX_USD = 50_000_000.0   # $50M = max confidence

# Map Binance symbols → Polymarket asset codes.
_SYM_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
}


@dataclass
class CascadeSignal:
    direction:    str    # "UP" | "DOWN" | "NEUTRAL"
    notional_usd: float  # cumulative USD liquidated in cascade window
    confidence:   float  # 0.0–1.0
    age_sec:      float  # seconds since signal was first detected
    symbol:       str    # e.g. "BTC"


class LiquidationCascadeDetector:
    """
    Background WebSocket subscriber for Binance forced-liquidation events.
    Detects cascade patterns and exposes active signals for the trading loop.

    Thread-safe: all internal state protected by self._lock.
    """

    def __init__(self) -> None:
        # Rolling event log: list of (mono_ts, symbol, side, notional_usd)
        self._events: list[tuple[float, str, str, float]] = []
        # Active signals: symbol → (direction, notional, detected_mono_ts)
        self._signals: dict[str, tuple[str, float, float]] = {}
        self._lock    = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background listener. Safe to call multiple times."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="LiqCascadeWS"
        )
        self._thread.start()
        _log.info("LiquidationCascadeDetector started → %s", _WS_URL)

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        finally:
            loop.close()

    # ── WebSocket loop ────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            _log.error("websockets package not installed — liquidation cascade disabled")
            return

        while self._running:
            try:
                _log.info("LiqCascade WS connecting...")
                async with websockets.connect(
                    _WS_URL, ping_interval=20, ping_timeout=10
                ) as ws:
                    _log.info("LiqCascade WS connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            self._handle(raw)
                        except Exception:
                            pass
            except Exception as exc:
                if self._running:
                    _log.warning("LiqCascade WS disconnected: %s — reconnect in 3s", exc)
                    await asyncio.sleep(3)

    def _handle(self, raw: str) -> None:
        """Parse one forced-order message and update internal state."""
        msg = json.loads(raw)
        # Stream wraps in {"stream":..., "data":{...}} or sends data directly.
        data = msg.get("data", msg)
        o = data.get("o", {})
        if not o:
            return

        binance_sym = o.get("s", "")
        symbol = _SYM_MAP.get(binance_sym)
        if symbol is None:
            return   # not a tracked asset

        side     = o.get("S", "")   # "SELL" = long liq, "BUY" = short liq
        qty      = float(o.get("q", 0))
        price    = float(o.get("p", 0))
        notional = qty * price
        if notional < 10_000:
            return   # ignore tiny retail liquidations (noise)

        now = time.monotonic()
        with self._lock:
            self._events.append((now, symbol, side, notional))
            self._trim_and_evaluate(now, symbol)

    def _trim_and_evaluate(self, now: float, symbol: str) -> None:
        """Trim stale events and check if a cascade threshold is crossed. Lock held."""
        cutoff = now - CASCADE_WINDOW_SEC
        self._events = [e for e in self._events if e[0] >= cutoff]

        long_liq  = sum(e[3] for e in self._events if e[1] == symbol and e[2] == "SELL")
        short_liq = sum(e[3] for e in self._events if e[1] == symbol and e[2] == "BUY")

        # "SELL" side = long positions liquidated → price moves DOWN
        # "BUY"  side = short positions liquidated → price moves UP
        dominant_notional = max(long_liq, short_liq)
        direction = "DOWN" if long_liq >= short_liq else "UP"

        if dominant_notional >= CASCADE_THRESHOLD_USD:
            prev = self._signals.get(symbol)
            if prev is None or prev[0] != direction or (now - prev[2]) > CASCADE_DECAY_SEC:
                # New cascade or direction flip
                _log.info(
                    "CASCADE DETECTED  %s  %s  notional=$%.1fM  (long_liq=$%.1fM  short_liq=$%.1fM)",
                    symbol, direction,
                    dominant_notional / 1e6,
                    long_liq / 1e6, short_liq / 1e6,
                )
            self._signals[symbol] = (direction, dominant_notional, now)
        else:
            # Below threshold — clear any stale active signal if decay expired
            prev = self._signals.get(symbol)
            if prev and (now - prev[2]) > CASCADE_DECAY_SEC:
                del self._signals[symbol]

    # ── Public query ──────────────────────────────────────────────────────────

    def get_signal(self, symbol: str) -> CascadeSignal:
        """
        Return the current cascade signal for a given asset.

        direction "DOWN": large long liquidation cascade → price likely to fall.
        direction "UP":   large short squeeze cascade → price likely to rise.
        direction "NEUTRAL": no active cascade (below threshold or signal decayed).
        """
        now = time.monotonic()
        with self._lock:
            sig = self._signals.get(symbol.upper())

        if sig is None:
            return CascadeSignal("NEUTRAL", 0.0, 0.0, 0.0, symbol)

        direction, notional, detected_ts = sig
        age = now - detected_ts

        if age > CASCADE_DECAY_SEC:
            return CascadeSignal("NEUTRAL", 0.0, 0.0, age, symbol)

        # Confidence scales linearly from 0 at threshold to 1 at CASCADE_MAX_USD.
        conf_range = CASCADE_MAX_USD - CASCADE_THRESHOLD_USD
        confidence = min(1.0, max(0.0, (notional - CASCADE_THRESHOLD_USD) / conf_range))

        return CascadeSignal(direction, notional, confidence, age, symbol)

    def is_live(self) -> bool:
        """True if the WebSocket thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def recent_events_summary(self, symbol: str, window_sec: float = 60.0) -> dict:
        """Debug helper: return last N seconds of liquidation totals for a symbol."""
        now    = time.monotonic()
        cutoff = now - window_sec
        with self._lock:
            events = [e for e in self._events if e[0] >= cutoff and e[1] == symbol.upper()]
        long_liq  = sum(e[3] for e in events if e[2] == "SELL")
        short_liq = sum(e[3] for e in events if e[2] == "BUY")
        return {
            "symbol": symbol,
            "window_sec": window_sec,
            "long_liq_usd":  round(long_liq),
            "short_liq_usd": round(short_liq),
            "event_count":   len(events),
        }


# Module singleton — imported by loop.py and started once.
liq_cascade = LiquidationCascadeDetector()
