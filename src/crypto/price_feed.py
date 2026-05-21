"""
Binance REST API wrapper + WebSocket live price tracker for BTC/ETH.
No API key required for public market data endpoints.

WebSocket tracker (BinanceLiveFeed):
  - Connects to wss://stream.binance.com:9443/ws/<sym>@aggTrade
  - Runs in a background daemon thread
  - Exposes get_live_price(symbol) → latest trade price (<50ms latency)
  - Auto-reconnects on disconnect
  - Falls back to REST get_price() if WebSocket not started or stale
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx

_ws_log = logging.getLogger("crypto.price_feed.ws")

BINANCE_BASE = "https://api.binance.com/api/v3"
_CLIENT = httpx.Client(timeout=10.0)


@dataclass
class Candle:
    open_time: int      # ms epoch
    open: float
    high: float
    low: float
    close: float
    volume: float       # base asset volume (BTC)
    quote_volume: float # USDT volume


@dataclass
class OrderbookSnapshot:
    bids: list[tuple[float, float]]  # (price, qty) sorted desc
    asks: list[tuple[float, float]]  # (price, qty) sorted asc

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def imbalance(self, levels: int = 10) -> float:
        """Bid volume / (bid + ask volume) for top N levels. >0.5 = buy pressure."""
        bid_vol = sum(qty for _, qty in self.bids[:levels])
        ask_vol = sum(qty for _, qty in self.asks[:levels])
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5


def get_candles(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 60) -> list[Candle]:
    """Fetch recent 1-minute OHLCV candles from Binance."""
    resp = _CLIENT.get(
        f"{BINANCE_BASE}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    resp.raise_for_status()
    candles = []
    for row in resp.json():
        candles.append(Candle(
            open_time=row[0],
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
        ))
    return candles


def get_price(symbol: str = "BTCUSDT") -> float:
    """Current best price."""
    resp = _CLIENT.get(f"{BINANCE_BASE}/ticker/price", params={"symbol": symbol})
    resp.raise_for_status()
    return float(resp.json()["price"])


def get_orderbook(symbol: str = "BTCUSDT", limit: int = 20) -> OrderbookSnapshot:
    """Fetch top N levels of the Binance orderbook."""
    resp = _CLIENT.get(f"{BINANCE_BASE}/depth", params={"symbol": symbol, "limit": limit})
    resp.raise_for_status()
    data = resp.json()
    bids = [(float(p), float(q)) for p, q in data["bids"]]
    asks = [(float(p), float(q)) for p, q in data["asks"]]
    return OrderbookSnapshot(bids=bids, asks=asks)


def get_24h_stats(symbol: str = "BTCUSDT") -> dict:
    """24h ticker stats: priceChangePercent, volume, quoteVolume, etc."""
    resp = _CLIENT.get(f"{BINANCE_BASE}/ticker/24hr", params={"symbol": symbol})
    resp.raise_for_status()
    return resp.json()


_MLOFI_WEIGHTS = [1.0, 0.5, 0.25, 0.125, 0.0625]  # geometric decay across 5 depth levels

def get_mlofi(symbol: str = "BTCUSDT", levels: int = 5) -> Optional[float]:
    """
    Multi-Level Order Flow Imbalance from Binance spot depth, distance-weighted.

    Weights: [1, 0.5, 0.25, 0.125, 0.0625] for levels 1–5 (best-to-worst price).
    Returns (weighted_bid_vol - weighted_ask_vol) / (weighted_bid_vol + weighted_ask_vol)
    in [-1, +1]. Positive = buy-side pressure.  Returns None on fetch failure.

    Captures hidden depth that aggregated CVD misses — large resting bids/asks signal
    institutional intent before they execute.
    """
    try:
        resp = _CLIENT.get(
            f"{BINANCE_BASE}/depth",
            params={"symbol": symbol, "limit": max(levels, 5)},
            timeout=3.0,
        )
        resp.raise_for_status()
        data = resp.json()
        bids = [(float(p), float(q)) for p, q in data["bids"]]
        asks = [(float(p), float(q)) for p, q in data["asks"]]
        w_bid = sum(_MLOFI_WEIGHTS[i] * bids[i][1] for i in range(min(levels, len(bids))))
        w_ask = sum(_MLOFI_WEIGHTS[i] * asks[i][1] for i in range(min(levels, len(asks))))
        total = w_bid + w_ask
        if total < 1e-9:
            return None
        return max(-1.0, min(1.0, (w_bid - w_ask) / total))
    except Exception:
        return None


# ── Binance Futures public endpoints (no auth required) ───────────────────────

BINANCE_FAPI_BASE = "https://fapi.binance.com"


def get_futures_funding_rate(symbol: str = "BTCUSDT") -> float:
    """Current funding rate from Binance Futures. Positive = longs paying shorts."""
    resp = _CLIENT.get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex",
        params={"symbol": symbol},
    )
    resp.raise_for_status()
    return float(resp.json()["lastFundingRate"])


def get_futures_open_interest(symbol: str = "BTCUSDT") -> float:
    """Current open interest in base asset units from Binance Futures."""
    resp = _CLIENT.get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/openInterest",
        params={"symbol": symbol},
    )
    resp.raise_for_status()
    return float(resp.json()["openInterest"])


# ── Coinbase spot price ───────────────────────────────────────────────────────

COINBASE_BASE = "https://api.coinbase.com/v2/prices"


def get_coinbase_price(symbol: str = "BTC-USD") -> Optional[float]:
    """Current Coinbase spot price for a symbol (e.g. 'BTC-USD', 'SOL-USD').
    No auth required. Returns None on any network or parse error.
    """
    try:
        resp = _CLIENT.get(f"{COINBASE_BASE}/{symbol}/spot", timeout=3.0)
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])
    except Exception:
        return None


# ── WebSocket live price tracker ──────────────────────────────────────────────

class BinanceLiveFeed:
    """
    Persistent WebSocket connection to Binance aggTrade streams.
    Runs in a background daemon thread. Provides sub-50ms price updates.

    Usage:
        feed = BinanceLiveFeed(["BTCUSDT", "ETHUSDT"])
        feed.start()
        price = feed.get_price("BTCUSDT")   # latest trade price
        ret   = feed.get_return("BTCUSDT")  # return since feed started (or since reset)
        feed.set_reference("BTCUSDT", price) # set a new reference price for return calc
    """

    WS_BASE = "wss://stream.binance.com:9443/ws"
    STALE_SECONDS = 10.0   # consider feed stale if no update for this long

    # How many seconds of price/trade history to retain for leading-indicator features.
    _HISTORY_WINDOW = 120.0

    def __init__(self, symbols: list[str]):
        self._symbols   = [s.lower() for s in symbols]
        self._prices:   dict[str, float] = {}   # symbol → latest price
        self._refs:     dict[str, float] = {}   # symbol → reference price for return calc
        self._updated:  dict[str, float] = {}   # symbol → monotonic time of last update
        # CVD tracking — reset each time set_reference() is called (window open)
        self._cvd:      dict[str, float] = {}   # symbol → cumulative volume delta (base units)
        self._cvd_vol:  dict[str, float] = {}   # symbol → total volume since reset (for normalisation)
        # Leading-indicator: rolling price history [(mono_ts, price), ...]
        self._price_history: dict[str, list] = {}
        # Leading-indicator: rolling trade list [(mono_ts, notional_usd, is_buy), ...]
        self._trade_history: dict[str, list] = {}
        # Hawkes excitement: continuously decaying buy/sell intensity (not reset on window open)
        # More weight to recent trades vs. simple CVD cumulation.
        self._hawkes_buy:     dict[str, float] = {}   # symbol → decayed buy intensity
        self._hawkes_sell:    dict[str, float] = {}   # symbol → decayed sell intensity
        self._hawkes_last_ts: dict[str, float] = {}   # symbol → monotonic ts of last update
        self._lock      = threading.Lock()
        self._thread:   Optional[threading.Thread] = None
        self._running   = False

    def start(self) -> None:
        """Start the background WebSocket thread. Safe to call multiple times."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="BinanceWS")
        self._thread.start()
        _ws_log.info("BinanceLiveFeed started for %s", self._symbols)

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        """Background thread: run asyncio event loop for WebSocket."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        finally:
            loop.close()

    async def _ws_loop(self) -> None:
        """Connect to combined stream, reconnect on any error."""
        import websockets
        stream = "/".join(f"{s}@aggTrade" for s in self._symbols)
        url = f"{self.WS_BASE}/{stream}"
        while self._running:
            try:
                _ws_log.info("WS connecting: %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    _ws_log.info("WS connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            # aggTrade fields:
                            #   s = symbol, p = price, q = quantity (base asset)
                            #   m = isBuyerMaker: true → taker SOLD (aggressive sell)
                            #                    false → taker BOUGHT (aggressive buy)
                            sym   = msg.get("s", "").upper()
                            price = float(msg["p"])
                            qty   = float(msg.get("q", 0))
                            is_sell = bool(msg.get("m", False))
                            mono  = time.monotonic()
                            notional = price * qty  # USD notional of this trade
                            with self._lock:
                                self._prices[sym]  = price
                                self._updated[sym] = mono
                                # Set reference on first price received
                                if sym not in self._refs:
                                    self._refs[sym] = price
                                # Rolling price history (trim to _HISTORY_WINDOW)
                                hist = self._price_history.setdefault(sym, [])
                                hist.append((mono, price))
                                cutoff = mono - self._HISTORY_WINDOW
                                while hist and hist[0][0] < cutoff:
                                    hist.pop(0)
                                # Rolling trade history for whale detection
                                trades = self._trade_history.setdefault(sym, [])
                                trades.append((mono, notional, not is_sell))
                                while trades and trades[0][0] < cutoff:
                                    trades.pop(0)
                                # CVD: accumulate signed volume since last reset
                                delta = -qty if is_sell else qty
                                self._cvd[sym]     = self._cvd.get(sym, 0.0) + delta
                                self._cvd_vol[sym] = self._cvd_vol.get(sym, 0.0) + qty
                                # Hawkes: exponentially decaying buy/sell intensity
                                # decay_per_sec=0.95 → intensity halves in ~13s
                                _ht = self._hawkes_last_ts.get(sym)
                                if _ht is not None:
                                    _dt = mono - _ht
                                    self._hawkes_buy[sym]  = self._hawkes_buy.get(sym, 0.0)  * (0.95 ** _dt)
                                    self._hawkes_sell[sym] = self._hawkes_sell.get(sym, 0.0) * (0.95 ** _dt)
                                if is_sell:
                                    self._hawkes_sell[sym] = self._hawkes_sell.get(sym, 0.0) + qty
                                else:
                                    self._hawkes_buy[sym]  = self._hawkes_buy.get(sym, 0.0)  + qty
                                self._hawkes_last_ts[sym] = mono
                        except Exception:
                            pass
            except Exception as exc:
                if self._running:
                    _ws_log.warning("WS disconnected: %s — reconnecting in 2s", exc)
                    await asyncio.sleep(2)

    def get_price(self, symbol: str) -> Optional[float]:
        """Latest trade price. Returns None if no data or stale."""
        sym = symbol.upper()
        with self._lock:
            updated = self._updated.get(sym)
            if updated is None or (time.monotonic() - updated) > self.STALE_SECONDS:
                return None
            return self._prices.get(sym)

    def get_return(self, symbol: str) -> Optional[float]:
        """
        Return since set_reference() was last called.
        Positive = price went up. None if no data.
        """
        sym = symbol.upper()
        with self._lock:
            price = self._prices.get(sym)
            ref   = self._refs.get(sym)
            updated = self._updated.get(sym)
            if price is None or ref is None or ref == 0:
                return None
            if updated and (time.monotonic() - updated) > self.STALE_SECONDS:
                return None
            return (price - ref) / ref

    def set_reference(self, symbol: str, price: float) -> None:
        """Set the reference price for return calculation (e.g. at window open).
        Also resets CVD accumulation so CVD measures pressure since this window opened."""
        sym = symbol.upper()
        with self._lock:
            self._refs[sym]    = price
            self._cvd[sym]     = 0.0
            self._cvd_vol[sym] = 0.0

    def get_cvd_score(self, symbol: str) -> Optional[float]:
        """
        Normalised CVD score in [-1, +1].
        Positive = net aggressive buying since window open.
        Negative = net aggressive selling.
        Returns None if no volume accumulated yet.

        Formula: cvd / total_volume  (naturally bounded [-1, +1])
        """
        sym = symbol.upper()
        with self._lock:
            cvd = self._cvd.get(sym)
            vol = self._cvd_vol.get(sym)
        if cvd is None or vol is None or vol < 0.01:
            return None
        return max(-1.0, min(1.0, cvd / vol))

    def get_hawkes_ratio(self, symbol: str) -> Optional[float]:
        """
        Hawkes buy/sell excitement ratio in [-1, +1].
        Positive = recent aggressive buying decayed-weighted.
        Negative = recent aggressive selling decayed-weighted.
        Returns None if insufficient volume (< 0.01 BTC decayed total).

        Decay: 0.95^Δt per second → intensity halves in ~13s.
        """
        sym = symbol.upper()
        with self._lock:
            buy  = self._hawkes_buy.get(sym, 0.0)
            sell = self._hawkes_sell.get(sym, 0.0)
        total = buy + sell
        if total < 0.01:
            return None
        return max(-1.0, min(1.0, (buy - sell) / total))

    def get_recent_return(self, symbol: str, window_seconds: float = 30.0) -> Optional[float]:
        """
        Price return over the last `window_seconds` on Binance.
        Leading indicator: measures CURRENT momentum, not the full window return.
        Returns None if insufficient history.
        """
        sym = symbol.upper()
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            hist = self._price_history.get(sym, [])
            if not hist:
                return None
            latest_price = hist[-1][1]
            # Find oldest price within the window
            baseline = None
            for ts, px in hist:
                if ts >= cutoff:
                    baseline = px
                    break
        if baseline is None or baseline == 0:
            return None
        return (latest_price - baseline) / baseline

    def get_max_trade_notional(self, symbol: str, window_seconds: float = 30.0,
                               min_notional: float = 10_000.0
                               ) -> tuple[float, Optional[bool]]:
        """
        Largest single trade notional (USD) in the last `window_seconds`.
        Returns (max_notional, is_buy). Returns (0.0, None) if no trades above
        min_notional in window, or no data.
        Leading indicator: a whale hitting the book is predictive before market absorbs it.
        """
        sym = symbol.upper()
        now = time.monotonic()
        cutoff = now - window_seconds
        max_notional = 0.0
        whale_is_buy: Optional[bool] = None
        with self._lock:
            trades = self._trade_history.get(sym, [])
            for ts, notional, is_buy in trades:
                if ts >= cutoff and notional > max_notional:
                    max_notional = notional
                    whale_is_buy = is_buy
        if max_notional < min_notional:
            return 0.0, None
        return max_notional, whale_is_buy

    def get_trade_imbalance(self, symbol: str, window_seconds: float = 60.0) -> Optional[float]:
        """
        Buy/sell notional imbalance over the last window_seconds.
        Returns (buy_notional - sell_notional) / total in [-1, +1].
        Positive = aggressive buying; negative = aggressive selling.
        Returns None if total notional < $1,000 in window (insufficient data).
        """
        sym = symbol.upper()
        now = time.monotonic()
        cutoff = now - window_seconds
        buy_n = sell_n = 0.0
        with self._lock:
            trades = self._trade_history.get(sym, [])
            for ts, notional, is_buy in trades:
                if ts >= cutoff:
                    if is_buy:
                        buy_n += notional
                    else:
                        sell_n += notional
        total = buy_n + sell_n
        if total < 1_000.0:
            return None
        return (buy_n - sell_n) / total

    def is_live(self, symbol: str) -> bool:
        """True if we have a fresh price for this symbol."""
        return self.get_price(symbol.upper()) is not None


# Global singleton — started once by loop.py at startup.
# Import and call live_feed.start() to activate.
live_feed = BinanceLiveFeed(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
