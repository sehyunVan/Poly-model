"""
Chainlink oracle feed — reads BTC/ETH/SOL settlement oracle prices directly
from Polygon on-chain contracts, then compares with Binance spot to compute
the oracle lag signal.

Why on-chain instead of RTDS WebSocket:
  Oracle Cloud datacenter IPs are blocked by Cloudflare (same issue as Discord).
  The Chainlink contracts are accessible via standard Polygon RPC — no Cloudflare.
  This is also MORE accurate: the contract is literally what Polymarket settles with.

Oracle lag signal:
  lag = (binance_price - chainlink_price) / chainlink_price
  Positive → Binance already moved UP but Chainlink hasn't settled yet.
  Chainlink will catch up, so UP token is more likely to win.
  Negative → Binance dropped, Chainlink will follow → DOWN more likely.

Chainlink BTC/USD on Polygon: 0xc907E116054Ad103354f2D350FD2514433D57F6f
Chainlink ETH/USD on Polygon: 0xF9680D99D6C9589e2a93a78A04A279e509205945
Chainlink SOL/USD on Polygon: 0x10C8264C0935b3B9870013e057f330Ff3e9C56dC

Updates: every ~15s or when price moves ≥0.5% (heartbeat-based aggregators).

Usage:
    rtds_feed.start()
    score = rtds_feed.get_oracle_lag_score("BTC")  # [-1, +1]
    is_up = rtds_feed.is_live()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

_log = logging.getLogger("crypto.rtds_feed")

# Chainlink AggregatorV3 addresses on Polygon Mainnet
_CHAINLINK_ADDRS = {
    "BTC": "0xc907E116054Ad103354f2D350FD2514433D57F6f",
    "ETH": "0xF9680D99D6C9589e2a93a78A04A279e509205945",
    "SOL": "0x10C8264C0935b3B9870013e057f330Ff3e9C56dC",
}

# Minimal AggregatorV3Interface ABI — only latestRoundData needed
_AGG_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId",       "type": "uint80"},
            {"name": "answer",        "type": "int256"},
            {"name": "startedAt",     "type": "uint256"},
            {"name": "updatedAt",     "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Polygon RPC endpoints (same fallback list as redeem.py)
_POLYGON_RPCS = [
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon.llamarpc.com",
]

_BINANCE_SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
_SCALE = 0.001   # ±0.1% divergence → ±1.0 oracle lag score


class RTDSFeed:
    """
    Polls Chainlink oracle prices on Polygon every POLL_INTERVAL seconds.
    Compares with Binance live prices (from BinanceLiveFeed) to compute oracle lag.

    Public API (matches the original WS design so loop.py needs no changes):
        start()                    → begin background polling thread
        stop()                     → set _running = False
        get_oracle_lag_score(sym)  → Optional[float] in [-1, +1]
        get_prices(sym)            → (binance_price, chainlink_price) or (None, None)
        is_live()                  → True if last oracle poll succeeded
    """

    POLL_INTERVAL  = 15.0   # seconds between Chainlink polls
    STALE_SECONDS  = 60.0   # consider price stale if not updated in this long

    def __init__(self, symbols: list[str] | None = None):
        self._symbols = symbols or ["BTC", "ETH", "SOL"]
        self._cl_prices:  dict[str, float] = {}   # symbol → chainlink price
        self._cl_updated: dict[str, float] = {}   # symbol → monotonic ts of last poll
        self._lock     = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running  = False
        self._w3       = None   # web3 instance (lazy init on first poll)
        self._contracts: dict[str, object] = {}   # symbol → Contract
        self._decimals: dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="ChainlinkOracle"
        )
        self._thread.start()
        _log.info("ChainlinkOracleFeed started for %s", self._symbols)

    def stop(self) -> None:
        self._running = False

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _get_w3(self):
        """Return a working web3 instance, trying fallback RPCs."""
        try:
            from web3 import Web3
        except ImportError:
            _log.error("web3 not installed — Chainlink oracle feed disabled")
            return None
        for rpc in _POLYGON_RPCS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5}))
                if w3.is_connected():
                    _log.info("ChainlinkOracle: connected via %s", rpc)
                    return w3
            except Exception:
                continue
        _log.warning("ChainlinkOracle: all RPCs failed")
        return None

    def _get_contract(self, symbol: str, w3):
        if symbol in self._contracts:
            return self._contracts[symbol]
        addr = _CHAINLINK_ADDRS.get(symbol)
        if not addr:
            return None
        try:
            from web3 import Web3
            c = w3.eth.contract(
                address=Web3.to_checksum_address(addr),
                abi=_AGG_ABI,
            )
            self._contracts[symbol] = c
            self._decimals[symbol]  = c.functions.decimals().call()
            return c
        except Exception as exc:
            _log.debug("ChainlinkOracle: contract init failed for %s: %s", symbol, exc)
            return None

    def _poll_loop(self) -> None:
        """Poll all Chainlink oracles every POLL_INTERVAL seconds."""
        w3 = self._get_w3()
        if w3 is None:
            _log.warning("ChainlinkOracle: no RPC available — oracle lag signal disabled")
            return

        while self._running:
            try:
                for sym in self._symbols:
                    if sym not in _CHAINLINK_ADDRS:
                        continue
                    c = self._get_contract(sym, w3)
                    if c is None:
                        continue
                    try:
                        _, answer, _, updated_at, _ = c.functions.latestRoundData().call()
                        dec = self._decimals.get(sym, 8)
                        price = float(answer) / (10 ** dec)
                        mono  = time.monotonic()
                        with self._lock:
                            self._cl_prices[sym]  = price
                            self._cl_updated[sym] = mono
                        _log.debug("ChainlinkOracle %s: %.2f (on-chain ts=%d)", sym, price, updated_at)
                    except Exception as exc:
                        _log.debug("ChainlinkOracle poll failed for %s: %s", sym, exc)
                        # Try reconnecting
                        w3 = self._get_w3()
                        if w3 is None:
                            break
                        self._contracts.clear()  # force re-init contracts on new w3
            except Exception as exc:
                _log.debug("ChainlinkOracle loop error: %s", exc)

            # Sleep in small increments so stop() is responsive
            for _ in range(int(self.POLL_INTERVAL)):
                if not self._running:
                    return
                time.sleep(1)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_prices(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """Return (binance_price, chainlink_price) or (None, None)."""
        sym = symbol.upper()
        # Binance price from the module-level live_feed singleton
        try:
            from crypto.price_feed import live_feed as _live_feed
            bnc_sym = _BINANCE_SYM.get(sym)
            bnc = _live_feed.get_price(bnc_sym) if bnc_sym else None
        except Exception:
            bnc = None
        with self._lock:
            cl = self._cl_prices.get(sym)
        return bnc, cl

    def get_oracle_lag_score(self, symbol: str) -> Optional[float]:
        """
        Oracle lag signal in [-1, +1].

        lag = (binance_price - chainlink_price) / chainlink_price
        Normalised: ±0.1% (0.001) maps to ±1.0.

        Positive → Binance is above Chainlink oracle → predicts UP settlement.
        Negative → Binance fell below oracle → predicts DOWN settlement.
        Returns None when either price is unavailable or stale.
        """
        sym = symbol.upper()
        mono = time.monotonic()
        with self._lock:
            cl     = self._cl_prices.get(sym)
            cl_upd = self._cl_updated.get(sym, 0.0)
        if cl is None or cl == 0 or (mono - cl_upd) > self.STALE_SECONDS:
            return None
        try:
            from crypto.price_feed import live_feed as _live_feed
            bnc_sym = _BINANCE_SYM.get(sym)
            bnc = _live_feed.get_price(bnc_sym) if bnc_sym else None
        except Exception:
            bnc = None
        if bnc is None:
            return None
        lag = (bnc - cl) / cl
        return max(-1.0, min(1.0, lag / _SCALE))

    def is_live(self) -> bool:
        """True if at least one oracle price was fetched recently."""
        mono = time.monotonic()
        with self._lock:
            return any(
                (mono - upd) < self.STALE_SECONDS
                for upd in self._cl_updated.values()
            )


# Global singleton — started once by loop.py at startup.
rtds_feed = RTDSFeed(["BTC", "ETH", "SOL"])
