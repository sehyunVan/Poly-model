"""
src/crypto/settle_arb.py — Polymarket 5m settlement arbitrage.

Concept:
  The main flow loop enters at 200-270s into the 5m window.
  This module acts at 272-290s — AFTER the main loop's entry window closes.

  At that point, Chainlink has been polling for ~4m45s since window open.
  If Chainlink drift from open is clear (≥0.10%), the resolution is nearly
  certain. But the CLOB sometimes still prices the winning token at 0.75-0.90
  (market hasn't fully repriced). Buying at 0.85 and winning $1.00 = $0.13
  net of fee in under 30 seconds.

  This is not prediction — it's reading a near-deterministic oracle vs a
  lagging market price.

Edge conditions:
  - Chainlink drift ≥ MIN_DRIFT from window open
  - CLOB winning-token ask < MAX_ASK (there's still gap to exploit)
  - Net payout after 2% taker fee ≥ MIN_PAYOUT

What this is NOT:
  - Not competing with the main flow loop (different time window)
  - Not using flow signal scores (purely oracle math)
  - Not a large position (small validation bets only)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("crypto.settle_arb")

GAMMA_BASE    = "https://gamma-api.polymarket.com"
CLOB_BASE     = "https://clob.polymarket.com"

# Per-timeframe settings:
#   window_secs    — window length
#   start_elapsed  — earliest elapsed-seconds at which we'll act (final stretch)
#   end_elapsed    — latest elapsed at which we'll act (must finish before close)
#   min_drift_def  — default drift threshold (config can override per tf)
#
# Math justification (annualized BTC vol ≈ 60%):
#   σ_remaining = 0.60 × √(secs_left / 31_536_000)
#   5m:  at elapsed 285-295 → 5-15s left → noise ≈ 0.024-0.041% → 0.05% drift = 1.2-2× noise
#   15m: at elapsed 855-885 → 15-45s left → noise ≈ 0.041-0.072% → 0.10% drift = 1.4-2.4× noise
#   1h:  at elapsed 3420-3540 → 60-180s left → noise ≈ 0.083-0.143% → 0.20% drift = 1.4-2.4× noise
_TIMEFRAMES = {
    "5m":  {"window_secs": 300,  "start_elapsed": 285,  "end_elapsed": 295,  "min_drift_def": 0.0005},
    "15m": {"window_secs": 900,  "start_elapsed": 855,  "end_elapsed": 885,  "min_drift_def": 0.0010},
    "1h":  {"window_secs": 3600, "start_elapsed": 3420, "end_elapsed": 3540, "min_drift_def": 0.0020},
}

_SYMBOLS = {
    "BTC": "btc",
    "ETH": "eth",
    "SOL": "sol",
}

_HTTP = requests.Session()
_HTTP.headers["User-Agent"] = "poly-settle-arb/1.0"


# ── Market discovery (mirrors loop.py logic) ──────────────────────────────────

def _window_starts_near_close(now_ts: int, window_secs: int, look_back: int = 3) -> list[int]:
    """Return window start timestamps for windows currently near close."""
    aligned = (now_ts // window_secs) * window_secs
    return [aligned - i * window_secs for i in range(look_back)]


def _fetch_market(slug: str) -> Optional[dict]:
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=8)
        r.raise_for_status()
        events = r.json()
        if not events:
            return None
        ev = events[0]
        markets = ev.get("markets", [])
        if not markets:
            return None
        m = markets[0]
        m["_event_title"] = ev.get("title", slug)
        return m
    except Exception:
        return None


def _parse_window(m: dict, now_ts: int, tf: str, window_secs: int) -> Optional[dict]:
    """Parse a Gamma market dict into a window descriptor."""
    try:
        end_str = m.get("endDate", "") or m.get("endDateIso", "")
        if not end_str:
            return None
        if end_str.endswith("Z"):
            end_dt = datetime.fromisoformat(end_str[:-1]).replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.fromisoformat(end_str).astimezone(timezone.utc)

        window_start_ts = int(end_dt.timestamp()) - window_secs
        elapsed = now_ts - window_start_ts
        secs_left = window_secs - elapsed

        if secs_left <= 0 or elapsed < 0:
            return None

        prices_raw    = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
        current_price = float(prices_raw[0])
        if not (0.05 < current_price < 0.95):
            return None

        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) < 2:
            return None

        slug  = m.get("slug", "").lower()
        title = m.get("_event_title", "").lower()
        if   "btc" in slug or "bitcoin" in title:
            symbol = "BTC"
        elif "eth" in slug or "ethereum" in title:
            symbol = "ETH"
        elif "sol" in slug or "solana" in title:
            symbol = "SOL"
        else:
            return None

        return {
            "market_id":       m.get("conditionId", m.get("id", "")),
            "symbol":          symbol,
            "timeframe":       tf,
            "window_secs":     window_secs,
            "up_token":        token_ids[0],
            "down_token":      token_ids[1],
            "window_start_ts": window_start_ts,
            "elapsed":         elapsed,
            "secs_left":       secs_left,
            "slug":            slug,
        }
    except Exception:
        return None


def _discover_windows(now_ts: int, timeframes: list[str]) -> list[dict]:
    """Find currently active windows across all configured timeframes."""
    found = []
    seen: set[str] = set()
    for tf in timeframes:
        if tf not in _TIMEFRAMES:
            continue
        ws = _TIMEFRAMES[tf]["window_secs"]
        # look_back=2 windows is enough — anything older has already closed
        for ts in _window_starts_near_close(now_ts, ws, look_back=2):
            for sym, prefix in _SYMBOLS.items():
                slug = f"{prefix}-updown-{tf}-{ts}"
                m = _fetch_market(slug)
                if m is None:
                    continue
                w = _parse_window(m, now_ts, tf, ws)
                if w is None or w["market_id"] in seen:
                    continue
                seen.add(w["market_id"])
                found.append(w)
    return found


# ── CLOB book fetch ───────────────────────────────────────────────────────────

def _get_best_ask(token_id: str) -> Optional[float]:
    """Fetch best ask from CLOB REST. Falls back gracefully."""
    try:
        r = _HTTP.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=6)
        r.raise_for_status()
        asks = r.json().get("asks", [])
        if not asks:
            return None
        return min(float(a["price"]) for a in asks)
    except Exception as e:
        log.debug("CLOB ask fetch failed for %s: %s", token_id[:12], e)
        return None


# ── Window state ──────────────────────────────────────────────────────────────

@dataclass
class _WindowState:
    market_id:        str
    slug:             str
    symbol:           str
    timeframe:        str
    window_secs:      int
    up_token:         str
    down_token:       str
    window_start_ts:  int
    chainlink_open:   float       # Chainlink price when first seen
    first_seen_ts:    int
    traded:           bool = False
    skip_logged:      bool = False


# ── Main class ────────────────────────────────────────────────────────────────

class SettlementArb:
    """
    Standalone settlement arb checker.
    Call check() every loop cycle. Executes at most once per window.
    """

    def __init__(self, rtds_feed, backend, cfg: dict):
        """
        rtds_feed : RTDSFeed instance (from crypto.rtds_feed)
        backend   : ExecutionBackend (LiveBackend or VirtualBackend)
        cfg       : dict with settle_arb_* keys (from crypto_params.yaml)
        """
        self._rtds    = rtds_feed
        self._backend = backend
        self._cfg     = cfg
        self._windows: dict[str, _WindowState] = {}

    # ── Config helpers ────────────────────────────────────────────────────────

    def _tf_cfg(self, tf: str) -> dict:
        """Per-timeframe action params. Yaml can override via:
           settle_arb_{tf}_start_elapsed, settle_arb_{tf}_end_elapsed, settle_arb_{tf}_min_drift.
           Legacy: settle_arb_start_elapsed etc. (no tf suffix) still applies to 5m."""
        base = _TIMEFRAMES[tf]
        # Legacy 5m keys (no tf suffix) for backwards compat
        if tf == "5m":
            start = int(self._cfg.get("settle_arb_start_elapsed", base["start_elapsed"]))
            end   = int(self._cfg.get("settle_arb_end_elapsed",   base["end_elapsed"]))
            drift = float(self._cfg.get("settle_arb_min_drift",   base["min_drift_def"]))
        else:
            start = int(self._cfg.get(f"settle_arb_{tf}_start_elapsed", base["start_elapsed"]))
            end   = int(self._cfg.get(f"settle_arb_{tf}_end_elapsed",   base["end_elapsed"]))
            drift = float(self._cfg.get(f"settle_arb_{tf}_min_drift",   base["min_drift_def"]))
        return {"start_elapsed": start, "end_elapsed": end, "min_drift": drift,
                "window_secs": base["window_secs"]}

    @property
    def _max_ask(self) -> float:
        return float(self._cfg.get("settle_arb_max_ask", 0.92))

    @property
    def _min_ask(self) -> float:
        # Sanity floor — refuse to buy below this. Protects against stale REST
        # book data or extreme illiquidity (a winning token quoted at <10¢ with
        # 30s to go usually means the book is broken, not that we found alpha).
        return float(self._cfg.get("settle_arb_min_ask", 0.10))

    @property
    def _min_payout(self) -> float:
        return float(self._cfg.get("settle_arb_min_payout", 0.06))

    @property
    def _bet_abs(self) -> float:
        return float(self._cfg.get("settle_arb_bet_abs", 3.0))

    @property
    def _fee_rate(self) -> float:
        return float(self._cfg.get("fee_rate", 0.02))

    @property
    def _active_symbols(self) -> list[str]:
        # Prefer settle_arb-specific list (set 2026-05-20 to BTC/ETH/SOL); fall
        # back to crypto loop's active_symbols (currently [BTC] due to crypto
        # loop's filter regime, but settle_arb has no such WR-driven restriction).
        return self._cfg.get("settle_arb_symbols",
                             self._cfg.get("active_symbols", ["BTC"]))

    @property
    def _active_timeframes(self) -> list[str]:
        """Timeframes to scan. Default: 5m only (backwards compat).
           Set settle_arb_timeframes: [5m, 15m, 1h] in yaml to expand."""
        return self._cfg.get("settle_arb_timeframes", ["5m"])

    # ── Cycle ─────────────────────────────────────────────────────────────────

    def check(self) -> None:
        """Run one cycle. Discover windows, record opens, act near close."""
        now_ts = int(time.time())
        windows = _discover_windows(now_ts, self._active_timeframes)

        # Prune stale states (window closed >60s ago, using its own window_secs)
        stale = [mid for mid, ws in self._windows.items()
                 if now_ts - ws.window_start_ts > ws.window_secs + 60]
        for mid in stale:
            del self._windows[mid]

        for w in windows:
            mid    = w["market_id"]
            symbol = w["symbol"]
            tf     = w["timeframe"]

            if symbol not in self._active_symbols:
                continue

            tf_cfg  = self._tf_cfg(tf)
            elapsed = w["elapsed"]

            # ── First sight: record Chainlink open price ───────────────────
            if mid not in self._windows:
                _, cl_price = self._rtds.get_prices(symbol)
                if cl_price is None or cl_price == 0:
                    log.debug("SETTLE_ARB: Chainlink unavailable for %s, skipping window", symbol)
                    continue
                self._windows[mid] = _WindowState(
                    market_id=mid,
                    slug=w["slug"],
                    symbol=symbol,
                    timeframe=tf,
                    window_secs=w["window_secs"],
                    up_token=w["up_token"],
                    down_token=w["down_token"],
                    window_start_ts=w["window_start_ts"],
                    chainlink_open=cl_price,
                    first_seen_ts=now_ts,
                )
                log.debug("SETTLE_ARB: registered %s %s %s  cl_open=%.2f  elapsed=%.0fs",
                          symbol, tf, w["slug"][-16:], cl_price, elapsed)
                continue

            ws = self._windows[mid]

            # ── Not in action window yet ───────────────────────────────────
            if not (tf_cfg["start_elapsed"] <= elapsed <= tf_cfg["end_elapsed"]):
                continue

            if ws.traded:
                continue

            # ── Read current Chainlink price ───────────────────────────────
            _, cl_now = self._rtds.get_prices(symbol)
            if cl_now is None or cl_now == 0:
                log.debug("SETTLE_ARB: Chainlink stale at action time for %s", symbol)
                continue

            drift = (cl_now - ws.chainlink_open) / ws.chainlink_open

            if abs(drift) < tf_cfg["min_drift"]:
                if not ws.skip_logged:
                    log.info(
                        "SETTLE_ARB skip %s %s %s  drift=%.4f%% < min=%.4f%%  elapsed=%.0fs",
                        symbol, tf, ws.slug[-16:], drift * 100, tf_cfg["min_drift"] * 100, elapsed,
                    )
                    ws.skip_logged = True
                continue

            # ── Determine winning token ────────────────────────────────────
            direction = "UP" if drift > 0 else "DOWN"
            token_id  = ws.up_token if drift > 0 else ws.down_token

            # ── Fetch CLOB ask ─────────────────────────────────────────────
            ask = _get_best_ask(token_id)
            if ask is None:
                log.debug("SETTLE_ARB: no ask for %s %s", symbol, direction)
                continue

            if ask >= self._max_ask:
                log.info(
                    "SETTLE_ARB skip %s %s %s  ask=%.3f >= max=%.3f  drift=%.4f%%  elapsed=%.0fs",
                    symbol, tf, ws.slug[-16:], ask, self._max_ask, drift * 100, elapsed,
                )
                ws.skip_logged = True
                continue

            if ask < self._min_ask:
                # Likely a stale book or extreme illiquidity. A winning token quoted
                # below the floor usually means the CLOB REST returned dust asks —
                # if we filled at this price the order might also be stale.
                log.info(
                    "SETTLE_ARB skip %s %s %s  ask=%.3f < min=%.3f (likely stale/illiquid)  drift=%.4f%%  elapsed=%.0fs",
                    symbol, tf, ws.slug[-16:], ask, self._min_ask, drift * 100, elapsed,
                )
                ws.skip_logged = True
                continue

            payout = (1.0 - ask) - self._fee_rate
            if payout < self._min_payout:
                log.info(
                    "SETTLE_ARB skip %s %s %s  payout=%.3f < min=%.3f  ask=%.3f  elapsed=%.0fs",
                    symbol, tf, ws.slug[-16:], payout, self._min_payout, ask, elapsed,
                )
                ws.skip_logged = True
                continue

            bet = self._bet_abs
            log.info(
                "SETTLE_ARB SIGNAL  %s %s %s  direction=%s  drift=%.4f%%  "
                "cl_open=%.2f cl_now=%.2f  ask=%.3f  payout=%.3f  bet=$%.2f  elapsed=%.0fs",
                symbol, tf, ws.slug[-16:], direction, drift * 100,
                ws.chainlink_open, cl_now, ask, payout, bet, elapsed,
            )

            # ── Execute ────────────────────────────────────────────────────
            try:
                result = self._backend.place_order(
                    token_id=token_id,
                    price=ask,
                    size_usdc=bet,
                )
                if result:
                    ws.traded = True
                    log.info(
                        "SETTLE_ARB FILLED  %s %s %s %s @ %.3f  bet=$%.2f  payout≈$%.2f",
                        symbol, tf, ws.slug[-16:], direction, ask, bet, bet * payout,
                    )
                else:
                    log.warning("SETTLE_ARB: place_order returned None for %s %s %s",
                                symbol, tf, ws.slug[-16:])
            except Exception as exc:
                log.error("SETTLE_ARB execute error: %s", exc, exc_info=True)
