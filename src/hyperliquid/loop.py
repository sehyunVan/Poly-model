"""
hyperliquid/loop.py — leveraged BTC/ETH perp trading on Hyperliquid.

Strategy:
  Uses the Polymarket 5-min BTC/ETH UP/DOWN crowd-flow signal as an entry
  trigger.  When the crowd's conviction crosses signal_threshold_for_hl (0.35),
  open a corresponding long or short perp position on Hyperliquid.

  Same signal that drives Polymarket bets — but larger position sizes through
  leverage and no $5 cap constraint.

  Runs every 20 seconds in its own screen session (screen -S hl).
  Completely independent of the Polymarket crypto loop.

Data flow:
  Gamma API / Polymarket CLOB → crypto.flow.compute_signal → FlowSignal
  Hyperliquid feed.py          → liq_score, funding_score   → FlowSignal (augmented)
  If |signal.score| > HL threshold → place_hl_order() → data/hl_state.json

Entry point: src/hl_main.py
Config:      config/hl_params.yaml
State:       data/hl_state.json  (live) | data/hl_virt_state.json (paper)
Env vars:    HL_ADDRESS, HL_KEY, HL_VIRTUAL_MODE
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import httpx

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv
for _p in [_SRC.parent, _SRC.parent / "polymarket-mcp-main" / "polymarket-mcp-main"]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT    = _SRC.parent
_HL_CONFIG_PATH = _ROOT / "config" / "hl_params.yaml"
_LOG_DIR = _ROOT / "logs"


def _load_hl_config() -> dict:
    """Load config/hl_params.yaml with hardcoded fallback defaults."""
    defaults = {
        "loop_interval":          20,
        "leverage":               3,
        "tp_pct":                 0.008,
        "sl_pct":                 0.004,
        "max_position_usdc":      50.0,
        "max_open_positions":     2,
        "max_hold_seconds":       240,
        "signal_threshold_for_hl": 0.35,
        "daily_loss_limit_pct":   0.20,
        "virtual_budget":         1000.0,
        "funding_fade_threshold": 0.0005,
        "liq_lookback_seconds":   60,
        "liq_min_notional":       10_000.0,
        "hl_poly_symbols":        ["BTC"],
        "hl_breakout_symbols":    [],
        "breakout_entry_threshold": 0.35,
    }
    try:
        import yaml
        with open(_HL_CONFIG_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return {**defaults, **loaded}
    except Exception:
        return defaults


_CFG = _load_hl_config()

LOOP_INTERVAL         = int(_CFG["loop_interval"])
LEVERAGE              = int(_CFG["leverage"])
TP_PCT                = float(_CFG["tp_pct"])
SL_PCT                = float(_CFG["sl_pct"])
MAX_POSITION_USDC     = float(_CFG["max_position_usdc"])
MAX_OPEN_POSITIONS    = int(_CFG["max_open_positions"])
MAX_HOLD_SECONDS      = int(_CFG["max_hold_seconds"])
SIGNAL_THRESHOLD_HL   = float(_CFG["signal_threshold_for_hl"])
DAILY_LOSS_LIMIT_PCT  = float(_CFG["daily_loss_limit_pct"])
VIRTUAL_BUDGET        = float(_CFG["virtual_budget"])
FUNDING_FADE_THRESH   = float(_CFG["funding_fade_threshold"])
LIQ_LOOKBACK          = int(_CFG["liq_lookback_seconds"])
LIQ_MIN_NOTIONAL      = float(_CFG["liq_min_notional"])
# Multi-asset: Poly-flow symbols and pure-breakout symbols (from hl_params.yaml)
HL_POLY_SYMBOLS       = list(_CFG.get("hl_poly_symbols", ["BTC"]))
HL_BREAKOUT_SYMBOLS   = list(_CFG.get("hl_breakout_symbols", []))
BREAKOUT_ENTRY_THRESH = float(_CFG.get("breakout_entry_threshold", 0.35))

# ── Virtual mode ──────────────────────────────────────────────────────────────
# HL_VIRTUAL_MODE is separate from VIRTUAL_MODE (Polymarket loop).
# Default: paper trading until you explicitly set HL_VIRTUAL_MODE=false.
HL_VIRTUAL_MODE = os.getenv("HL_VIRTUAL_MODE", "true").lower() not in ("false", "0", "no")

_STATE_FILE = _ROOT / "data" / ("hl_virt_state.json" if HL_VIRTUAL_MODE else "hl_state.json")

# Only import execution module in live mode (mirrors crypto loop pattern)
if not HL_VIRTUAL_MODE:
    try:
        from hyperliquid.execution import (      # type: ignore
            place_hl_order    as _place_hl_order,
            close_hl_position as _close_hl_position,
            get_hl_balance    as _get_hl_balance,
            get_open_position as _get_open_position,
        )
    except ImportError as _e:
        raise RuntimeError(
            f"HL_VIRTUAL_MODE=false but hyperliquid.execution failed to import: {_e}"
        ) from _e
else:
    _place_hl_order    = None  # type: ignore
    _close_hl_position = None  # type: ignore
    _get_hl_balance    = None  # type: ignore
    _get_open_position = None  # type: ignore

_HTTP = httpx.Client(timeout=8.0)

# ── Shutdown flag ─────────────────────────────────────────────────────────────
_shutdown = {"requested": False}
_balance_fail = {"until": 0.0}  # monotonic time — skip live order attempts until this clears


def _handle_sigterm(sig, frame):
    _shutdown["requested"] = True


# ── Position model ────────────────────────────────────────────────────────────

@dataclass
class HLPosition:
    id:             str
    coin:           str          # "BTC" or "ETH"
    side:           str          # "long" or "short"
    size_contracts: float
    size_usd:       float        # leveraged notional (margin × leverage)
    entry_price:    float
    entry_time:     str          # ISO format UTC
    tp_price:       float
    sl_price:       float
    tp_order_id:    Optional[int] = None
    sl_order_id:    Optional[int] = None
    realized_pnl:   Optional[float] = None
    closed:         bool = False
    close_reason:   Optional[str] = None
    margin_usdc:    float = 0.0  # actual USDC deducted from available balance

    def age_seconds(self) -> float:
        try:
            entry = datetime.fromisoformat(self.entry_time)
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - entry).total_seconds()
        except Exception:
            return 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        """Estimate unrealized PnL given current mid price."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.size_contracts
        else:
            return (self.entry_price - current_price) * self.size_contracts


# ── State persistence ─────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load HL loop state from JSON file. Returns fresh state if file missing."""
    if not _STATE_FILE.exists():
        today = date.today().isoformat()
        return {
            "positions":       [],
            "closed_positions": [],
            "available_usdc":  VIRTUAL_BUDGET if HL_VIRTUAL_MODE else 0.0,
            "daily_pnl":       0.0,
            "day_start_capital": VIRTUAL_BUDGET if HL_VIRTUAL_MODE else 0.0,
            "daily_halt":      False,
            "trade_date":      today,
            "last_updated":    datetime.now(timezone.utc).isoformat(),
        }
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _load_state.__wrapped__() if hasattr(_load_state, "__wrapped__") else {}


def _save_state(state: dict) -> None:
    """Atomically save state to JSON file."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(_STATE_FILE)
    except Exception as exc:
        logging.getLogger("hl.loop").warning("_save_state failed: %s", exc)


def _positions_from_state(state: dict) -> list[HLPosition]:
    """Deserialise open positions from state dict."""
    result = []
    for d in state.get("positions", []):
        try:
            result.append(HLPosition(**{k: d[k] for k in HLPosition.__dataclass_fields__}))
        except Exception:
            pass
    return result


def _state_with_positions(state: dict, positions: list[HLPosition],
                           closed: list[HLPosition]) -> dict:
    """Return updated state dict with serialised positions."""
    state["positions"]        = [asdict(p) for p in positions if not p.closed]
    state["closed_positions"] = (
        state.get("closed_positions", []) + [asdict(p) for p in closed]
    )
    return state


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "hl.log", maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if sys.stdout.isatty():
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger("hl.loop")


# ── Market discovery (shared with crypto loop) ────────────────────────────────
# Import private function directly — it's pure REST with no side effects.
# Coupling is intentional and documented here.

from crypto.loop import (        # type: ignore
    _discover_markets,
    _get_clob_spread,
    ACTIVE_SYMBOLS,
    MIN_WINDOW_ELAPSED,
    MAX_WINDOW_ELAPSED,
)
from crypto.flow      import compute_signal, SIGNAL_THRESHOLD  # type: ignore
from crypto.price_feed import live_feed as _live_feed          # type: ignore
from hyperliquid.feed import (                                  # type: ignore
    get_mid_price,
    get_liq_imbalance_score,
    get_funding_score,
)
from hyperliquid.breakout import compute_breakout_score         # type: ignore

# Breakout opposition threshold: if breakout score opposes signal by more than
# this value, skip the trade.  0.5 = only block on strong confirmed breakdown
# opposing a LONG (or vice versa).  Set to 1.0 to effectively disable.
_BREAKOUT_BLOCK_THRESHOLD = 0.5

# Chop filter threshold: skip any trade when |breakout_score| < this value.
# Near-zero breakout = BTC is range-bound on 15-min TF, no directional trend.
# Momentum trades in choppy markets consistently hit SL before TP.
# 0.20 = require at least mild directional conviction to enter.
_BREAKOUT_CHOP_THRESHOLD = 0.20


# ── Position management ────────────────────────────────────────────────────────

def _manage_positions(
    positions: list[HLPosition],
    state: dict,
    log: logging.Logger,
) -> tuple[list[HLPosition], list[HLPosition]]:
    """
    Check each open position for:
      1. Max hold time expiry → force-close.
      2. Virtual mode: TP/SL simulation via current HL mid price.
      3. Live mode: poll HL to detect TP/SL fills.

    Returns (still_open, newly_closed).
    """
    still_open:    list[HLPosition] = []
    newly_closed:  list[HLPosition] = []

    for pos in positions:
        if pos.closed:
            newly_closed.append(pos)
            continue

        age = pos.age_seconds()
        mid = get_mid_price(pos.coin)

        # ── Force-close at max_hold_time ──────────────────────────────────────
        if age >= MAX_HOLD_SECONDS:
            log.info(
                "FORCE_CLOSE %s %s  age=%.0fs >= max=%ds",
                pos.side.upper(), pos.coin, age, MAX_HOLD_SECONDS,
            )
            pnl = _close_position(pos, mid, "max_hold_time", log)
            state["daily_pnl"]    = state.get("daily_pnl", 0.0) + pnl
            state["available_usdc"] = state.get("available_usdc", 0.0) + (pos.margin_usdc + pnl)
            pos.realized_pnl  = pnl
            pos.closed        = True
            pos.close_reason  = "max_hold_time"
            newly_closed.append(pos)
            continue

        if mid is None:
            still_open.append(pos)
            continue

        if HL_VIRTUAL_MODE:
            # ── Virtual TP/SL check ───────────────────────────────────────────
            hit_tp = (pos.side == "long"  and mid >= pos.tp_price) or \
                     (pos.side == "short" and mid <= pos.tp_price)
            hit_sl = (pos.side == "long"  and mid <= pos.sl_price) or \
                     (pos.side == "short" and mid >= pos.sl_price)

            if hit_tp or hit_sl:
                reason = "tp_hit" if hit_tp else "sl_hit"
                close_price = pos.tp_price if hit_tp else pos.sl_price
                pnl = _calc_pnl(pos, close_price)
                log.info(
                    "%s  %s %s  entry=%.4f  exit=%.4f  pnl=%+.2f",
                    reason.upper(), pos.side.upper(), pos.coin,
                    pos.entry_price, close_price, pnl,
                )
                state["daily_pnl"]      = state.get("daily_pnl", 0.0) + pnl
                state["available_usdc"] = state.get("available_usdc", 0.0) + (pos.margin_usdc + pnl)
                pos.realized_pnl = pnl
                pos.closed       = True
                pos.close_reason = reason
                newly_closed.append(pos)
                continue
        else:
            # ── Live mode: check if position still exists on HL ───────────────
            live_pos = _get_open_position(pos.coin, log)
            if live_pos is None:
                # Position gone → TP or SL fired on exchange
                # Estimate PnL from current mid (or entry if mid unavailable)
                close_px = mid or pos.entry_price
                pnl      = _calc_pnl(pos, close_px)
                log.info(
                    "POSITION_CLOSED_BY_BRACKET %s %s  entry=%.4f  est_exit=%.4f  pnl=%+.2f",
                    pos.side.upper(), pos.coin, pos.entry_price, close_px, pnl,
                )
                state["daily_pnl"]      = state.get("daily_pnl", 0.0) + pnl
                state["available_usdc"] = state.get("available_usdc", 0.0) + (pos.margin_usdc + pnl)
                pos.realized_pnl = pnl
                pos.closed       = True
                pos.close_reason = "bracket_fired"
                newly_closed.append(pos)
                continue

        still_open.append(pos)

    return still_open, newly_closed


def _calc_pnl(pos: HLPosition, close_price: float) -> float:
    """Gross PnL for a perp position (no fee modelling for simplicity)."""
    if pos.side == "long":
        return round((close_price - pos.entry_price) * pos.size_contracts, 4)
    else:
        return round((pos.entry_price - close_price) * pos.size_contracts, 4)


def _close_position(
    pos: HLPosition,
    mid: Optional[float],
    reason: str,
    log: logging.Logger,
) -> float:
    """Close position: virtual uses mid estimate; live calls close_hl_position."""
    if HL_VIRTUAL_MODE:
        close_px = mid or pos.entry_price
        return _calc_pnl(pos, close_px)
    else:
        result   = _close_hl_position(
            pos.coin, pos.side, pos.size_contracts,
            pos.tp_order_id, pos.sl_order_id, log,
        )
        close_px = result.get("fill_price", mid or pos.entry_price)
        return _calc_pnl(pos, close_px)


# ── Daily reset ───────────────────────────────────────────────────────────────

def _maybe_reset_day(state: dict, log: logging.Logger) -> dict:
    """Reset daily PnL and circuit breaker at midnight UTC."""
    today = date.today().isoformat()
    if state.get("trade_date") != today:
        log.info(
            "New trading day %s — resetting daily PnL (prev=%.2f)",
            today, state.get("daily_pnl", 0.0),
        )
        state["daily_pnl"]       = 0.0
        state["day_start_capital"] = state.get("available_usdc", VIRTUAL_BUDGET)
        state["daily_halt"]      = False
        state["trade_date"]      = today
    return state


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(log: logging.Logger) -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)

    mode_str = "VIRTUAL" if HL_VIRTUAL_MODE else "LIVE"
    log.info(
        "HL loop starting  mode=%s  leverage=%dx  tp=%.1f%%  sl=%.1f%%  "
        "max_pos=$%.0f  threshold=%.2f",
        mode_str, LEVERAGE, TP_PCT * 100, SL_PCT * 100,
        MAX_POSITION_USDC, SIGNAL_THRESHOLD_HL,
    )

    state = _load_state()

    # ── C1: Balance sync for live mode ────────────────────────────────────────
    if not HL_VIRTUAL_MODE:
        bal = _get_hl_balance(log)
        if bal is None:
            log.critical("HL live mode: cannot fetch HL balance — refusing to start")
            sys.exit(1)
        state["available_usdc"]    = bal
        state["day_start_capital"] = bal
        log.info("HL balance synced: $%.2f", bal)
        _save_state(state)  # create hl_state.json immediately so dashboard shows live data

    # Start Binance WebSocket (singleton — safe to call multiple times)
    _live_feed.start()

    positions: list[HLPosition] = _positions_from_state(state)
    # Track which markets already have an HL position this cycle
    _traded_coins: set[str] = set()

    log.info("HL loop ready — entering main cycle")

    while not _shutdown["requested"]:
        cycle_start = time.monotonic()
        now         = datetime.now(timezone.utc)

        # ── Daily reset ───────────────────────────────────────────────────────
        state = _maybe_reset_day(state, log)

        # ── Manage open positions ─────────────────────────────────────────────
        positions, newly_closed = _manage_positions(positions, state, log)
        if newly_closed:
            state = _state_with_positions(state, positions, newly_closed)
            _save_state(state)

        # ── Daily circuit breaker (C2) ────────────────────────────────────────
        daily_pnl      = state.get("daily_pnl", 0.0)
        day_start_cap  = state.get("day_start_capital", VIRTUAL_BUDGET)
        if day_start_cap > 0 and (-daily_pnl / day_start_cap) >= DAILY_LOSS_LIMIT_PCT:
            if not state.get("daily_halt", False):
                log.warning(
                    "HL DAILY LOSS LIMIT  pnl=%.2f  cap=%.2f  limit=%.0f%%",
                    daily_pnl, day_start_cap, DAILY_LOSS_LIMIT_PCT * 100,
                )
                state["daily_halt"] = True
                _save_state(state)
            _sleep_to_next(cycle_start, log)
            continue

        # ── Skip if max positions already open ────────────────────────────────
        n_open = len(positions)
        if n_open >= MAX_OPEN_POSITIONS:
            _sleep_to_next(cycle_start, log)
            continue

        # ── Discover Polymarket markets ────────────────────────────────────────
        try:
            markets = _discover_markets(now)
        except Exception as exc:
            log.debug("_discover_markets failed: %s", exc)
            _sleep_to_next(cycle_start, log)
            continue

        for parsed in markets:
            if _shutdown["requested"]:
                break

            (market_id, symbol, up_token, down_token,
             current_price, window_start_ts, secs_left, end_dt) = parsed

            # Skip if symbol not in HL_POLY_SYMBOLS (BTC + ETH by default)
            if symbol not in HL_POLY_SYMBOLS:
                continue

            coin = symbol   # "BTC" or "ETH"

            # Skip if we already have an HL position in this coin this cycle
            if coin in _traded_coins:
                continue

            # Skip if already have an open HL position for this coin
            if any(p.coin == coin for p in positions):
                continue

            # Skip if outside entry window
            window_elapsed = 300 - secs_left
            if window_elapsed < MIN_WINDOW_ELAPSED or window_elapsed > MAX_WINDOW_ELAPSED:
                continue

            # Skip if not enough capital
            available = state.get("available_usdc", 0.0)
            if available < MAX_POSITION_USDC * 0.5:
                log.debug("Insufficient HL capital: $%.2f", available)
                continue

            # ── Fetch HL-specific signals ─────────────────────────────────────
            liq_score     = get_liq_imbalance_score(
                coin, lookback_seconds=LIQ_LOOKBACK, min_notional=LIQ_MIN_NOTIONAL,
            )
            funding_score = get_funding_score(coin, fade_threshold=FUNDING_FADE_THRESH)

            # ── Compute flow signal (same as crypto loop) ─────────────────────
            cvd_score  = _live_feed.get_cvd_score(f"{coin}USDT")
            macd_score = _get_macd_score(coin)

            try:
                sig = compute_signal(
                    up_token, down_token, current_price,
                    _get_window_open_price(window_start_ts, current_price, coin),
                    cvd_score=cvd_score,
                    macd_score=macd_score,
                    liq_score=liq_score,
                    funding_score=funding_score,
                )
            except Exception as exc:
                log.debug("compute_signal failed: %s", exc)
                continue

            # ── HL signal threshold (stricter than Polymarket's 0.25) ─────────
            if abs(sig.score) < SIGNAL_THRESHOLD_HL:
                continue

            direction = sig.direction   # "UP" or "DOWN"
            if direction == "NO_TRADE":
                continue

            hl_side = "long" if direction == "UP" else "short"

            # ── Breakout filter (moondev 15-min BB + Donchian scanner) ────────
            # Block trade if 15-min breakout strongly opposes signal direction.
            # Only fires on strong confirmed breakouts (score > 0.5 threshold).
            # Runs async on first call per 10-min window (cached thereafter).
            try:
                bk_score = compute_breakout_score(coin)
            except Exception as _bk_exc:
                log.debug("breakout_score(%s) failed: %s", coin, _bk_exc)
                bk_score = None

            signal_sign = 1 if hl_side == "long" else -1
            if (bk_score is not None
                    and bk_score * signal_sign < -_BREAKOUT_BLOCK_THRESHOLD):
                log.info(
                    "HL BREAKOUT BLOCK  %s %s  breakout=%.3f  signal=%.3f",
                    hl_side.upper(), coin, bk_score, sig.score,
                )
                continue

            # ── Chop filter: skip when 15-min has no directional conviction ───
            # If breakout score is near zero, BTC is range-bound on 15-min TF.
            # Momentum trades in choppy markets hit SL before TP consistently.
            # Threshold 0.20: require at least mild directional conviction.
            if bk_score is not None and abs(bk_score) < _BREAKOUT_CHOP_THRESHOLD:
                log.info(
                    "HL CHOP SKIP  %s %s  breakout=%.3f (< %.2f — no 15m conviction)",
                    hl_side.upper(), coin, bk_score, _BREAKOUT_CHOP_THRESHOLD,
                )
                continue

            # ── Get current HL mid price ──────────────────────────────────────
            hl_mid = get_mid_price(coin)
            if hl_mid is None:
                log.debug("HL: no mid price for %s — skip", coin)
                continue

            position_size = min(MAX_POSITION_USDC, available * 0.5)

            log.info(
                "HL SIGNAL  %s %s  score=%.3f  alpha=%.3f  liq=%.2f  "
                "funding=%.2f  breakout=%.2f  mid=%.2f  size=$%.0f",
                hl_side.upper(), coin, sig.score, sig.alpha,
                liq_score or 0.0, funding_score or 0.0,
                bk_score if bk_score is not None else 0.0,
                hl_mid, position_size,
            )

            # ── Place order ───────────────────────────────────────────────────
            if HL_VIRTUAL_MODE:
                result = _virtual_entry(coin, hl_side, position_size, hl_mid, log)
            else:
                result = _place_hl_order(
                    coin=coin,
                    side=hl_side,
                    size_usdc=position_size,
                    leverage=LEVERAGE,
                    tp_pct=TP_PCT,
                    sl_pct=SL_PCT,
                    log=log,
                    mid_price_hint=hl_mid,
                )

            if result["status"] not in ("FILLED", "VIRTUAL"):
                log.warning("HL order not filled: %s", result.get("reason", "unknown"))
                continue

            pos = HLPosition(
                id=str(uuid.uuid4())[:8],
                coin=coin,
                side=hl_side,
                size_contracts=result["size_contracts"],
                size_usd=result["size_usd"],
                entry_price=result["fill_price"],
                entry_time=datetime.now(timezone.utc).isoformat(),
                tp_price=result["tp_price"],
                sl_price=result["sl_price"],
                tp_order_id=result.get("tp_order_id"),
                sl_order_id=result.get("sl_order_id"),
                margin_usdc=position_size,
            )
            positions.append(pos)
            _traded_coins.add(coin)
            state["available_usdc"] = state.get("available_usdc", 0.0) - position_size
            state = _state_with_positions(state, positions, [])
            _save_state(state)

            log.info(
                "HL POSITION OPEN  id=%s  %s %s  entry=%.4f  tp=%.4f  sl=%.4f",
                pos.id, hl_side.upper(), coin,
                pos.entry_price, pos.tp_price, pos.sl_price,
            )

        # ── Pure breakout entries (SOL, AVAX, etc.) ──────────────────────────
        # No Polymarket market for these coins — use the breakout scanner alone.
        # Enter LONG if score > BREAKOUT_ENTRY_THRESH, SHORT if < -threshold.
        for coin in HL_BREAKOUT_SYMBOLS:
            if _shutdown["requested"]:
                break
            if coin in _traded_coins:
                continue
            if any(p.coin == coin for p in positions):
                continue
            n_open = len(positions)
            if n_open >= MAX_OPEN_POSITIONS:
                break
            available = state.get("available_usdc", 0.0)
            if available < MAX_POSITION_USDC * 0.5:
                break

            try:
                bk_score = compute_breakout_score(coin)
            except Exception as _exc:
                log.debug("breakout_score(%s) failed: %s", coin, _exc)
                continue

            if bk_score is None or abs(bk_score) < BREAKOUT_ENTRY_THRESH:
                continue

            hl_side = "long" if bk_score > 0 else "short"
            hl_mid  = get_mid_price(coin)
            if hl_mid is None:
                log.debug("HL: no mid price for %s — skip", coin)
                continue

            # Balance-failure backoff: if a recent order failed due to insufficient
            # balance, skip live attempts for 10 minutes to avoid log spam.
            if not HL_VIRTUAL_MODE and time.monotonic() < _balance_fail["until"]:
                continue

            position_size = min(MAX_POSITION_USDC, available * 0.5)

            log.info(
                "HL BREAKOUT ENTRY  %s %s  score=%.3f  mid=%.4f  size=$%.0f",
                hl_side.upper(), coin, bk_score, hl_mid, position_size,
            )

            if HL_VIRTUAL_MODE:
                result = _virtual_entry(coin, hl_side, position_size, hl_mid, log)
            else:
                result = _place_hl_order(
                    coin=coin,
                    side=hl_side,
                    size_usdc=position_size,
                    leverage=LEVERAGE,
                    tp_pct=TP_PCT,
                    sl_pct=SL_PCT,
                    log=log,
                    mid_price_hint=hl_mid,
                )

            if result["status"] not in ("FILLED", "VIRTUAL"):
                reason = result.get("reason", "unknown")
                if "insufficient balance" in reason:
                    _balance_fail["until"] = time.monotonic() + 600  # 10-min cooldown
                    log.warning("HL balance insufficient — pausing live order attempts for 10 min")
                else:
                    log.warning("HL breakout order not filled: %s", reason)
                continue

            pos = HLPosition(
                id=str(uuid.uuid4())[:8],
                coin=coin,
                side=hl_side,
                size_contracts=result["size_contracts"],
                size_usd=result["size_usd"],
                entry_price=result["fill_price"],
                entry_time=datetime.now(timezone.utc).isoformat(),
                tp_price=result["tp_price"],
                sl_price=result["sl_price"],
                tp_order_id=result.get("tp_order_id"),
                sl_order_id=result.get("sl_order_id"),
                margin_usdc=position_size,
            )
            positions.append(pos)
            _traded_coins.add(coin)
            state["available_usdc"] = state.get("available_usdc", 0.0) - position_size
            state = _state_with_positions(state, positions, [])
            _save_state(state)

            log.info(
                "HL POSITION OPEN  id=%s  %s %s  entry=%.4f  tp=%.4f  sl=%.4f",
                pos.id, hl_side.upper(), coin,
                pos.entry_price, pos.tp_price, pos.sl_price,
            )

        # Reset per-cycle coin tracker
        _traded_coins.clear()
        _sleep_to_next(cycle_start, log)

    log.info("HL loop shutdown requested — exiting cleanly")
    _save_state(state)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Cache of window open prices: {window_start_ts → price}
_open_price_cache: dict[int, float] = {}


def _get_window_open_price(window_start_ts: int, current_price: float,
                           coin: str = "BTC") -> float:
    """
    Return the UP token price recorded at window open.
    First time we see a window, current_price IS the open price.
    Also resets the Binance live feed CVD reference so CVD measures
    pressure since THIS window opened (not since bot startup).
    Subsequent calls return the cached first-seen price.
    """
    if window_start_ts not in _open_price_cache:
        _open_price_cache[window_start_ts] = current_price
        # Reset CVD reference so it measures from this window's open
        spot_price = _live_feed.get_price(f"{coin}USDT")
        if spot_price:
            _live_feed.set_reference(f"{coin}USDT", spot_price)
        # Prune old entries (keep last 20 windows)
        if len(_open_price_cache) > 20:
            oldest = min(_open_price_cache)
            del _open_price_cache[oldest]
    return _open_price_cache[window_start_ts]


# MACD cache for the HL loop (reuses Binance candles logic from crypto loop)
_macd_cache: dict[str, tuple[Optional[float], float]] = {}
_MACD_TTL = 60.0   # seconds


def _get_macd_score(coin: str) -> Optional[float]:
    """
    Compute MACD(3,15,3) histogram score on 1-min Binance candles.
    Cached for 60 seconds to avoid redundant candle fetches.
    Returns None if not enough candles or computation fails.
    """
    now = time.monotonic()
    cached = _macd_cache.get(coin)
    if cached and (now - cached[1]) < _MACD_TTL:
        return cached[0]

    try:
        from crypto.price_feed import get_candles  # type: ignore
        candles = get_candles(f"{coin}USDT", "1m", limit=30)
        if len(candles) < 20:
            _macd_cache[coin] = (None, now)
            return None
        closes = [c.close for c in candles]
        score  = _macd_histogram(closes, fast=3, slow=15, signal=3)
        _macd_cache[coin] = (score, now)
        return score
    except Exception:
        _macd_cache[coin] = (None, now)
        return None


def _ema(series: list[float], period: int) -> list[float]:
    k   = 2.0 / (period + 1)
    ema = [series[0]]
    for p in series[1:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def _macd_histogram(closes: list[float], fast: int, slow: int, signal: int) -> Optional[float]:
    """Return normalised MACD histogram in [-1, +1]. None if computation fails."""
    try:
        fast_ema = _ema(closes, fast)
        slow_ema = _ema(closes, slow)
        macd     = [f - s for f, s in zip(fast_ema, slow_ema)]
        sig_line = _ema(macd, signal)
        hist     = macd[-1] - sig_line[-1]
        # Normalise by recent price scale (~0.1% of price = ±1)
        scale = closes[-1] * 0.001
        if scale == 0:
            return None
        return max(-1.0, min(1.0, hist / scale))
    except Exception:
        return None


def _virtual_entry(
    coin: str,
    side: str,
    size_usdc: float,
    mid_price: float,
    log: logging.Logger,
) -> dict:
    """Simulate an HL entry in virtual mode. Uses current mid price as fill."""
    leverage = LEVERAGE
    n_dec    = {"BTC": 5, "ETH": 4}.get(coin, 5)
    sz       = round(size_usdc * leverage / mid_price, n_dec)

    is_long  = (side == "long")
    tp_price = round(mid_price * (1 + TP_PCT), 2) if is_long else round(mid_price * (1 - TP_PCT), 2)
    sl_price = round(mid_price * (1 - SL_PCT), 2) if is_long else round(mid_price * (1 + SL_PCT), 2)

    log.info(
        "VIRTUAL ENTRY %s %s  sz=%.6f  mid=%.2f  tp=%.2f  sl=%.2f",
        side.upper(), coin, sz, mid_price, tp_price, sl_price,
    )
    return {
        "status":         "VIRTUAL",
        "fill_price":     mid_price,
        "size_contracts": sz,
        "size_usd":       round(sz * mid_price, 2),
        "tp_price":       tp_price,
        "sl_price":       sl_price,
        "tp_order_id":    None,
        "sl_order_id":    None,
    }


def _sleep_to_next(cycle_start: float, log: logging.Logger) -> None:
    elapsed   = time.monotonic() - cycle_start
    sleep_for = max(0.0, LOOP_INTERVAL - elapsed)
    if elapsed > LOOP_INTERVAL * 1.5:
        log.debug("HL cycle slow: %.1fs", elapsed)
    time.sleep(sleep_for)
