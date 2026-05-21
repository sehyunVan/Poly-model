"""
crypto/loop.py — crowd-momentum trading loop for Polymarket 5-minute markets.

Runs every 20 seconds.  For each open BTC/ETH up/down market:
  1. Record the UP price the first time we see the window (= open_price).
  2. On subsequent cycles, compute the flow signal (price_drift + ob_imbalance).
  3. Enter a position if:
       - abs(signal.score) > SIGNAL_THRESHOLD
       - window is 60–180 s old  (price has formed, still 2+ min left)
       - no existing position for this market
  4. Settle resolved positions each cycle.
  5. At midnight: send daily summary + retrain if enough labeled rows.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import math
import os
import signal
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv
for _p in [_SRC.parent, _SRC.parent / "polymarket-mcp-main" / "polymarket-mcp-main"]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

from crypto.flow import compute_signal, SIGNAL_THRESHOLD, FlowSignal
from crypto.price_feed import get_candles, get_coinbase_price, get_mlofi, live_feed as _live_feed
from crypto.clob_feed import clob_feed as _clob_feed
from crypto.deribit import get_pcr_score as _get_pcr_score
from crypto.rtds_feed import rtds_feed as _rtds_feed
from crypto.pairs import get_pairs_signal_cached
from crypto.liquidation_cascade import liq_cascade as _liq_cascade
from virtual.portfolio import (
    VirtualPosition, load_virtual_portfolio, save_virtual_portfolio,
)
from monitoring.alerts import send_alert

_ROOT          = _SRC.parent
_VIRTUAL_STATE = _ROOT / "data" / "virtual_state.json"   # overridden below after VIRTUAL_MODE is set
_CRYPTO_CACHE  = _ROOT / "data" / "crypto_cache.jsonl"
_LOG_DIR       = _ROOT / "logs"

_CRYPTO_CONFIG_PATH = _ROOT / "config" / os.getenv("CRYPTO_CONFIG_FILE", "crypto_params.yaml")


def _load_crypto_config() -> dict:
    """Load crypto loop parameters from YAML, falling back to hardcoded defaults."""
    defaults = {
        "loop_interval":      20,
        "min_window_elapsed": 60,
        "max_window_elapsed": 180,
        "kelly_fraction":     0.30,
        "max_bet_pct":        0.04,
        "max_bet_abs":        20.0,
        "min_bet_abs":        2.0,
        "retrain_min_rows":   30,
        "virtual_budget":     1000.0,
    }
    try:
        import yaml
        with open(_CRYPTO_CONFIG_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return {**defaults, **loaded}
    except Exception:
        return defaults


_CFG = _load_crypto_config()

# Constants are loaded from config/crypto_params.yaml (with hardcoded defaults as fallback).
# Call _load_crypto_config() explicitly so the path block above is defined first.
LOOP_INTERVAL      = int(_CFG["loop_interval"])
MIN_WINDOW_ELAPSED = int(_CFG["min_window_elapsed"])
MAX_WINDOW_ELAPSED = int(_CFG["max_window_elapsed"])
KELLY_FRACTION     = float(_CFG["kelly_fraction"])
MAX_BET_PCT        = float(_CFG["max_bet_pct"])
MAX_BET_ABS        = float(_CFG["max_bet_abs"])
MIN_BET_ABS        = float(_CFG["min_bet_abs"])
RETRAIN_MIN_ROWS      = int(_CFG["retrain_min_rows"])
VIRTUAL_BUDGET        = float(_CFG["virtual_budget"])
# C2: halt new trades when daily loss exceeds this fraction of starting balance.
# E.g. 0.30 = stop trading after losing 30% of the day's starting capital.
DAILY_LOSS_LIMIT_PCT  = float(_CFG.get("daily_loss_limit_pct", 0.30))
# Fraction of total CLOB wallet reserved for the crypto loop.
# The rest is reserved for the swarm bot.  0.50 = crypto gets max 50%.
WALLET_SHARE = float(_CFG.get("wallet_share", 1.0))
# CLOB price band: two disjoint profitable zones — accepted: [zone2_min,zone2_max] ∪ [min,max]
# E.g. [0.68–0.70] ∪ [0.72–0.76]. The losing 0.70–0.72 middle is excluded.
MIN_CLOB_PRICE = float(_CFG.get("min_clob_price", 0.0))
MAX_CLOB_PRICE = float(_CFG.get("max_clob_price", 1.0))
# Upper band score floor: when fill >= this threshold, require stronger signal.
# Analysis of 259 settled signal_log entries in [0.80-0.90): low-score signals (<0.45)
# have negative/marginal EV (-0.010 to +0.019), while |score|>=0.45 + trend!=UP = EV +0.079.
UPPER_BAND_START     = float(_CFG.get("upper_band_start", MAX_CLOB_PRICE))
UPPER_BAND_MIN_SCORE = float(_CFG.get("upper_band_min_score", 0.0))
# Score cap: skip when |score| >= this value — high scores mean crowd is already certain.
# Analysis of 544 signal_log entries shows score >0.65 loses edge at high fills.
# Set to 1.0 to disable.
MAX_SIGNAL_SCORE = float(_CFG.get("max_signal_score", 1.0))
# Fade mode: when crowd is over-committed (high |score|, late in window),
# instead of skipping, FLIP and buy the OPPOSITE token at the cheap side.
# Asymmetric payout: pay $0.10–0.30 to win $0.70–0.90; only need ~25% WR.
FADE_MODE_ENABLED       = bool(_CFG.get("fade_mode_enabled", False))
FADE_SCORE_THRESHOLD    = float(_CFG.get("fade_score_threshold", 0.80))
FADE_MIN_WINDOW_ELAPSED = float(_CFG.get("fade_min_window_elapsed", 240.0))
FADE_MIN_SIGNAL_FILL    = float(_CFG.get("fade_min_signal_fill", 0.78))
FADE_BAND_MIN           = float(_CFG.get("fade_band_min", 0.10))
FADE_BAND_MAX           = float(_CFG.get("fade_band_max", 0.30))
FADE_MAX_BET_ABS        = float(_CFG.get("fade_max_bet_abs", 1.50))
CLOB_ZONE2_MIN = float(_CFG.get("clob_zone2_min", MIN_CLOB_PRICE))  # lower zone min
CLOB_ZONE2_MAX = float(_CFG.get("clob_zone2_max", MIN_CLOB_PRICE))  # lower zone max (= min → zone2 disabled)
# Ghost contrarian tracker: observe [GHOST_BAND_MIN, MIN_CLOB_PRICE) without real money.
GHOST_BAND_MIN  = float(_CFG.get("ghost_band_min", MIN_CLOB_PRICE))
# Upper ghost band: observe (MAX_CLOB_PRICE, GHOST_UPPER_MAX] without real money.
# Hypothesis: when crowd pays > MAX_CLOB_PRICE, they may be overcrowded → fade them.
GHOST_UPPER_MAX = float(_CFG.get("ghost_upper_max", MAX_CLOB_PRICE))
_GHOST_LOG       = _ROOT / "data" / "contrarian_ghost.jsonl"
_GHOST_UPPER_LOG = _ROOT / "data" / "contrarian_ghost_upper.jsonl"
_SIGNAL_LOG      = _ROOT / "data" / "signal_log.jsonl"   # every signal that fired, win or skip
# Contrarian mode: when True, the band-passing signal token is the entry trigger
# but we buy the OPPOSITE (cheap) token. Intended for 15m mean-reversion strategy.
CONTRARIAN_MODE = bool(_CFG.get("contrarian_mode", False))
# Inverse-fill sizing floor: 0.5 = bet reduces to 50% at MAX_CLOB_PRICE (default).
# Set 0.0 in 1d config to reduce bet to ~0 at expensive fills — correct when
# high-fill wins pay tiny amounts and you need WR near the fill price to break even.
INVERSE_FILL_WEIGHT_MIN = float(_CFG.get("inverse_fill_weight_min", 0.5))
_GHOST_STAKE     = 3.0   # hypothetical stake ($) used for ghost PnL calculation

# ── Timeframe support ─────────────────────────────────────────────────────────
# Set CRYPTO_CONFIG_FILE env var before import to load a different config.
# E.g. CRYPTO_CONFIG_FILE=crypto_15m_params.yaml for 15-minute markets.
_TF_TO_SECS    = {"5m": 300, "15m": 900, "1h": 3600}
TIMEFRAME      = str(_CFG.get("timeframe", "5m"))
WINDOW_SECONDS = int(_CFG.get("window_seconds", _TF_TO_SECS.get(TIMEFRAME, 300)))
DRIFT_SCALE_CFG = float(_CFG.get("drift_scale", 0.06))   # passed to compute_signal; larger for longer windows
_LOG_FILE_NAME  = str(_CFG.get("log_file", "crypto.log"))
_CRYPTO_CACHE   = _ROOT / "data" / str(_CFG.get("cache_file", "crypto_cache.jsonl"))  # override early definition


def _in_clob_band(price: float) -> bool:
    """Return True if price falls in any accepted profitable zone."""
    in_main  = MIN_CLOB_PRICE <= price < MAX_CLOB_PRICE
    in_zone2 = CLOB_ZONE2_MIN <= price < CLOB_ZONE2_MAX and CLOB_ZONE2_MAX > CLOB_ZONE2_MIN
    return in_main or in_zone2
# Session filter: only enter positions during proven profitable UTC hours.
TRADE_HOUR_START = int(_CFG.get("trade_hour_start", 0))
TRADE_HOUR_END   = int(_CFG.get("trade_hour_end", 24))
TRADE_HOUR_BLOCK = set(int(h) for h in _CFG.get("trade_hour_block", []))
# Per-symbol extra block: applied on top of TRADE_HOUR_BLOCK.
# ETH disabled entirely — ETH PnL -$66 vs BTC +$6 (423 real trades).
ETH_HOUR_BLOCK   = set(int(h) for h in _CFG.get("eth_hour_block", []))
# SOL added 2026-04-21 — start with minimal [0, 1] block; widen if WR < 65%.
SOL_HOUR_BLOCK   = set(int(h) for h in _CFG.get("sol_hour_block", []))
# Symbols allowed to trade. Defaults to both; set to [BTC] to disable ETH.
ACTIVE_SYMBOLS      = set(_CFG.get("active_symbols", ["BTC", "ETH"]))
# Force-UP hours (UTC): override DOWN → UP in structurally bullish hours.
# 8am KST (= 23 UTC): 86.2% UP rate (n=94) — Asian market-open flow.
FORCE_UP_HOURS_UTC  = set(int(h) for h in _CFG.get("force_up_hours_utc", []))
# 1-min Binance spot confirmation: require this % move in signal direction before entry.
BINANCE_1M_CONFIRM_PCT = float(_CFG.get("binance_1m_confirm_pct", 0.0))
# Variable bet sizing: weakest signals get MIN_SIZING_PCT × MAX_BET_ABS,
# strongest signals get MAX_BET_ABS. Mirrors swarm's score-norm sizing.
# 0.60 → bet ranges from 60% to 100% of max_bet_abs based on confidence.
MIN_SIZING_PCT = float(_CFG.get("min_sizing_pct", 0.60))
# Cross-asset pairs signal: compare BTC/ETH/SOL returns over a short window.
# When one asset leads another, the lagger tends to follow. Used as a
# confirmation filter: agree → boost bet; disagree → block trade.
# Disabled by default — enable after verifying correlation in signal_log.
PAIRS_FILTER_ENABLED = bool(_CFG.get("pairs_filter_enabled", False))
PAIRS_AGREE_BOOST    = float(_CFG.get("pairs_agree_boost", 1.15))   # multiply bet when pairs agrees
PAIRS_DISAGREE_BLOCK = bool(_CFG.get("pairs_disagree_block", True))  # skip when pairs conflicts
PAIRS_WINDOW_SEC     = float(_CFG.get("pairs_window_sec", 60.0))     # comparison lookback (seconds)
PAIRS_MIN_DIVERGENCE = float(_CFG.get("pairs_min_divergence", 0.002))# min return gap to emit signal
# Liquidation cascade filter: when Binance forced-liquidations exceed the $10M threshold
# in 15s, the cascade signal fires. Agree → boost; disagree → block.
# Disabled by default — enable after accumulating 50+ cascade events in signal_log.
CASCADE_FILTER_ENABLED = bool(_CFG.get("cascade_filter_enabled", False))
CASCADE_AGREE_BOOST    = float(_CFG.get("cascade_agree_boost", 1.25))   # bet multiplier on agreement
CASCADE_DISAGREE_BLOCK = bool(_CFG.get("cascade_disagree_block", True))  # skip when cascade conflicts
# Spot trend filter: block non-NEUTRAL 15m trends. Enabled for live loop (79% WR NEUTRAL
# vs 61-65% non-NEUTRAL). Disabled for paper loops to collect data across all trend conditions.
SPOT_TREND_FILTER = bool(_CFG.get("spot_trend_filter", True))

# Mid-window profit-taking exit: sell owned tokens when price appreciates enough.
# Disabled by default; enable after validating in virtual mode (crypto15).
MID_EXIT_ENABLED         = bool(_CFG.get("mid_exit_enabled", False))
MID_EXIT_PRICE_THRESHOLD = float(_CFG.get("mid_exit_price_threshold", 0.88))

# ── ML entry filter ───────────────────────────────────────────────────────────
# Two modes:
#   Legacy classifier: ml_filter_threshold > 0  → predicts P(trade_win)
#   EV regressor:      ml_epnl_threshold set    → predicts expected PnL/dollar
ML_FILTER_THRESHOLD = float(_CFG.get("ml_filter_threshold", 0.0))
_ml_epnl_raw        = _CFG.get("ml_epnl_threshold")          # None = disabled
ML_EPNL_THRESHOLD: float | None = float(_ml_epnl_raw) if _ml_epnl_raw is not None else None
_ML_FILTER: dict | None = None
_ml_is_epnl: bool = False
if ML_FILTER_THRESHOLD > 0.0 or ML_EPNL_THRESHOLD is not None:
    try:
        import joblib as _joblib
        _ml_path = _ROOT / "models" / "crypto_filter.pkl"
        _ML_FILTER = _joblib.load(_ml_path)
        _ml_is_epnl = _ML_FILTER.get("objective") == "epnl"
        _log_startup = logging.getLogger(__name__)
        if _ml_is_epnl:
            _log_startup.info(
                "ML EV filter loaded — epnl_threshold=%s  n_samples=%d  base_epnl=%.4f",
                f"{ML_EPNL_THRESHOLD:+.3f}" if ML_EPNL_THRESHOLD is not None else "n/a",
                _ML_FILTER.get("n_samples", 0),
                _ML_FILTER.get("base_epnl", 0),
            )
        else:
            _log_startup.info(
                "ML entry filter loaded — threshold=%.2f  n_samples=%d  base_wr=%.1f%%",
                ML_FILTER_THRESHOLD,
                _ML_FILTER.get("n_samples", 0),
                _ML_FILTER.get("base_wr", 0) * 100,
            )
    except Exception as _ml_e:
        _log_startup = logging.getLogger(__name__)
        _log_startup.warning("ML filter load failed — running without it: %s", _ml_e)
        _ML_FILTER = None

# ── Crypto flow model (trained nightly on 2800+ labeled rows) ─────────────────
# LogisticRegression trained on [price_drift, ob_imbalance, score] to predict UP (1) vs DOWN (0).
# Load unconditionally — used to enhance signal confidence.
_CRYPTO_FLOW_MODEL = None
_CRYPTO_FLOW_MODEL_COEF = None
try:
    import pickle
    _flow_path = _ROOT / "models" / "crypto_flow_model.pkl"
    if _flow_path.exists():
        with open(_flow_path, "rb") as f:
            _CRYPTO_FLOW_MODEL = pickle.load(f)
        _log_startup = logging.getLogger(__name__)
        _log_startup.info(
            "Crypto flow model loaded from %s (2800+ labeled rows)",
            _flow_path.name,
        )
except Exception as _flow_e:
    _log_startup = logging.getLogger(__name__)
    _log_startup.warning("Crypto flow model load failed — using flow signal only: %s", _flow_e)

GAMMA_BASE    = "https://gamma-api.polymarket.com"
SYMBOLS       = {"BTC": "btc", "ETH": "eth", "SOL": "sol"}
_BINANCE_MAP  = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Polymarket CLOB taker fee — deducted from available capital on every fill
# to make virtual PnL reflect real trading economics.
FEE_RATE = float(_CFG.get("fee_rate", 0.02))

# Real-money mode: set VIRTUAL_MODE=false in .env to place live CLOB orders.
VIRTUAL_MODE = os.getenv("VIRTUAL_MODE", "true").lower() not in ("false", "0", "no")
# Use a separate state file for real trading so it starts with a clean slate.
# Config can override filenames for non-5m loops (e.g. 15m_virtual_state.json).
_vstate_name   = _CFG.get("state_file_virtual", "virtual_state.json")
_rstate_name   = _CFG.get("state_file_live", "real_state.json")
_VIRTUAL_STATE = _ROOT / "data" / (_vstate_name if VIRTUAL_MODE else _rstate_name)

# Order placement and balance queries are now routed through ExecutionBackend
# (see infra/backend.py). _place_crypto_order and _get_clob_balance are no
# longer imported here — the backend passed to run() handles both.
#
# CTF redemption (_redeem_positions, _get_pending_ctf) is NOT part of the
# backend because it involves on-chain transactions that are separate from
# CLOB order placement. It stays here as a direct import.
if not VIRTUAL_MODE:
    try:
        from crypto.redeem import (                              # type: ignore
            redeem_redeemable_positions as _redeem_positions,
            get_pending_ctf_value       as _get_pending_ctf,
            wrap_usdc_e_to_pusd         as _wrap_usdc_e,
        )
    except ImportError as _e:
        raise RuntimeError(
            "VIRTUAL_MODE=false but crypto.redeem could not be imported: "
            f"{_e}"
        ) from _e
else:
    _redeem_positions = None  # type: ignore
    _get_pending_ctf  = None  # type: ignore
    _wrap_usdc_e      = None  # type: ignore

from infra.backend import ExecutionBackend, LiveBackend  # noqa: E402  (for type hints + isinstance checks)

_HTTP = httpx.Client(timeout=8.0)


def _get_clob_spread(token_id: str) -> tuple[Optional[float], Optional[float]]:
    """
    Return (fill_price, bid_ask_spread) for buying a token on the CLOB.

    fill_price = min(asks)  — real cost to buy the token now.
    bid_ask_spread = min(asks) - max(bids) — pure market-maker cost to
    enter and immediately exit (1-4 cents typically for 5-min markets).

    These are different from CLOB-vs-Gamma divergence (which reflects
    the information gap between Gamma AMM and CLOB real-time pricing).

    Returns (None, None) on error or empty book.
    """
    try:
        r = _HTTP.get(
            "https://clob.polymarket.com/book",
            params={"token_id": token_id},
        )
        r.raise_for_status()
        book = r.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if asks:
            # Seed the WS cache so price_change events can apply going forward.
            _clob_feed.seed_book(token_id, bids, asks)
            fill   = min(float(a["price"]) for a in asks)
            ba_spd = (fill - max(float(b["price"]) for b in bids)) if bids else None
            return fill, ba_spd
    except Exception:
        pass
    return None, None


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(
            _LOG_DIR / _LOG_FILE_NAME, maxBytes=10*1024*1024, backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if sys.stdout.isatty():
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger("crypto.loop")


# ── Market discovery ──────────────────────────────────────────────────────────

def _current_window_starts(now: datetime, look_ahead: int = 4) -> list[int]:
    ts   = int(now.timestamp())
    base = ts - (ts % WINDOW_SECONDS)
    return [base + i * WINDOW_SECONDS for i in range(look_ahead)]


def _fetch_market(slug: str) -> Optional[dict]:
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/events", params={"slug": slug})
        r.raise_for_status()
        events = r.json()
        if not events:
            return None
        ev = events[0]
        if ev.get("closed", True):
            return None
        markets = ev.get("markets", [])
        if not markets:
            return None
        m = markets[0]
        m["_event_title"] = ev.get("title", slug)
        return m
    except Exception:
        return None


def _parse_market(m: dict, now: datetime) -> Optional[tuple]:
    """
    Returns (market_id, symbol, up_token_id, current_up_price, window_start_ts,
             secs_to_close, end_dt) or None.
    """
    try:
        end_str = m.get("endDate", "")
        if not end_str:
            return None
        end_dt    = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        secs_left = (end_dt - now).total_seconds()
        if secs_left <= 0 or secs_left > WINDOW_SECONDS + 10:
            return None

        prices_raw    = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
        current_price = float(prices_raw[0])
        if not (0.05 < current_price < 0.95):
            return None

        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) < 2:
            return None
        # Verified via settled-market data: outcomePrices[0] → 1.0 when UP wins,
        # and token_ids[0] also resolves to 0 when UP doesn't happen → token_ids[0]
        # is the UP-wins token.  token_ids[1] = DOWN-wins token.
        up_token   = token_ids[0]   # pays $1 if UP happens
        down_token = token_ids[1]   # pays $1 if DOWN happens

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

        market_id       = m.get("conditionId", m.get("id", ""))
        window_start_ts = int(end_dt.timestamp()) - WINDOW_SECONDS   # approximate

        return (market_id, symbol, up_token, down_token, current_price,
                window_start_ts, secs_left, end_dt)
    except Exception:
        return None


def _discover_markets(now: datetime) -> list[tuple]:
    found = []
    seen_ids: set[str] = set()
    for ts in _current_window_starts(now, look_ahead=4):
        for sym, prefix in SYMBOLS.items():
            slug = f"{prefix}-updown-{TIMEFRAME}-{ts}"
            m    = _fetch_market(slug)
            if m is None:
                continue
            parsed = _parse_market(m, now)
            if parsed is None:
                continue
            mid = parsed[0]
            if mid not in seen_ids:
                seen_ids.add(mid)
                found.append(parsed)
    return found


# ── Cache ─────────────────────────────────────────────────────────────────────

def _append_cache(market_id: str, symbol: str, slug: str,
                  sig: FlowSignal, open_price: float,
                  bet_size: float, direction: str,
                  ask_price: Optional[float] = None):
    row = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "market_id":   market_id,
        "slug":        slug,
        "symbol":      symbol,
        "features": {
            "price_drift":      sig.price_drift,
            "ob_imbalance":     sig.ob_imbalance,
            "score":            sig.score,
            "current_price":    sig.current_price,
            "open_price":       open_price,
            "clob_fill":        round(ask_price, 4) if ask_price is not None else None,
            "clob_vs_gamma":    round(ask_price - sig.current_price, 4) if ask_price is not None else None,
            "deribit_pcr":      round(sig.pcr_score, 4),
            "tick_velocity":    round(sig.tick_velocity_score, 4),
            "trade_imbalance":  round(sig.trade_imbalance_score, 4),
            "oracle_lag":       round(sig.oracle_lag_score, 4),
            "hawkes":           round(sig.hawkes_score, 4),
            "mlofi":            round(sig.mlofi_score, 4),
        },
        "prediction":  direction,
        "bet_size":    bet_size,
        "fee_usdc":    round(bet_size * FEE_RATE, 4),
        "label":       None,
    }
    _CRYPTO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CRYPTO_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _update_cache_label(market_id: str, label: int):
    if not _CRYPTO_CACHE.exists():
        return
    lines = _CRYPTO_CACHE.read_text(encoding="utf-8").splitlines()
    updated = []
    for line in lines:
        try:
            r = json.loads(line)
            if r.get("market_id") == market_id and r.get("label") is None:
                r["label"] = label
            updated.append(json.dumps(r))
        except Exception:
            updated.append(line)
    _CRYPTO_CACHE.write_text("\n".join(updated) + "\n", encoding="utf-8")


# ── Settlement ────────────────────────────────────────────────────────────────

import re as _re

def _slug_from_title(title: str) -> Optional[str]:
    """
    'BTC Up or Down (wn-5m-1773967500)' → 'btc-updown-{TIMEFRAME}-1773967500'
    TIMEFRAME is loaded from config (e.g. "5m" or "15m").
    """
    m = _re.search(r"(BTC|ETH|SOL).*?(\d{10,})", title)
    if not m:
        return None
    return f"{m.group(1).lower()}-updown-{TIMEFRAME}-{m.group(2)}"


def _fetch_outcome_by_slug(slug: str) -> Optional[float]:
    """
    Returns the UP token final price for a closed event, or None if not yet
    resolved (Chainlink hasn't snapped to 0/1) or event not found.
    """
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/events", params={"slug": slug})
        r.raise_for_status()
        events = r.json()
        if not events:
            return None
        ev = events[0]
        if not ev.get("closed", False):
            return None
        markets = ev.get("markets", [])
        if not markets:
            return None
        prices_raw = json.loads(markets[0].get("outcomePrices", "[0.5,0.5]"))
        price = float(prices_raw[0])
        if not (price < 0.05 or price > 0.95):
            return None   # not yet resolved by Chainlink
        return price
    except Exception:
        return None


def _settle_positions(vp, log: logging.Logger, state_path: Optional[Path] = None):
    changed = False
    now = datetime.now(timezone.utc)
    for pos in list(vp.positions):
        if pos.category != "crypto":
            continue
        try:
            slug = _slug_from_title(pos.title)
            if not slug:
                log.debug("Cannot derive slug from title: %s", pos.title)
                continue

            price_up_fin = _fetch_outcome_by_slug(slug)
            if price_up_fin is None:
                continue
            outcome_up   = price_up_fin > 0.5
            label        = 1 if outcome_up else 0

            won = (pos.direction == "YES" and outcome_up) or \
                  (pos.direction == "NO"  and not outcome_up)
            # Note: Polymarket taker fee (FEE_RATE) is already deducted from
            # available_usdc at bet placement (loop.py line 868: size + fee).
            # realized_pnl here is the gross win/loss before that fee.
            pnl = pos.size_usdc * (1.0 / pos.fill_price - 1.0) if won else -pos.size_usdc

            pos.outcome       = label
            pos.realized_pnl  = round(pnl, 4)
            pos.settle_time   = datetime.now(timezone.utc)
            vp.positions.remove(pos)
            vp.closed_positions.append(pos)
            vp.available_usdc += pos.size_usdc + pnl
            vp.mark_updated()
            changed = True

            _update_cache_label(pos.market_id, label)
            log.info(
                "SETTLED  %s  dir=%s  outcome=%s  pnl=%+.2f",
                pos.market_id[:16], pos.direction,
                "UP" if outcome_up else "DOWN", pnl,
            )
            # Alert if loss is greater than the stake (indicates a settlement bug,
            # not a normal loss — a normal loss is exactly -size_usdc).
            if pnl < -(pos.size_usdc * 1.5):
                send_alert(
                    f"Crypto abnormal loss: {pos.direction} pnl=${pnl:.2f} "
                    f"stake=${pos.size_usdc:.2f}",
                    level="WARNING",
                )
        except Exception as exc:
            log.warning("Settle check failed %s: %s", pos.market_id, exc)

        # Stuck position guard: if still open >15 min after fill, the market
        # has long since resolved. A failed slug lookup or API error is locking
        # capital. Log a warning so it can be investigated.
        age_min = (now - pos.fill_time).total_seconds() / 60
        if age_min > 15:
            log.warning(
                "STUCK POSITION: %s %s open %.0f min past fill — "
                "settlement may be failing; capital locked",
                pos.market_id[:16], pos.direction, age_min,
            )

    if changed:
        save_virtual_portfolio(vp, state_path or _VIRTUAL_STATE)


# ── Mid-window profit-taking exit ────────────────────────────────────────────

def _check_mid_window_exits(
    vp,
    backend: "ExecutionBackend",
    log: logging.Logger,
    state_path: "Optional[Path]" = None,
) -> None:
    """
    Check all open crypto positions for mid-window profit-taking opportunity.
    If the current CLOB ask price for an owned token reaches MID_EXIT_PRICE_THRESHOLD,
    place a SELL order (live) or simulate exit (virtual).

    Only fires when MID_EXIT_ENABLED is True (off by default).
    Positions opened < 15s ago are skipped to avoid spurious trigger on entry noise.
    """
    if not MID_EXIT_ENABLED:
        return

    changed = False
    now = datetime.now(timezone.utc)

    for pos in list(vp.positions):
        if pos.category != "crypto" or not pos.bought_token_id:
            continue
        # Skip positions entered very recently — wait for price to form
        age_s = (now - pos.fill_time).total_seconds()
        if age_s < 15:
            continue

        current_ask, _ = _get_clob_spread(pos.bought_token_id)
        if current_ask is None or current_ask < MID_EXIT_PRICE_THRESHOLD:
            continue

        # Threshold crossed — attempt exit
        token_count = pos.size_usdc / pos.fill_price
        gain_pct    = (current_ask - pos.fill_price) / pos.fill_price * 100
        log.info(
            "MID-EXIT trigger: %s %s token=%.3f threshold=%.3f "
            "entry=%.3f gain=%.1f%% age=%.0fs",
            pos.direction, pos.market_id[:12], current_ask,
            MID_EXIT_PRICE_THRESHOLD, pos.fill_price, gain_pct, age_s,
        )

        if isinstance(backend, LiveBackend):
            from crypto.execution import sell_crypto_position
            from infra.types import ClobTokenId, ConditionId
            result = sell_crypto_position(
                ClobTokenId(pos.bought_token_id),
                token_count,
                ConditionId(pos.market_id),
                log,
            )
            if result["status"] != "FILLED":
                log.info("MID-EXIT SELL failed (status=%s) — holding position", result["status"])
                continue
            proceeds = result["proceeds_usdc"]
        else:
            # Virtual: simulate exit at current ask (optimistic proxy; real bid ~1-2¢ lower)
            proceeds = round(current_ask * token_count, 6)

        realized_pnl = round(proceeds - pos.size_usdc, 6)
        vp.close_position_manually(pos, realized_pnl)
        changed = True
        log.info(
            "MID-EXIT closed: %s  proceeds=$%.4f  pnl=%+.4f  entry=%.3f  exit=%.3f",
            pos.market_id[:12], proceeds, realized_pnl, pos.fill_price, current_ask,
        )

    if changed:
        from virtual.portfolio import save_virtual_portfolio
        save_virtual_portfolio(vp, state_path or _VIRTUAL_STATE)


# ── Ghost contrarian tracker ──────────────────────────────────────────────────

def _log_ghost_trade(
    market_id: str, slug: str, symbol: str,
    signal_dir: str, signal_clob_ask: float, log: logging.Logger,
    log_path: "Path | None" = None,
    band_label: str = "lower",
    # enriched signal context
    score: float = 0.0,
    price_drift: float = 0.0,
    ob_imbalance: float = 0.0,
    alpha: float = 0.0,
    window_elapsed: float = 0.0,
    spot_trend: str = "NEUTRAL",
    gamma_price: float = 0.0,
) -> None:
    """
    Record a hypothetical contrarian bet without placing real money.

    Called when the flow signal fires but the fill price falls outside the
    accepted CLOB bands.  By betting the opposite direction on paper we test
    whether the excluded range has exploitable anti-correlation.

    log_path: which ghost JSONL file to write to (defaults to _GHOST_LOG).
    band_label: "lower" or "upper" — stored in the record for analysis.
    ghost_fill_est ≈ 1 - signal_clob_ask: approximate fill for opposite token.
    """
    if log_path is None:
        log_path = _GHOST_LOG
    contrarian_dir = "NO" if signal_dir == "YES" else "YES"
    ghost_fill_est = round(1.0 - signal_clob_ask, 4)
    record = {
        "market_id":       market_id,
        "slug":            slug,
        "symbol":          symbol,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "signal_dir":      signal_dir,
        "contrarian_dir":  contrarian_dir,
        "signal_clob_ask": signal_clob_ask,
        "ghost_fill_est":  ghost_fill_est,
        "band":            band_label,
        # signal internals — for future band calibration
        "score":           round(score, 4),
        "price_drift":     round(price_drift, 4),
        "ob_imbalance":    round(ob_imbalance, 4),
        "alpha":           round(alpha, 4),
        "window_elapsed":  round(window_elapsed, 1),
        "spot_trend":      spot_trend,
        "gamma_price":     round(gamma_price, 4),
        "settled":         False,
        "outcome_up":      None,
        "ghost_won":       None,
        "ghost_pnl":       None,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("ghost log (%s): write failed: %s", band_label, exc)
        return
    log.info(
        "GHOST-%s  %s  signal=%s at %.4f → contrarian=%s  est_fill=%.4f  score=%+.3f",
        band_label.upper(), symbol, signal_dir, signal_clob_ask, contrarian_dir, ghost_fill_est, score,
    )


def _log_signal(
    market_id: str, slug: str, symbol: str,
    signal_dir: str, clob_ask: float, executed: bool, skip_reason: str,
    score: float, price_drift: float, ob_imbalance: float, alpha: float,
    window_elapsed: float, spot_trend: str, gamma_price: float,
    stake: float, log: logging.Logger,
    mom_30s: "float | None" = None,
    whale_notional: float = 0.0,
    whale_is_buy: "bool | None" = None,
) -> None:
    """
    Append every fired signal to signal_log.jsonl — whether executed or skipped.
    Outcome (settled/won/pnl) is filled in later by _settle_signal_log().
    This is the master dataset for long-term signal analysis across all bands.
    """
    record = {
        "market_id":      market_id,
        "slug":           slug,
        "symbol":         symbol,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "signal_dir":     signal_dir,
        "clob_ask":       round(clob_ask, 4),
        "gamma_price":    round(gamma_price, 4),
        "score":          round(score, 4),
        "price_drift":    round(price_drift, 4),
        "ob_imbalance":   round(ob_imbalance, 4),
        "alpha":          round(alpha, 4),
        "window_elapsed": round(window_elapsed, 1),
        "spot_trend":     spot_trend,
        "executed":       executed,
        "skip_reason":    skip_reason,   # "" if executed, else e.g. "band_lower"/"band_upper"
        "stake":          round(stake, 4),
        # Leading indicators (data collection — not used in trading decisions yet)
        "mom_30s":        round(mom_30s, 6) if mom_30s is not None else None,
        "whale_notional": round(whale_notional) if whale_notional > 0 else None,
        "whale_dir":      ("buy" if whale_is_buy else "sell") if whale_notional > 0 else None,
        "settled":        False,
        "outcome_up":     None,
        "won":            None,
        "pnl":            None,
    }
    try:
        with open(_SIGNAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("signal_log: write failed: %s", exc)


def _settle_ghost_trades(
    log: logging.Logger, log_path: "Path | None" = None
) -> None:
    """Resolve pending ghost trades by fetching market outcomes from Gamma API."""
    if log_path is None:
        log_path = _GHOST_LOG
    if not log_path.exists():
        return
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        log.warning("ghost settle: read failed: %s", exc)
        return

    changed = False
    for r in records:
        if r.get("settled"):
            continue
        price_up_fin = _fetch_outcome_by_slug(r["slug"])
        if price_up_fin is None:
            continue
        outcome_up     = price_up_fin > 0.5
        contrarian_dir = r["contrarian_dir"]
        ghost_won = (contrarian_dir == "YES" and outcome_up) or \
                    (contrarian_dir == "NO"  and not outcome_up)
        fill      = r["ghost_fill_est"]
        ghost_pnl = round(
            _GHOST_STAKE * (1.0 / fill - 1.0) if ghost_won else -_GHOST_STAKE, 4,
        )
        r["settled"]    = True
        r["outcome_up"] = outcome_up
        r["ghost_won"]  = ghost_won
        r["ghost_pnl"]  = ghost_pnl
        changed = True
        band = r.get("band", "lower")
        log.info(
            "GHOST-%s SETTLED  %s  contrarian=%s  outcome=%s  won=%s  pnl=%+.2f",
            band.upper(), r["symbol"], contrarian_dir,
            "UP" if outcome_up else "DOWN", ghost_won, ghost_pnl,
        )

    if changed:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        except Exception as exc:
            log.warning("ghost settle: write failed: %s", exc)


def _settle_signal_log(log: logging.Logger, max_per_call: int = 40) -> None:
    """
    Scan signal_log.jsonl for unsettled entries and fill in outcome fields.

    For each unsettled record:
      - Fetch the final UP-token price via _fetch_outcome_by_slug()
      - Determine outcome_up (True if UP won)
      - Compute won (signal_dir matched outcome)
      - Compute pnl based on stake and clob_ask fill price

    max_per_call: cap API calls per cycle to avoid blocking the loop.
    At 20s interval and ~200ms per call, 40 entries ≈ 8s — safe headroom.
    The backlog clears within a few cycles; steady-state is 1-3 per cycle.

    Called every cycle alongside _settle_positions() and _settle_ghost_trades().
    This is what makes signal_log.jsonl useful for cross-band analysis.
    """
    if not _SIGNAL_LOG.exists():
        return
    try:
        with open(_SIGNAL_LOG, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        log.warning("signal_log settle: read failed: %s", exc)
        return

    changed = False
    attempted = 0
    for r in records:
        if r.get("settled"):
            continue
        if attempted >= max_per_call:
            break
        attempted += 1

        price_up_fin = _fetch_outcome_by_slug(r.get("slug", ""))
        if price_up_fin is None:
            continue

        outcome_up = price_up_fin > 0.5
        signal_dir = r.get("signal_dir", "")
        won = (signal_dir == "YES" and outcome_up) or \
              (signal_dir == "NO"  and not outcome_up)

        clob_ask = r.get("clob_ask", 0.0)
        stake    = r.get("stake", 0.0)
        if clob_ask > 0 and stake > 0:
            pnl = round(stake * (1.0 / clob_ask - 1.0) if won else -stake, 4)
        else:
            pnl = None

        r["settled"]    = True
        r["outcome_up"] = outcome_up
        r["won"]        = won
        r["pnl"]        = pnl
        changed = True
        log.info(
            "signal_log settled  %s  dir=%s  outcome=%s  won=%s  pnl=%s",
            r.get("slug", "")[-20:], signal_dir,
            "UP" if outcome_up else "DOWN", won, pnl,
        )

    if changed:
        try:
            with open(_SIGNAL_LOG, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        except Exception as exc:
            log.warning("signal_log settle: write failed: %s", exc)


# ── Position sizing ───────────────────────────────────────────────────────────

_SYMBOL_ENC = {"BTC": 0, "ETH": 1, "SOL": 2}

def _build_ml_feature_vec(
    sig: FlowSignal,
    clob_fill: float,
    clob_vs_gamma: float,
    ts: datetime,
    symbol: str = "BTC",
) -> list[float]:
    """Build the feature vector for the ML entry filter.

    Order must match FEATURE_COLS in scripts/train_crypto_filter.py.
    """
    hour = ts.hour
    return [
        abs(sig.score),
        clob_fill,
        sig.ob_imbalance,
        sig.price_drift,
        clob_vs_gamma,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        sig.pcr_score             or 0.0,
        sig.tick_velocity_score   or 0.0,
        sig.trade_imbalance_score or 0.0,
        sig.oracle_lag_score      or 0.0,
        sig.hawkes_score          or 0.0,
        sig.mlofi_score           or 0.0,
        float(_SYMBOL_ENC.get(symbol, 0)),
    ]


def _size_bet(signal_score: float, market_price: float,
              available: float, ml_prob: float | None = None,
              fill_price: float | None = None,
              pred_epnl: float | None = None) -> float:
    """
    Variable bet sizing.

    EV regressor (pred_epnl + fill_price):
      Derives implied win probability from predicted EV, then computes Kelly
      fraction for the specific fill price.  Fill is already baked into Kelly —
      the inverse-fill discount is skipped to avoid double-penalising high fills.

    Legacy classifier (ml_prob):
      Uses ML win-probability as a confidence proxy.

    Flat (neither):
      confidence=1.0, Kelly anchored to observed 77.8% WR.
    """
    if pred_epnl is not None and fill_price is not None:
        # EV = p*net_win - (1-p)  →  p = (EV+1) / (net_win+1)
        _fee    = float(_CFG.get("fee_rate", 0.02))
        net_win = (1.0 - fill_price) / fill_price * (1.0 - _fee)
        if net_win > 0:
            p_implied = min(max((pred_epnl + 1.0) / (net_win + 1.0), 0.0), 1.0)
            kelly_f   = max(0.0, p_implied - (1.0 - p_implied) / net_win)
        else:
            p_implied = kelly_f = 0.0
        confidence = kelly_f
        p_win      = p_implied if kelly_f > 0 else 0.0
        odds       = max((1.0 / fill_price) - 1.0, 0.001)
        kelly      = max(0.0, (p_win * odds - (1.0 - p_win)) / odds)
        apply_fill_discount = False   # fill already baked into Kelly

    elif ml_prob is not None and ML_FILTER_THRESHOLD < 1.0:
        prob_range = max(1.0 - ML_FILTER_THRESHOLD, 0.01)
        prob_norm  = min(max((ml_prob - ML_FILTER_THRESHOLD) / prob_range, 0.0), 1.0)
        confidence = prob_norm
        p_win      = 0.778 + 0.05 * confidence
        odds       = (1.0 / market_price) - 1.0
        if odds <= 0:
            return 0.0
        kelly      = max(0.0, (p_win * odds - (1.0 - p_win)) / odds)
        apply_fill_discount = True

    else:
        confidence = 1.0   # flat sizing: score inversely correlates with WR
        p_win      = 0.778 + 0.05 * confidence
        odds       = (1.0 / market_price) - 1.0
        if odds <= 0:
            return 0.0
        kelly      = max(0.0, (p_win * odds - (1.0 - p_win)) / odds)
        apply_fill_discount = True

    effective_cap = MAX_BET_ABS * (MIN_SIZING_PCT + (1.0 - MIN_SIZING_PCT) * confidence)
    kelly_size    = kelly * KELLY_FRACTION * available
    raw           = available * MAX_BET_PCT * (MIN_SIZING_PCT + (1.0 - MIN_SIZING_PCT) * confidence)
    result        = max(0.0, min(raw, kelly_size, effective_cap))

    if apply_fill_discount and fill_price is not None and MAX_CLOB_PRICE > MIN_CLOB_PRICE:
        fill_range  = MAX_CLOB_PRICE - MIN_CLOB_PRICE
        fill_weight = 1.0 - (1.0 - INVERSE_FILL_WEIGHT_MIN) * (fill_price - MIN_CLOB_PRICE) / fill_range
        result     *= max(INVERSE_FILL_WEIGHT_MIN, min(1.0, fill_weight))

    return result


# ── Midnight tasks ────────────────────────────────────────────────────────────

def _midnight_tasks(log: logging.Logger):
    # Daily summary — read directly from _VIRTUAL_STATE (real_state.json in live mode).
    # _load_crypto_stats() hardcodes virtual_state.json and would show paper-archive data.
    try:
        mode = "LIVE" if not VIRTUAL_MODE else "VIRTUAL"
        d = json.loads(_VIRTUAL_STATE.read_text(encoding="utf-8")) if _VIRTUAL_STATE.exists() else {}
        closed = d.get("closed_positions", [])
        crypto = [p for p in closed if p.get("category") == "crypto"]
        wins   = sum(1 for p in crypto if (p.get("realized_pnl") or 0) > 0)
        losses = len(crypto) - wins
        pnl    = sum(p.get("realized_pnl") or 0 for p in crypto)
        open_n = len(d.get("positions", []))
        hr     = f"  HR {wins/len(crypto)*100:.0f}%" if crypto else ""
        avail  = d.get("available_usdc", 0.0)
        rows_total, rows_labeled = 0, 0
        if _CRYPTO_CACHE.exists():
            for line in _CRYPTO_CACHE.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                    rows_total += 1
                    if r.get("label") is not None:
                        rows_labeled += 1
                except Exception:
                    pass
        sign = "+" if pnl >= 0 else ""
        send_alert(
            f"[{mode}] Crypto daily summary\n"
            f"Avail: ${avail:.2f} | PnL: {sign}${pnl:.2f}{hr}\n"
            f"{wins}W/{losses}L | Open: {open_n} | Settled: {len(crypto)}\n"
            f"Cache rows: {rows_labeled} labeled / {rows_total} total",
            level="INFO",
        )
    except Exception as exc:
        log.warning("Midnight summary failed: %s", exc)

    # Retrain if enough labeled rows
    if not _CRYPTO_CACHE.exists():
        return
    rows = []
    with open(_CRYPTO_CACHE, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("label") is not None:
                    rows.append(r)
            except Exception:
                pass
    n = len(rows)
    if n < RETRAIN_MIN_ROWS:
        log.info("Crypto retrain skipped — %d labeled rows (need %d)", n, RETRAIN_MIN_ROWS)
        return

    log.info("Retraining flow model on %d labeled rows...", n)
    try:
        from sklearn.linear_model import LogisticRegression
        import pickle
        FEATURE_KEYS = ["price_drift", "ob_imbalance", "score"]
        X = np.array([[r["features"].get(k, 0.0) for k in FEATURE_KEYS] for r in rows])
        y = np.array([int(r["label"]) for r in rows])
        if len(set(y)) < 2:
            log.info("Retrain skipped — single class in labels.")
            return
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        model_path = _ROOT / "models" / "crypto_flow_model.pkl"
        with open(model_path, "wb") as mf:
            pickle.dump(clf, mf)
        log.info("Flow model saved. Coef: %s", clf.coef_.tolist())
        send_alert(
            f"Crypto flow model retrained on {n} rows "
            f"(up={int(y.sum())}, down={n-int(y.sum())})",
            level="INFO",
        )
    except Exception as exc:
        log.error("Retrain failed: %s", exc)


def _get_flow_model_confidence(sig: FlowSignal, log: logging.Logger) -> tuple[Optional[float], Optional[float]]:
    """
    Get the trained flow model's confidence that the signal is correct.

    Returns (model_prob_up, model_prob_down) where model_prob_up is P(UP wins)
    from the LogisticRegression trained on [price_drift, ob_imbalance, score].

    Used to boost confidence in high-conviction signals from the model,
    especially when paper training shows the model learned real patterns.
    """
    if _CRYPTO_FLOW_MODEL is None:
        return None, None
    try:
        features = np.array([[sig.price_drift, sig.ob_imbalance, sig.score]])
        probs = _CRYPTO_FLOW_MODEL.predict_proba(features)[0]
        model_prob_down = probs[0]  # P(DOWN/0)
        model_prob_up = probs[1]    # P(UP/1)
        return model_prob_up, model_prob_down
    except Exception as exc:
        log.debug("Flow model prediction failed: %s", exc)
        return None, None


# ── Trend detection ───────────────────────────────────────────────────────────

# Cache trend so every market in the same cycle shares one Binance API call.
_trend_cache: dict[str, tuple[str, float]] = {}   # symbol → ("UP"|"DOWN"|"NEUTRAL", ts)
_TREND_TTL = 30.0   # seconds before re-fetching
_TREND_THRESHOLD = 0.002   # 0.2% move required to declare a trend


def _get_spot_trend(symbol: str, log: logging.Logger) -> str:
    """
    Return the 15-minute spot price trend for BTC or ETH from Binance 1m candles.

    Returns "UP", "DOWN", or "NEUTRAL".
    Cached for _TREND_TTL seconds so every window in one cycle shares one fetch.
    """
    now_ts = time.monotonic()
    cached = _trend_cache.get(symbol)
    if cached and (now_ts - cached[1]) < _TREND_TTL:
        return cached[0]

    binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
    try:
        candles = get_candles(binance_sym, interval="1m", limit=16)
        if len(candles) < 2:
            return "NEUTRAL"
        ret = (candles[-1].close - candles[0].open) / candles[0].open
        if ret > _TREND_THRESHOLD:
            trend = "UP"
        elif ret < -_TREND_THRESHOLD:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"
        _trend_cache[symbol] = (trend, now_ts)
        return trend
    except Exception as exc:
        log.debug("Trend fetch failed for %s: %s", symbol, exc)
        return "NEUTRAL"   # safe fallback: don't block on API error


# Cache 1-min Binance return per symbol to avoid duplicate fetches per cycle.
_binance_1m_cache: dict[str, tuple[float, float]] = {}  # symbol → (return, mono_ts)
_BINANCE_1M_TTL = 20.0  # seconds


def _get_binance_1m_return(symbol: str, log: logging.Logger) -> float:
    """
    Return the 1-minute Binance spot price return for BTC or ETH.
    Positive = price went up in the last minute.
    Returns 0.0 on error (safe — won't block trading on API failure).
    Cached for _BINANCE_1M_TTL seconds per cycle.
    """
    now_ts = time.monotonic()
    cached = _binance_1m_cache.get(symbol)
    if cached and (now_ts - cached[1]) < _BINANCE_1M_TTL:
        return cached[0]

    binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
    try:
        candles = get_candles(binance_sym, interval="1m", limit=2)
        if len(candles) < 2:
            return 0.0
        # Compare last completed candle's close vs its open
        last = candles[-2]  # last fully closed 1-min candle
        ret = (last.close - last.open) / last.open if last.open > 0 else 0.0
        _binance_1m_cache[symbol] = (ret, now_ts)
        log.debug("Binance 1m return %s: %+.4f%%", symbol, ret * 100)
        return ret
    except Exception as exc:
        log.debug("Binance 1m fetch failed for %s: %s", symbol, exc)
        return 0.0  # safe fallback: don't block on API error


# Cache MACD score per symbol to avoid redundant 50-candle fetches per cycle.
_macd_cache: dict[str, tuple[float, float]] = {}  # symbol → (macd_score, mono_ts)
_MACD_TTL = 20.0  # seconds — refresh once per loop cycle

# Cache Multi-Level OFI (REST call) per symbol per cycle.
_mlofi_cache: dict[str, tuple[Optional[float], float]] = {}  # symbol → (score, mono_ts)
_MLOFI_TTL = 20.0


def _get_mlofi_score(symbol: str, log: logging.Logger) -> Optional[float]:
    """MLOFI from Binance depth (5 levels, distance-weighted). Cached per cycle."""
    now_ts = time.monotonic()
    cached = _mlofi_cache.get(symbol)
    if cached and (now_ts - cached[1]) < _MLOFI_TTL:
        return cached[0]
    binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
    try:
        score = get_mlofi(binance_sym, levels=5)
        _mlofi_cache[symbol] = (score, now_ts)
        return score
    except Exception as exc:
        log.debug("MLOFI fetch failed for %s: %s", symbol, exc)
        _mlofi_cache[symbol] = (None, now_ts)
        return None


def _get_macd_score(symbol: str, log: logging.Logger) -> Optional[float]:
    """
    Return normalised MACD(3,15,3) histogram for BTC or ETH on 1-min candles.
    Score is in [-1, +1]: positive = bullish momentum, negative = bearish.
    Returns None if not enough candles or fetch fails (weight redistributed to drift).
    Cached for _MACD_TTL seconds.
    """
    from crypto.indicators import macd_score_normalized  # type: ignore

    now_ts = time.monotonic()
    cached = _macd_cache.get(symbol)
    if cached and (now_ts - cached[1]) < _MACD_TTL:
        return cached[0]

    binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
    try:
        # Need slow(15) + signal(3) - 1 = 17 candles minimum; fetch 50 for safety
        candles = get_candles(binance_sym, interval="1m", limit=50)
        if len(candles) < 17:
            return None
        closes = [c.close for c in candles]
        score  = macd_score_normalized(closes)
        if score is None:
            return None
        _macd_cache[symbol] = (score, now_ts)
        log.debug("MACD score %s: %+.3f", symbol, score)
        return score
    except Exception as exc:
        log.debug("MACD fetch failed for %s: %s", symbol, exc)
        return None


# ── Funding rate score (contrarian fade) ─────────────────────────────────────
# High positive funding = crowded longs = negative score contribution (fade UP).
# High negative funding = crowded shorts = positive score (fade DOWN).
# Cached 30 min — funding changes every 8h but drifts intra-period.
_funding_cache: dict[str, tuple[float, float]] = {}  # symbol → (rate, mono_ts)
_FUNDING_TTL = 1800.0
_FUNDING_SCALE = 0.002  # 0.2%/8h rate → ±1.0 score


def _get_funding_score(symbol: str, log: logging.Logger) -> Optional[float]:
    """Contrarian Binance Futures funding rate in [-1, +1]. None on fetch failure."""
    from crypto.price_feed import get_futures_funding_rate

    binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
    now_ts = time.monotonic()
    cached = _funding_cache.get(binance_sym)
    if cached and (now_ts - cached[1]) < _FUNDING_TTL:
        rate = cached[0]
    else:
        try:
            rate = get_futures_funding_rate(binance_sym)
            _funding_cache[binance_sym] = (rate, now_ts)
            log.debug("Funding rate %s: %+.5f (%.3f%%/8h)", binance_sym, rate, rate * 100)
        except Exception as exc:
            log.debug("Funding rate fetch failed: %s", exc)
            return None
    return max(-1.0, min(1.0, -rate / _FUNDING_SCALE))


# ── OI-delta liquidation cascade proxy ───────────────────────────────────────
# OI dropping while price moves = forced liquidations amplifying the current direction.
# OI rising = new positions entering = also trend-confirming.
# Both are combined with the price direction (current_price vs open_price) to get sign.
_oi_cache: dict[str, tuple[float, float]] = {}  # symbol → (oi, mono_ts)
_OI_STALE = 60.0   # discard prev OI if older than this
_OI_MIN_DELTA = 0.0005  # ignore deltas < 0.05% of OI (noise)
_OI_SCALE = 0.003        # 0.3% OI change → magnitude 1.0


def _get_liq_score(
    symbol: str, current_price: float, open_price: float, log: logging.Logger
) -> Optional[float]:
    """
    OI-delta liquidation proxy in [-1, +1].
    Positive = bullish cascade (short squeeze), negative = bearish (long liquidation).
    Returns None when delta is negligible or direction is ambiguous.
    """
    from crypto.price_feed import get_futures_open_interest

    binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
    now_ts = time.monotonic()
    prev = _oi_cache.get(binance_sym)

    try:
        current_oi = get_futures_open_interest(binance_sym)
    except Exception as exc:
        log.debug("OI fetch failed: %s", exc)
        return None

    _oi_cache[binance_sym] = (current_oi, now_ts)

    if prev is None or (now_ts - prev[1]) > _OI_STALE or prev[0] <= 0:
        return None  # no usable previous OI to delta against

    oi_delta_pct = (current_oi - prev[0]) / prev[0]
    if abs(oi_delta_pct) < _OI_MIN_DELTA:
        return None  # delta too small — noise

    if current_price == open_price:
        return None  # no price direction to assign cascade direction

    price_dir = 1.0 if current_price > open_price else -1.0
    magnitude = min(abs(oi_delta_pct) / _OI_SCALE, 1.0)
    liq = price_dir * magnitude
    log.debug("OI delta %s: %+.4f%% → liq_score=%+.3f", binance_sym, oi_delta_pct * 100, liq)
    return liq


# ── Startup validation ────────────────────────────────────────────────────────

def startup_checks(log: logging.Logger, backend: "ExecutionBackend | None" = None) -> None:
    """
    Validate system state before entering the trading loop.

    Raises RuntimeError with a clear message on any failure so the loop
    refuses to start rather than trading in an unknown state.

    Checks (in order):
      1. Required Python packages are importable.
      2. .env has KEY and FUNDER (live mode only).
      3. State file is loadable JSON (not corrupted from a crash).
      4. pnl_history is internally consistent (no phantom losses from prior phases).
      5. Gamma API is reachable (network / DNS).
      6. CLOB balance is fetchable and above MIN_BET_ABS (live mode only).

    Root cause: several past incidents (pyarrow missing, -$799 pnl_history from
    broken crypto phase, vp.available_usdc mismatch) could have been caught
    here before touching real money.
    """
    errors: list[str] = []

    # 1. Required packages
    for pkg in ("httpx", "yaml", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"missing package: {pkg!r} — run pip install {pkg}")

    # pyarrow: needed for feature cache Parquet save/load (labels → retraining).
    # Its absence silently breaks the cache without raising an error mid-loop.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        log.warning(
            "startup_checks: pyarrow not installed — feature cache Parquet will "
            "fail silently. Run: pip install pyarrow"
        )
        # Not fatal: loop can trade without retraining. Warn only.

    # 2. KEY / FUNDER present in live mode
    if not VIRTUAL_MODE:
        if not os.getenv("KEY"):
            errors.append("VIRTUAL_MODE=false but KEY is not set in .env")
        if not os.getenv("FUNDER"):
            errors.append("VIRTUAL_MODE=false but FUNDER is not set in .env")

    # 3. State file readable (if it exists)
    _check_state = backend.state_path() if backend is not None else _VIRTUAL_STATE
    if _check_state.exists():
        try:
            import json as _json
            raw = _json.loads(_check_state.read_text(encoding="utf-8"))
            # Pydantic validation — catches schema drift from prior versions
            from virtual.portfolio import VirtualPortfolio
            vp_test = VirtualPortfolio.model_validate(raw)

            # 4. pnl_history sanity — cumulative loss > 90% of initial budget
            #    indicates corrupted history from a prior broken phase (root cause
            #    of the "trading halted repeatedly" incident with -$799 entries).
            cum_pnl = sum(e.get("pnl", 0) for e in vp_test.pnl_history)
            if vp_test.initial_budget > 0 and cum_pnl < -(vp_test.initial_budget * 0.90):
                errors.append(
                    f"pnl_history cumulative PnL ${cum_pnl:.2f} exceeds 90% loss of "
                    f"initial_budget ${vp_test.initial_budget:.2f} — likely corrupted "
                    f"from a prior broken phase. Clear pnl_history in {_check_state} "
                    f"before restarting."
                )
        except Exception as exc:
            errors.append(f"state file {_check_state} is unreadable: {exc}")

    # 5. Gamma API reachable
    try:
        import httpx as _httpx
        r = _httpx.get(f"{GAMMA_BASE}/markets", params={"limit": 1}, timeout=8.0)
        r.raise_for_status()
    except Exception as exc:
        errors.append(f"Gamma API unreachable: {exc}")

    # 6. Live mode: CLOB balance check (via backend if provided)
    _is_live = backend is not None and isinstance(backend, LiveBackend)
    if _is_live and not errors:  # skip if env keys already missing
        balance = backend.get_balance(log)  # type: ignore[union-attr]
        if balance is None:
            errors.append(
                "Cannot fetch CLOB wallet balance — check KEY/FUNDER in .env "
                "and Polymarket API connectivity."
            )
        elif balance < MIN_BET_ABS:
            errors.append(
                f"CLOB wallet balance ${balance:.2f} is below MIN_BET_ABS "
                f"${MIN_BET_ABS:.2f} — fund the wallet before starting."
            )

    if errors:
        msg = "startup_checks FAILED — refusing to start:\n" + "\n".join(
            f"  [{i+1}] {e}" for i, e in enumerate(errors)
        )
        log.error(msg)
        raise RuntimeError(msg)

    log.info(
        "startup_checks PASSED (mode=%s)",
        "LIVE" if (backend is not None and isinstance(backend, LiveBackend)) else "VIRTUAL",
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(log: logging.Logger, backend: "ExecutionBackend | None" = None):
    """
    Main trading loop.

    Args:
        log     : logger from _setup_logging()
        backend : ExecutionBackend instance from make_backend().
                  Defaults to VirtualBackend if not provided (backward compat).
    """
    # Fall back to VirtualBackend if called without one (e.g. in tests).
    if backend is None:
        from infra.backend import VirtualBackend
        backend = VirtualBackend()
        log.warning("run() called without a backend — defaulting to VirtualBackend")

    _is_live = isinstance(backend, LiveBackend)
    # Single authoritative path for all state I/O — derived from the backend
    # so live and virtual state files are never mixed (root cause of the
    # "dashboard mixed virtual history with real state" incident).
    _state_path = backend.state_path()

    # ── Start Binance WebSocket live price feed ────────────────────────────────
    # Runs in a daemon thread; provides <50ms price updates for oracle-lag filter.
    _live_feed.start()
    log.info("Binance WebSocket live feed started (BTC + ETH)")

    # ── Start Polymarket CLOB WebSocket feed ──────────────────────────────────
    # Maintains live orderbooks per token_id. Replaces REST /book polling in
    # flow.py (_cross_imbalance) and loop.py (_get_clob_spread).
    # Falls back to REST automatically when stale — zero risk on disconnect.
    _clob_feed.start()
    log.info("Polymarket CLOB WebSocket feed started")

    # ── Start Polymarket RTDS feed (Binance relay + Chainlink oracle) ─────────
    # Connects to wss://ws-live-data.polymarket.com and tracks both the Binance
    # price relay and Chainlink oracle price per symbol. The gap between them
    # (oracle_lag_score) predicts which direction the settlement will go since
    # Chainlink lags Binance by 10–30 s on average.
    _rtds_feed.start()
    log.info("Polymarket RTDS feed started (Binance relay + Chainlink oracle)")

    # ── Start Binance liquidation cascade detector ─────────────────────────────
    # Subscribes to wss://fstream.binance.com/ws/!forceOrder@arr (no auth needed).
    # Daemon thread — does not block the main loop. Signal available immediately
    # but typically empty until a real cascade event (>$10M in 15s) fires.
    _liq_cascade.start()
    log.info("Liquidation cascade detector started")

    vp = load_virtual_portfolio(_state_path, VIRTUAL_BUDGET)

    # ── C1: sync vp.available_usdc to real wallet balance at startup ──────────
    # In virtual mode vp.available_usdc reflects accumulated paper-trading
    # history ($10k+).  In real mode we must size bets against actual wallet
    # funds, not the virtual number, or every order will be mis-sized.
    if _is_live:
        log.info("LIVE MODE — syncing available_usdc with real CLOB wallet balance...")
        real_balance = backend.get_balance(log)
        if real_balance is None:
            log.error(
                "Cannot fetch real wallet balance — refusing to start in live mode. "
                "Check KEY/FUNDER in .env and Polymarket API connectivity."
            )
            return
        if real_balance < 1.0:
            log.error(
                "Real wallet balance $%.2f is too low to trade — fund the wallet first.",
                real_balance,
            )
            return
        crypto_budget = round(real_balance * WALLET_SHARE, 4)
        vp.available_usdc     = crypto_budget
        vp.real_clob_balance  = real_balance   # dashboard: real wallet (liquid USDC only)
        # Also fix initial_budget if it's still the virtual default (1000.0).
        # This ensures the dashboard shows the real starting capital, not $1,000.
        if vp.initial_budget == VIRTUAL_BUDGET:
            vp.initial_budget = crypto_budget
            log.info("initial_budget set to crypto budget: $%.2f", crypto_budget)
        # Set initial_real_clob_balance once — never overwrite.
        # This is the true baseline for all-time PnL calculation.
        if vp.initial_real_clob_balance == 0.0:
            vp.initial_real_clob_balance = real_balance
            log.info("PnL baseline set: initial_real_clob=$%.2f", real_balance)
        save_virtual_portfolio(vp, _state_path)
        log.info(
            "C1: real CLOB $%.2f × wallet_share=%.2f → crypto budget $%.2f  "
            "real_pnl_alltime=$%.2f",
            real_balance, WALLET_SHARE, crypto_budget, vp.real_pnl_all_time(),
        )

    log.info(
        "Crypto flow loop started | mode=%s | interval=%ds | "
        "entry_window=%d-%ds | threshold=%.2f",
        "LIVE" if _is_live else "VIRTUAL",
        LOOP_INTERVAL, MIN_WINDOW_ELAPSED, MAX_WINDOW_ELAPSED, SIGNAL_THRESHOLD,
    )
    log.info(
        "Budget: available=%.2f  open=%d  closed=%d",
        vp.available_usdc, len(vp.positions), len(vp.closed_positions),
    )
    if _ML_FILTER is not None and _ml_is_epnl:
        log.info(
            "ML EV filter ACTIVE — epnl_threshold=%s  n_samples=%d  base_epnl=%.4f",
            f"{ML_EPNL_THRESHOLD:+.3f}" if ML_EPNL_THRESHOLD is not None else "n/a",
            _ML_FILTER.get("n_samples", 0),
            _ML_FILTER.get("base_epnl", 0),
        )
    elif _ML_FILTER is not None:
        log.info(
            "ML entry filter ACTIVE — threshold=%.2f  n_samples=%d  base_wr=%.1f%%",
            ML_FILTER_THRESHOLD,
            _ML_FILTER.get("n_samples", 0),
            _ML_FILTER.get("base_wr", 0) * 100,
        )
    elif ML_FILTER_THRESHOLD > 0.0 or ML_EPNL_THRESHOLD is not None:
        log.warning("ML filter threshold set but model failed to load — running without filter")
    else:
        log.info("ML entry filter DISABLED")

    _shutdown = {"requested": False}
    def _sig(s, f):
        log.info("Shutdown signal received.")
        _shutdown["requested"] = True
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Memory: track the UP price the first time we see each market window
    # key = market_id, value = (open_price, first_seen_ts)
    open_prices: dict[str, tuple[float, float]] = {}

    last_retrain_date = date.today().isoformat()
    traded_markets: set[str] = set()   # market_ids we've already placed a bet on
    _last_redeem_mono: float = 0.0      # monotonic time of last balance-sync check (10 min)
    _last_ar_run_ts:  float = 0.0      # wall-clock time of last actual AR redemption attempt (12 h)
    _pending_redemption_credit: float = 0.0  # amount just redeemed — suppresses deposit detection

    # ── C2: daily loss circuit breaker ────────────────────────────────────────
    # Track starting capital at the top of each day; halt if daily loss
    # exceeds DAILY_LOSS_LIMIT_PCT of that starting amount.
    def _current_capital(vp) -> float:
        return vp.available_usdc + sum(p.size_usdc for p in vp.positions)

    day_start_capital = _current_capital(vp)
    daily_halt        = False   # set True when limit hit; reset at midnight

    while not _shutdown["requested"]:
        cycle_start = time.monotonic()
        now         = datetime.now(timezone.utc)
        now_ts      = now.timestamp()

        # Midnight tasks + C2 daily reset
        # NOTE: do NOT add "and now.hour == 0" — if the loop hangs past 01:00 UTC
        # the condition would never fire, leaving daily_halt=True forever.
        today_str = date.today().isoformat()
        if today_str != last_retrain_date:
            _midnight_tasks(log)
            last_retrain_date = today_str
            # C2: reset daily circuit breaker for the new day
            vp               = load_virtual_portfolio(_state_path, VIRTUAL_BUDGET)
            day_start_capital = _current_capital(vp)
            daily_halt        = False
            log.info("Daily reset: start_capital=%.2f  daily_halt cleared", day_start_capital)

        vp = load_virtual_portfolio(_state_path, VIRTUAL_BUDGET)
        _settle_positions(vp, log, _state_path)
        _check_mid_window_exits(vp, backend, log, _state_path)
        _settle_ghost_trades(log)
        _settle_ghost_trades(log, log_path=_GHOST_UPPER_LOG)
        _settle_signal_log(log)

        # ── Auto-redeem: recover USDC.e from settled CTF tokens ───────────────
        # Check real CLOB balance (not virtual accounting) every 10 minutes.
        # If the actual wallet is below 2× the max bet, redeem winning CTF tokens.
        if not VIRTUAL_MODE and _redeem_positions is not None:
            _mono = time.monotonic()
            if _mono - _last_redeem_mono > 600:
                _last_redeem_mono = _mono
                _key    = os.getenv("KEY", "")
                _funder = os.getenv("FUNDER", "")
                if _key and _funder:
                    try:
                        real_bal = backend.get_balance(log) or 0.0
                        # Always sync real_clob_balance and available_usdc to the
                        # real CLOB every 10 min.  In live mode, wins land as CTF
                        # tokens (not USDC) so available_usdc drifts above the real
                        # CLOB balance between redemptions.  Without this sync, C2
                        # compares day_start_capital ($48) against a drifted
                        # available_usdc ($18) and falsely declares a 61% loss.

                        # ── Deposit detection ─────────────────────────────────
                        # Max a single winning trade can return: MAX_BET_ABS / min_clob_price
                        # If balance jumped more than 4× that, it must be an external deposit —
                        # UNLESS the jump matches a recent CTF redemption (_pending_redemption_credit).
                        # Redemption returns can be large (many positions redeemed at once) and must
                        # not be counted as deposits or real_pnl_all_time will be understated.
                        _max_trade_win = MAX_BET_ABS / max(MIN_CLOB_PRICE, 0.50)
                        _delta = real_bal - vp.real_clob_balance
                        if vp.real_clob_balance > 0 and _delta > _max_trade_win * 4:
                            if _pending_redemption_credit > 0 and _delta <= _pending_redemption_credit * 1.5:
                                log.info(
                                    "balance-sync: balance jump +$%.2f matches CTF redemption $%.2f"
                                    " — not counting as deposit",
                                    _delta, _pending_redemption_credit,
                                )
                            else:
                                vp.total_detected_deposits = round(
                                    vp.total_detected_deposits + _delta, 4
                                )
                                vp.detected_deposits.append({
                                    "ts":     now.isoformat(),
                                    "amount": round(_delta, 4),
                                    "from":   round(vp.real_clob_balance, 4),
                                    "to":     round(real_bal, 4),
                                    "note":   "external top-up",
                                })
                                log.info(
                                    "DEPOSIT DETECTED: +$%.2f  (CLOB $%.2f → $%.2f)  "
                                    "total_deposits=$%.2f",
                                    _delta, vp.real_clob_balance, real_bal,
                                    vp.total_detected_deposits,
                                )
                            _pending_redemption_credit = 0.0

                        crypto_bal = round(real_bal * WALLET_SHARE, 4)
                        vp.available_usdc    = crypto_bal
                        vp.real_clob_balance = real_bal
                        # Refresh pending CTF value (winning positions not yet redeemed).
                        # Included in real_pnl_all_time() for accurate PnL display.
                        if _get_pending_ctf is not None:
                            try:
                                vp.pending_ctf_usdc = _get_pending_ctf(_funder, log)
                            except Exception:
                                pass
                        save_virtual_portfolio(vp, _state_path)
                        # Only raise day_start_capital — never lower it.
                        # Lowering it on every sync lets losses erode the C2
                        # baseline, effectively disabling the circuit breaker.
                        # Raise it only when redemption pushes balance above baseline
                        # (e.g. CTF tokens redeemed → genuine capital recovery).
                        if crypto_bal > day_start_capital:
                            day_start_capital = crypto_bal
                        log.info(
                            "balance-sync: real CLOB $%.2f × share=%.2f → crypto $%.2f  "
                            "real_pnl_alltime=$%.2f  deposits=$%.2f  day_start=$%.2f",
                            real_bal, WALLET_SHARE, crypto_bal,
                            vp.real_pnl_all_time(), vp.total_detected_deposits,
                            day_start_capital,
                        )
                        _AR_INTERVAL = 43200  # 12 hours (routine sweep)
                        _AR_EMERGENCY_INTERVAL = 600   # 10 min (low-balance emergency)
                        _secs_since_ar = now.timestamp() - _last_ar_run_ts
                        _low_balance = (
                            crypto_bal < MAX_BET_ABS
                            and vp.pending_ctf_usdc > 0
                            and _secs_since_ar > _AR_EMERGENCY_INTERVAL
                        )
                        _routine = _secs_since_ar > _AR_INTERVAL
                        if _routine or _low_balance:
                            _last_ar_run_ts = now.timestamp()
                            _reason = (
                                f"low balance ${crypto_bal:.2f} < max_bet ${MAX_BET_ABS:.2f}"
                                f", pending_ctf=${vp.pending_ctf_usdc:.2f}"
                                if _low_balance and not _routine
                                else f"12-hour interval elapsed"
                            )
                            log.info(
                                "auto-redeem: %s — attempting (crypto share $%.2f)...",
                                _reason, crypto_bal,
                            )
                            # Step 1: wrap any USDC.e that V1 CTF redemptions deposited
                            # to wallet → pUSD so it is visible to the CLOB.
                            wrapped = _wrap_usdc_e(_funder, _key, log) if _wrap_usdc_e else 0.0
                            # Step 2: redeem winning CTF positions on-chain
                            redeemed = _redeem_positions(_funder, _key, log)
                            if redeemed > 0 or wrapped > 0:
                                # Credit = full pending CTF value before redemption,
                                # not just the redeemed amount.  Polymarket sometimes
                                # auto-redeems additional positions in a separate batch
                                # minutes after our AR tx, causing a second balance jump
                                # that would otherwise be misclassified as a user deposit.
                                # Using the pre-redemption CTF total as the ceiling ensures
                                # any subsequent auto-redemption jump is absorbed.
                                _pending_redemption_credit = max(redeemed, vp.pending_ctf_usdc)
                                # Immediately decrement pending_ctf_usdc so the
                                # dashboard does not show stale unredeemed value
                                # until the next balance-sync refresh.
                                vp.pending_ctf_usdc = max(0.0, round(vp.pending_ctf_usdc - redeemed, 4))
                                new_bal = backend.get_balance(log)
                                if new_bal and new_bal > 0:
                                    new_crypto = round(new_bal * WALLET_SHARE, 4)
                                    vp.available_usdc    = new_crypto
                                    vp.real_clob_balance = new_bal
                                    save_virtual_portfolio(vp, _state_path)
                                    day_start_capital = _current_capital(vp)
                                    log.info(
                                        "auto-redeem: resynced real CLOB $%.2f → crypto $%.2f"
                                        " (redeemed=$%.2f wrapped=$%.2f)"
                                        " pending_ctf=$%.2f  day_start_capital reset to $%.2f",
                                        new_bal, new_crypto, redeemed, wrapped,
                                        vp.pending_ctf_usdc, day_start_capital,
                                    )
                    except Exception as _redeem_exc:
                        log.warning("auto-redeem: error — %s", _redeem_exc)

        # Purge open_prices entries for old windows (> 15 min ago)
        stale = [mid for mid, (_, seen_ts) in open_prices.items()
                 if now_ts - seen_ts > 900]
        for mid in stale:
            del open_prices[mid]

        # Discover live markets
        markets = _discover_markets(now)

        for (market_id, symbol, up_token, down_token, current_price,
             window_start_ts, secs_left, end_dt) in markets:

            slug = f"{SYMBOLS[symbol]}-updown-{TIMEFRAME}-{window_start_ts}"

            # Record opening price the first time we see this market.
            # Also snapshot the live Binance price as WebSocket reference —
            # this lets get_return() measure how much spot has moved since window open.
            if market_id not in open_prices:
                open_prices[market_id] = (current_price, now_ts)
                binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
                live_open = _live_feed.get_price(binance_sym)
                if live_open:
                    _live_feed.set_reference(binance_sym, live_open)
                # Subscribe CLOB WS feed to this market's tokens so the book
                # cache is warm before the entry window opens (~200s from now).
                _clob_feed.subscribe([up_token, down_token])
                log.debug("New window: %s %s  open_price=%.3f  binance_ref=%.2f  secs_left=%.0f  clob_ws=subscribed",
                          symbol, slug[-16:], current_price, live_open or 0, secs_left)
                continue   # wait at least one cycle before acting

            open_price, first_seen_ts = open_prices[market_id]
            window_elapsed = now_ts - window_start_ts

            # Entry window: 60–180 s after window opened
            if not (MIN_WINDOW_ELAPSED <= window_elapsed <= MAX_WINDOW_ELAPSED):
                log.info(
                    "%s  %s  elapsed=%.0fs  outside entry window [%d-%d]",
                    symbol, slug[-16:], window_elapsed,
                    MIN_WINDOW_ELAPSED, MAX_WINDOW_ELAPSED,
                )
                continue

            # Skip if already traded
            if market_id in traded_markets:
                continue

            # ── Session filter ────────────────────────────────────────────
            # Only enter new positions during hours with proven positive edge.
            # ★ Hour-based session filter — DISABLED 2026-04-27 filter removal test.
            # TRADE_HOUR_START/END set to 0/24 (24 hours), TRADE_HOUR_BLOCK empty.
            # Per-symbol blocks (ETH, SOL) also empty. All hours are allowed.
            _hr = now.hour
            # if not (TRADE_HOUR_START <= _hr < TRADE_HOUR_END) or _hr in TRADE_HOUR_BLOCK:  [DISABLED]
            #     continue
            # Per-symbol hour blocks — DISABLED (empty arrays).
            # if symbol == "ETH" and _hr in ETH_HOUR_BLOCK:  [DISABLED — eth_hour_block is []]
            #     log.info(...); continue
            # if symbol == "SOL" and _hr in SOL_HOUR_BLOCK:  [DISABLED — sol_hour_block is []]
            #     log.info(...); continue

            # ★ Active symbol filter — ENABLED FOR ALL 2026-04-27.
            # ACTIVE_SYMBOLS=[BTC, ETH, SOL]. All symbols trade.
            if any(p.market_id == market_id for p in vp.positions
                   if p.category == "crypto"):
                continue

            # ── C2: daily loss circuit breaker ────────────────────────────────
            # Use realized PnL from today's settled crypto positions, not balance
            # math.  Balance-based checks fire spuriously when the swarm bot
            # spends from the shared wallet, making the CLOB sync reset
            # available_usdc downward even though no crypto positions closed.
            # NOTE: always re-evaluate loss_pct each cycle — day_start_capital can
            # rise (AR redemption) after daily_halt was set, bringing loss_pct back
            # below the limit.  If so, clear daily_halt automatically.
            _today = date.today().isoformat()
            def _fill_date(p) -> str:
                ft = getattr(p, "fill_time", None)
                if ft is None:
                    return ""
                if hasattr(ft, "date"):   # datetime object
                    return ft.date().isoformat()
                return str(ft)[:10]      # string or other
            today_realized = sum(
                p.realized_pnl
                for p in vp.closed_positions
                if p.category == "crypto"
                and p.realized_pnl is not None
                and _fill_date(p) == _today
            )
            daily_loss = -today_realized   # positive when we've lost money
            loss_pct   = daily_loss / day_start_capital if day_start_capital > 0 else 0
            if loss_pct >= DAILY_LOSS_LIMIT_PCT:
                if not daily_halt:
                    send_alert(
                        f"DAILY LOSS LIMIT HIT  "
                        f"realized_loss=${daily_loss:.2f} ({loss_pct*100:.1f}%)  "
                        f"start=${day_start_capital:.2f}  "
                        f"limit={DAILY_LOSS_LIMIT_PCT*100:.0f}%  "
                        f"mode={'LIVE' if _is_live else 'VIRTUAL'}",
                        level="WARNING",
                    )
                    log.warning(
                        "DAILY LOSS LIMIT HIT — halting new trades  "
                        "loss=$%.2f (%.1f%%)  limit=%.0f%%",
                        daily_loss, loss_pct * 100, DAILY_LOSS_LIMIT_PCT * 100,
                    )
                daily_halt = True
                continue
            elif daily_halt:
                # day_start_capital rose (e.g. AR redemption) and loss_pct is now
                # below the limit — automatically lift the halt.
                daily_halt = False
                log.info(
                    "C2 cleared — loss_pct=%.1f%% now below limit=%.0f%%  "
                    "start=$%.2f  loss=$%.2f",
                    loss_pct * 100, DAILY_LOSS_LIMIT_PCT * 100,
                    day_start_capital, daily_loss,
                )

            # Compute flow signal (14 sources: drift, OB, CVD, MACD, liq, funding,
            #   CLOB-CVD, PCR, tick-vel, trade-imb, oracle-lag, hawkes, mlofi)
            binance_sym = _BINANCE_MAP.get(symbol, "BTCUSDT")
            cvd      = _live_feed.get_cvd_score(binance_sym)
            macd     = _get_macd_score(symbol, log)
            funding  = _get_funding_score(symbol, log)
            liq      = _get_liq_score(symbol, current_price, open_price, log)
            clob_cvd = _clob_feed.get_clob_cvd_score(up_token)
            # Signal batch 2: leading-indicator signals (deployed 2026-04-24 session 1)
            pcr       = _get_pcr_score()   # Deribit BTC PCR (5-min cache)
            trade_imb = _live_feed.get_trade_imbalance(binance_sym, window_seconds=60.0)
            tick_vel  = _clob_feed.get_tick_velocity(up_token)
            _clob_feed.record_price(up_token, current_price)
            # Signal batch 3: oracle lag + hawkes + MLOFI (deployed 2026-04-24 session 2)
            oracle_lag = _rtds_feed.get_oracle_lag_score(symbol)
            hawkes     = _live_feed.get_hawkes_ratio(binance_sym)
            mlofi      = _get_mlofi_score(symbol, log)

            # Cross-exchange divergence: kept for data collection, weight now 0.
            _bnc_price  = _live_feed.get_price(binance_sym)
            _cb_symbol  = "BTC-USD" if symbol == "BTC" else "SOL-USD" if symbol == "SOL" else None
            _coin_price = get_coinbase_price(_cb_symbol) if _cb_symbol else None

            # Leading indicators (30s momentum + whale detection — data collection only).
            mom_30s = _live_feed.get_recent_return(binance_sym, window_seconds=30.0)
            whale_notional, whale_is_buy = _live_feed.get_max_trade_notional(
                binance_sym, window_seconds=30.0, min_notional=50_000.0
            )

            try:
                sig = compute_signal(
                    up_token, down_token, current_price, open_price,
                    cvd, macd, liq, funding,
                    clob_cvd_score=clob_cvd,
                    drift_scale=DRIFT_SCALE_CFG,
                    clob_feed=_clob_feed,
                    binance_price=_bnc_price,
                    coinbase_price=_coin_price,
                    pcr_score=pcr,
                    tick_velocity_score=tick_vel,
                    trade_imbalance_score=trade_imb,
                    oracle_lag_score=oracle_lag,
                    hawkes_score=hawkes,
                    mlofi_score=mlofi,
                    mom_30s_raw=mom_30s,
                )
            except Exception as exc:
                log.warning("Signal computation failed: %s", exc)
                continue

            # Get trained flow model confidence (2800+ labeled rows)
            model_prob_up, model_prob_down = _get_flow_model_confidence(sig, log)

            log.info(
                "%s  %s  elapsed=%.0fs  drift=%+.3f  ob=%.2f  cvd=%s  macd=%s  "
                "liq=%s  funding=%s  clob_cvd=%s  pcr=%s  tick=%s  imb=%s  "
                "lag=%s  hwk=%s  ofi=%s  "
                "score=%+.2f  alpha=%+.3f  dir=%s  model=%s  mom30s=%s  whale=%s",
                symbol, slug[-16:], window_elapsed,
                sig.price_drift, sig.ob_imbalance,
                f"{cvd:+.2f}" if cvd is not None else "n/a",
                f"{macd:+.2f}" if macd is not None else "n/a",
                f"{liq:+.2f}" if liq is not None else "n/a",
                f"{funding:+.2f}" if funding is not None else "n/a",
                f"{clob_cvd:+.2f}" if clob_cvd is not None else "n/a",
                f"{pcr:+.2f}" if pcr is not None else "n/a",
                f"{tick_vel:+.2f}" if tick_vel is not None else "n/a",
                f"{trade_imb:+.2f}" if trade_imb is not None else "n/a",
                f"{oracle_lag:+.2f}" if oracle_lag is not None else "n/a",
                f"{hawkes:+.2f}" if hawkes is not None else "n/a",
                f"{mlofi:+.2f}" if mlofi is not None else "n/a",
                sig.score, sig.alpha, sig.direction,
                f"UP={model_prob_up:.2f}" if model_prob_up is not None else "n/a",
                f"{mom_30s*100:+.3f}%" if mom_30s is not None else "n/a",
                f"{'BUY' if whale_is_buy else 'SELL'} ${whale_notional/1000:.0f}k"
                if whale_notional > 0 else "none",
            )

            if sig.direction == "NO_TRADE":
                continue

            # ── Force-UP hour override ────────────────────────────────────
            # In structurally bullish hours (e.g. 8am KST = UTC 23), DOWN
            # predictions are wrong ~73% of the time.  Flip DOWN → UP while
            # keeping all downstream filters (band, score cap, etc.) intact.
            if _hr in FORCE_UP_HOURS_UTC and sig.direction == "DOWN":
                log.info(
                    "%s  %s  FORCE_UP_HOUR UTC=%02d — overriding DOWN → UP "
                    "(8am KST structural edge: 86.2%% UP rate n=94)",
                    symbol, slug[-16:], _hr,
                )
                sig = FlowSignal(
                    direction="UP",
                    score=abs(sig.score),
                    price_drift=sig.price_drift,
                    ob_imbalance=sig.ob_imbalance,
                    cvd_score=sig.cvd_score,
                    macd_score=sig.macd_score,
                    current_price=sig.current_price,
                    alpha=sig.alpha,
                    liq_score=sig.liq_score,
                    funding_score=sig.funding_score,
                    clob_cvd_score=sig.clob_cvd_score,
                    exchange_div_score=sig.exchange_div_score,
                    pcr_score=sig.pcr_score,
                    tick_velocity_score=sig.tick_velocity_score,
                    trade_imbalance_score=sig.trade_imbalance_score,
                    oracle_lag_score=sig.oracle_lag_score,
                    hawkes_score=sig.hawkes_score,
                    mlofi_score=sig.mlofi_score,
                )

            # ── Direction force filter ────────────────────────────────────
            # Allow restricting the loop to trade only in one direction (UP or DOWN).
            # Config param: force_direction = "" (both), "UP" (UP only), "DOWN" (DOWN only).
            # Used by 15m (DOWN only) and 1h (UP only) loops to optimize for their edges.
            FORCE_DIRECTION = _CFG.get("force_direction", "").strip().upper()
            if FORCE_DIRECTION and sig.direction != "NO_TRADE":
                if FORCE_DIRECTION not in ("UP", "DOWN"):
                    log.warning("Invalid force_direction config: %s — ignoring", FORCE_DIRECTION)
                elif sig.direction != FORCE_DIRECTION:
                    log.info(
                        "%s  %s  DIRECTION FORCE — %s rejected (forced to %s only)",
                        symbol, slug[-16:], sig.direction, FORCE_DIRECTION,
                    )
                    continue

            # ── Score cap filter ──────────────────────────────────────────
            # High |score| means the crowd has fully priced the outcome in →
            # fills are very expensive and edge disappears (signal_log analysis:
            # score >0.65 → -3% to -11% edge at avg fill 0.93+).
            # "Honest uncertainty" zone: [SIGNAL_THRESHOLD, MAX_SIGNAL_SCORE).
            if abs(sig.score) >= MAX_SIGNAL_SCORE:
                log.info(
                    "%s  %s  SCORE CAP — |score|=%.3f >= %.2f (crowd already certain — skip)",
                    symbol, slug[-16:], abs(sig.score), MAX_SIGNAL_SCORE,
                )
                continue

            # ── Chop filter — DISABLED 2026-04-15 ────────────────────────
            # Removed: crypto_cache analysis showed chop-proxy executed trades
            # had WR=79.8% vs 75.0% for away-mid trades — filter was blocking
            # profitable signals. With CLOB band [0.72-0.80] already proving
            # crowd conviction via fill price, Gamma price near 0.5 is just lag.

            # ── 15-minute spot trend filter ────────────────────────────────────────────
            spot_trend = _get_spot_trend(symbol, log)
            if SPOT_TREND_FILTER and spot_trend != "NEUTRAL":
                log.info("TREND FILTER  %s  %s  trend=%s — skip", symbol, slug[-16:], spot_trend)
                continue

            # ── mom_30s directional alignment gate (2026-05-09) ──────────────
            # signal_log analysis (n=7,239 last 14d): when sig.direction opposes
            # the last-30s Binance return by ≥ 0.0001 (0.01%), WR collapses:
            #   YES with mom_30s ≤ -0.0001: 22-36% WR
            #   NO  with mom_30s ≥ +0.0001: 28-40% WR
            # Skip these worst-quadrant trades. mom_30s is also weighted into the
            # score itself (W_MOM_30S=0.20), this gate catches the residual where
            # score-magnitude survives but direction directly opposes mom_30s.
            MOM_ALIGN_TOL = 0.0001  # 0.01% — below this is noise, no gate
            if mom_30s is not None and abs(mom_30s) >= MOM_ALIGN_TOL:
                if sig.direction == "UP" and mom_30s < -MOM_ALIGN_TOL:
                    log.info(
                        "MOM ALIGN SKIP  %s  %s  signal=UP but mom_30s=%+.4f%% (price falling) — skip",
                        symbol, slug[-16:], mom_30s * 100,
                    )
                    continue
                if sig.direction == "DOWN" and mom_30s > MOM_ALIGN_TOL:
                    log.info(
                        "MOM ALIGN SKIP  %s  %s  signal=DOWN but mom_30s=%+.4f%% (price rising) — skip",
                        symbol, slug[-16:], mom_30s * 100,
                    )
                    continue

            # ── Binance live price confirmation — DISABLED 2026-04-27 filter removal test ─
            # BINANCE_1M_CONFIRM_PCT set to 0.0, so this filter is disabled.
            # Allow trades without spot price confirmation.
            # if BINANCE_1M_CONFIRM_PCT > 0:  [DISABLED — threshold is 0.0]
            #     ... confirmation check [DISABLED]

            # ── YES direction — ENABLED 2026-04-27 filter removal test ────────
            # Allow UP (YES) bets. Historical analysis blocked these, but
            # enable to test if signal quality without filters is better.

            is_up        = sig.direction == "UP"
            market_price = current_price if is_up else (1.0 - current_price)
            bet_size     = _size_bet(sig.score, market_price, vp.available_usdc)

            if bet_size < MIN_BET_ABS:
                log.info("Bet too small (%.2f) — skip", bet_size)
                continue

            # ── Spread measurement ────────────────────────────────────────
            # real_fill   = CLOB ask = what we'd actually pay
            # ba_spread   = min(ask) - max(bid) = pure market-maker cost
            # clob_vs_gamma = real_fill - Gamma_price (information gap)
            bet_token = up_token if is_up else down_token
            # Try WS cache first (sub-second fresh); fall back to REST on miss.
            clob_ask, ba_spread = _clob_feed.get_fill_price(bet_token)
            _spread_src = "WS"
            if clob_ask is None:
                clob_ask, ba_spread = _get_clob_spread(bet_token)
                _spread_src = "REST"
            log.debug("SPREAD src=%s  token=%s...", _spread_src, bet_token[:8])
            if clob_ask is not None:
                clob_vs_gamma = clob_ask - market_price
                log.info(
                    "SPREAD[%s]  %s  gamma=%.4f  clob_fill=%.4f  "
                    "clob_vs_gamma=%+.4f  ba_spread=%.4f",
                    _spread_src, symbol, market_price, clob_ask,
                    clob_vs_gamma, ba_spread or 0.0,
                )
                # ── Fade mode: flip direction when crowd is over-committed ──────
                # Triggers when: score is extreme, late in window, signal fill genuinely expensive.
                # Buys the OPPOSITE (cheap) token at small bet — asymmetric payout.
                fade_triggered = False
                if (FADE_MODE_ENABLED
                        and abs(sig.score) >= FADE_SCORE_THRESHOLD
                        and window_elapsed >= FADE_MIN_WINDOW_ELAPSED
                        and clob_ask >= FADE_MIN_SIGNAL_FILL):
                    _orig_dir   = "UP" if is_up else "DOWN"
                    _orig_ask   = clob_ask
                    is_up        = not is_up
                    bet_token    = up_token if is_up else down_token
                    market_price = 1.0 - market_price
                    _f_ask, _f_spread = _clob_feed.get_fill_price(bet_token)
                    if _f_ask is None:
                        _f_ask, _f_spread = _get_clob_spread(bet_token)
                    if _f_ask is None:
                        log.info(
                            "FADE  %s  %s  cannot price fade token — skip",
                            symbol, slug[-16:],
                        )
                        continue
                    clob_ask      = _f_ask
                    ba_spread     = _f_spread
                    clob_vs_gamma = clob_ask - market_price
                    log.info(
                        "FADE FLIP  %s  %s  score=%+.3f  signal=%s@%.4f → bet=%s@%.4f  elapsed=%.0fs",
                        symbol, slug[-16:], sig.score, _orig_dir, _orig_ask,
                        "UP" if is_up else "DOWN", clob_ask, window_elapsed,
                    )
                    fade_triggered = True

                # CLOB price band filter — RE-ENABLED to match yaml [0.72–0.80].
                # Out-of-band fills are logged to ghost files for future EV analysis.
                # Fade trades use FADE_BAND instead (cheap side, [0.10–0.30] by default).
                if fade_triggered:
                    if not (FADE_BAND_MIN <= clob_ask <= FADE_BAND_MAX):
                        log.info(
                            "FADE BAND SKIP  %s  %s  fade_fill=%.4f outside [%.2f-%.2f]",
                            symbol, slug[-16:], clob_ask, FADE_BAND_MIN, FADE_BAND_MAX,
                        )
                        continue
                elif not _in_clob_band(clob_ask):
                    _signal_dir_label = "YES" if is_up else "NO"
                    if GHOST_BAND_MIN <= clob_ask < MIN_CLOB_PRICE:
                        _log_ghost_trade(
                            market_id, slug, symbol, _signal_dir_label, clob_ask, log,
                            band_label="lower",
                            score=sig.score, price_drift=sig.price_drift,
                            ob_imbalance=sig.ob_imbalance, alpha=sig.alpha,
                            window_elapsed=window_elapsed, spot_trend=spot_trend,
                            gamma_price=market_price,
                        )
                    elif MAX_CLOB_PRICE <= clob_ask <= GHOST_UPPER_MAX:
                        _log_ghost_trade(
                            market_id, slug, symbol, _signal_dir_label, clob_ask, log,
                            log_path=_GHOST_UPPER_LOG, band_label="upper",
                            score=sig.score, price_drift=sig.price_drift,
                            ob_imbalance=sig.ob_imbalance, alpha=sig.alpha,
                            window_elapsed=window_elapsed, spot_trend=spot_trend,
                            gamma_price=market_price,
                        )
                    log.info(
                        "BAND SKIP  %s  %s  fill=%.4f  band=[%.2f–%.2f]",
                        symbol, slug[-16:], clob_ask, MIN_CLOB_PRICE, MAX_CLOB_PRICE,
                    )
                    continue
            else:
                # No CLOB book — empty orderbook means we cannot verify price
                # or apply band filter. Skip rather than execute blind.
                log.info("%s  %s  CLOB unavailable — skip (empty book)", symbol, slug[-16:])
                continue

            # ── Contrarian flip ───────────────────────────────────────────
            # When CONTRARIAN_MODE is True, the signal-token band check (above)
            # acts as a trigger: crowd is over-committed at 0.72–0.90, so we
            # fade them by buying the OPPOSITE token at ~0.10–0.28 instead.
            # clob_ask / market_price / clob_vs_gamma are updated to the
            # contrarian token so all downstream sizing and logging are correct.
            if CONTRARIAN_MODE:
                _sig_direction_label = "UP" if is_up else "DOWN"
                is_up        = not is_up
                bet_token    = up_token if is_up else down_token
                market_price = 1.0 - market_price
                _c_ask, _c_spread = _clob_feed.get_fill_price(bet_token)
                if _c_ask is None:
                    _c_ask, _c_spread = _get_clob_spread(bet_token)
                if _c_ask is None:
                    log.info(
                        "CONTRARIAN  %s  %s  cannot price contrarian token — skip",
                        symbol, slug[-16:],
                    )
                    continue
                log.info(
                    "CONTRARIAN  %s  signal=%s@%.4f → bet=%s@%.4f",
                    symbol, _sig_direction_label, clob_ask,
                    "UP" if is_up else "DOWN", _c_ask,
                )
                clob_ask      = _c_ask
                ba_spread     = _c_spread
                clob_vs_gamma = clob_ask - market_price

            # ★ Upper band score floor — DISABLED 2026-04-27 filter removal test.
            # UPPER_BAND_MIN_SCORE set to 0.0, so this check is unreachable.
            # Allow all scores at all prices.
            # if UPPER_BAND_MIN_SCORE > 0.0 and clob_ask >= UPPER_BAND_START:  [DISABLED]
            #     ... skip logic [DISABLED]

            # Recompute bet_size now that clob_ask is confirmed, applying the
            # inverse-fill discount. The initial estimate (line ~1789) used
            # market_price (Gamma) because CLOB fill wasn't known yet.
            bet_size = _size_bet(sig.score, market_price, vp.available_usdc,
                                 fill_price=clob_ask)
            # Fade trades: hard cap at FADE_MAX_BET_ABS regardless of Kelly —
            # contrarian small bet, asymmetric payout already provides upside.
            if fade_triggered:
                bet_size = min(bet_size, FADE_MAX_BET_ABS)
            if bet_size < MIN_BET_ABS:
                log.info("Bet too small after fill discount (%.2f) — skip", bet_size)
                continue

            # ── ML entry filter ────────────────────────────────────────────
            _pred_epnl: float | None = None
            if _ML_FILTER is not None:
                _fv = _build_ml_feature_vec(sig, clob_ask, clob_vs_gamma, now, symbol)
                if _ml_is_epnl and ML_EPNL_THRESHOLD is not None:
                    _model_min = _ML_FILTER.get("min_fill", 0.0)
                    _model_max = _ML_FILTER.get("max_fill", 1.0)
                    if not (_model_min <= clob_ask <= _model_max):
                        log.info(
                            "ML_EV OOB SKIP  %s  %s  fill=%.4f outside training range [%.2f–%.2f]",
                            symbol, slug[-16:], clob_ask, _model_min, _model_max,
                        )
                        continue
                    _pred_epnl = float(_ML_FILTER["model"].predict([_fv])[0])
                    if _pred_epnl < ML_EPNL_THRESHOLD:
                        log.info(
                            "ML_EV SKIP  %s  %s  epnl=%.3f < min=%.3f",
                            symbol, slug[-16:], _pred_epnl, ML_EPNL_THRESHOLD,
                        )
                        continue
                    log.info("ML_EV PASS  %s  %s  epnl=%.3f", symbol, slug[-16:], _pred_epnl)
                    # Recompute bet using EV-based Kelly (fill baked in, skips inverse-fill discount)
                    bet_size = _size_bet(sig.score, market_price, vp.available_usdc,
                                         fill_price=clob_ask, pred_epnl=_pred_epnl)
                    if bet_size < MIN_BET_ABS:
                        log.info("Bet too small after EV sizing (%.2f) — skip", bet_size)
                        continue
                elif not _ml_is_epnl and ML_FILTER_THRESHOLD > 0.0:
                    _prob = float(_ML_FILTER["model"].predict_proba([_fv])[0][1])
                    if _prob < ML_FILTER_THRESHOLD:
                        log.info(
                            "ML_FILTER SKIP  %s  %s  prob=%.3f < threshold=%.2f",
                            symbol, slug[-16:], _prob, ML_FILTER_THRESHOLD,
                        )
                        continue
                    log.info("ML_FILTER PASS  %s  %s  prob=%.3f", symbol, slug[-16:], _prob)
                    bet_size = _size_bet(sig.score, market_price, vp.available_usdc,
                                         ml_prob=_prob, fill_price=clob_ask)

            # ── Cross-asset pairs signal ──────────────────────────────────
            # When BTC/ETH/SOL diverge in 60s return, the lagging asset tends
            # to follow the leader within minutes (beta momentum transfer).
            # E.g. BTC +0.4% / ETH +0.05% in 60s → predict UP on ETH.
            # Always logged so we accumulate signal-log correlation data.
            # Gating (block/boost) is only active when PAIRS_FILTER_ENABLED.
            if not CONTRARIAN_MODE:
                pairs = get_pairs_signal_cached(
                    symbol, _live_feed,
                    window_sec=PAIRS_WINDOW_SEC,
                    min_divergence=PAIRS_MIN_DIVERGENCE,
                )
                flow_dir = "UP" if is_up else "DOWN"
                if pairs.direction != "NEUTRAL":
                    log.info(
                        "PAIRS  %s  %s  leader=%s  div=%+.4f%%  pairs_dir=%s  flow_dir=%s  conf=%.2f",
                        symbol, slug[-16:], pairs.leader,
                        pairs.divergence * 100, pairs.direction, flow_dir,
                        pairs.confidence,
                    )
                    if PAIRS_FILTER_ENABLED:
                        if pairs.direction == flow_dir:
                            _boost = 1.0 + (PAIRS_AGREE_BOOST - 1.0) * pairs.confidence
                            bet_size = min(bet_size * _boost, MAX_BET_ABS)
                            log.info(
                                "PAIRS AGREE  %s  boost=%.3f×  bet=%.2f",
                                symbol, _boost, bet_size,
                            )
                        elif PAIRS_DISAGREE_BLOCK:
                            log.info(
                                "PAIRS DISAGREE BLOCK  %s  pairs=%s vs flow=%s",
                                symbol, pairs.direction, flow_dir,
                            )
                            continue

            # ── Liquidation cascade signal ─────────────────────────────────
            # Large forced-liquidation cascades on Binance perpetuals (>$10M in
            # 15s) continue in the same direction ~70% of the time for 2-5 min.
            # SELL-side liq (longs blown out) → price DOWN; BUY-side (shorts
            # squeezed) → price UP. Valid for CASCADE_DECAY_SEC (120s).
            # Always logged; gating only active when CASCADE_FILTER_ENABLED.
            if not CONTRARIAN_MODE:
                flow_dir_cascade = "UP" if is_up else "DOWN"
                casc = _liq_cascade.get_signal(symbol)
                if casc.direction != "NEUTRAL":
                    log.info(
                        "CASCADE  %s  %s  dir=%s  notional=$%.1fM  age=%.0fs"
                        "  conf=%.2f  flow=%s",
                        symbol, slug[-16:], casc.direction,
                        casc.notional_usd / 1e6, casc.age_sec,
                        casc.confidence, flow_dir_cascade,
                    )
                    if CASCADE_FILTER_ENABLED:
                        if casc.direction == flow_dir_cascade:
                            _casc_boost = 1.0 + (CASCADE_AGREE_BOOST - 1.0) * casc.confidence
                            bet_size = min(bet_size * _casc_boost, MAX_BET_ABS)
                            log.info(
                                "CASCADE AGREE  %s  boost=%.3f×  bet=%.2f",
                                symbol, _casc_boost, bet_size,
                            )
                        elif CASCADE_DISAGREE_BLOCK:
                            log.info(
                                "CASCADE DISAGREE BLOCK  %s  cascade=%s vs flow=%s"
                                "  notional=$%.1fM",
                                symbol, casc.direction, flow_dir_cascade,
                                casc.notional_usd / 1e6,
                            )
                            continue

            # ── Contrarian flip filter ────────────────────────────────────
            # When fill price is extreme, market strongly disagrees with signal.
            # Flip direction to bet WITH the market instead of AGAINST it.
            # Extreme thresholds: < 0.15 (1-15% outcome) or > 0.85 (85-99% outcome)
            # Gated by CONTRARIAN_MODE — disabled when running momentum-follow strategy.
            flip_direction = False
            if CONTRARIAN_MODE and clob_ask < 0.15:
                log.info(
                    "FLIP CONTRARIAN  %s  fill=%.4f too cheap (market says %d%% unlikely) "
                    "→ flip %s to %s",
                    symbol, clob_ask, int(clob_ask*100),
                    "YES→NO" if is_up else "NO→YES",
                    "NO" if is_up else "YES",
                )
                is_up = not is_up
                market_price = 1.0 - market_price  # recalculate for flipped direction
                flip_direction = True
            elif CONTRARIAN_MODE and clob_ask > 0.85:
                log.info(
                    "FLIP CONTRARIAN  %s  fill=%.4f too expensive (market says %d%% likely) "
                    "→ flip %s to %s",
                    symbol, clob_ask, int(clob_ask*100),
                    "YES→NO" if is_up else "NO→YES",
                    "NO" if is_up else "YES",
                )
                is_up = not is_up
                market_price = 1.0 - market_price  # recalculate for flipped direction
                flip_direction = True

            # ── Execute order ─────────────────────────────────────────────
            outcome = "YES" if is_up else "NO"

            # If direction was flipped, update bet_token to the opposite token
            if flip_direction:
                bet_token = up_token if is_up else down_token

            # Dispatch through ExecutionBackend — VirtualBackend returns a
            # synthetic fill instantly; LiveBackend places a real CLOB order.
            # M1: pass clob_ask as price_hint (already fetched above) so
            #     LiveBackend skips a redundant CLOB book fetch.
            # B1: pass band limits so re-quotes are also band-checked inside
            #     LiveBackend (prevents stale price from bypassing the filter).
            from infra.types import ClobTokenId, ConditionId
            result = backend.place_order(
                ClobTokenId(bet_token),
                bet_size,
                ConditionId(market_id),
                log,
                price_hint=clob_ask,
                band_min=MIN_CLOB_PRICE,
                band_max=MAX_CLOB_PRICE,
            )
            if result["status"] != "FILLED":
                log.warning(
                    "ORDER NO_FILL  %s  %s  reason=%s",
                    symbol, outcome, result["reason"],
                )
                continue
            actual_fill_price = result["fill_price"]
            actual_bet_size   = result["filled_usdc"]
            if _is_live:
                log.info(
                    "REAL FILL  %s  %s  order_id=%s  fill=%.4f  usdc=$%.4f",
                    symbol, outcome, result["order_id"],
                    actual_fill_price, actual_bet_size,
                )

            # Deduct Polymarket taker fee (FEE_RATE) in addition to the bet.
            # In virtual mode this simulates real costs; in real mode this
            # accounts for the fee already taken by the platform.
            fee = round(actual_bet_size * FEE_RATE, 4)

            new_pos = VirtualPosition(
                market_id       = market_id,
                title           = f"{symbol} Up or Down ({slug[-16:]})",
                direction       = outcome,
                size_usdc       = actual_bet_size,
                fill_price      = actual_fill_price,
                fill_time       = now,
                category        = "crypto",
                bought_token_id = bet_token,
            )
            vp.positions.append(new_pos)
            vp.available_usdc -= actual_bet_size + fee
            vp.mark_updated()
            save_virtual_portfolio(vp, _state_path)
            traded_markets.add(market_id)

            _append_cache(market_id, symbol, slug, sig, open_price,
                          actual_bet_size, "UP" if is_up else "DOWN", ask_price=clob_ask)

            # Log executed signal to signal_log.jsonl for unified analysis
            _log_signal(
                market_id, slug, symbol, outcome, actual_fill_price,
                executed=True, skip_reason="",
                score=sig.score, price_drift=sig.price_drift,
                ob_imbalance=sig.ob_imbalance, alpha=sig.alpha,
                window_elapsed=window_elapsed, spot_trend=spot_trend,
                gamma_price=market_price, stake=actual_bet_size, log=log,
                mom_30s=mom_30s, whale_notional=whale_notional,
                whale_is_buy=whale_is_buy,
            )

            log.info(
                "FILL  %s  %s  size=$%.2f  price=%.3f  fee=$%.2f  "
                "drift=%+.3f  score=%+.2f  elapsed=%.0fs",
                symbol, outcome, actual_bet_size, actual_fill_price, fee,
                sig.price_drift, sig.score, window_elapsed,
            )

        elapsed    = time.monotonic() - cycle_start
        sleep_for  = max(0.0, LOOP_INTERVAL - elapsed)
        time.sleep(sleep_for)

    log.info("Crypto flow loop stopped.")
