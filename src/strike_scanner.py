"""
Crypto-strike lottery scanner — entry point.

Scans Polymarket Gamma for "Will <asset> reach $X by <date>?" markets and bets
small flat tickets on long-shots that either:
  (1) have a Black-Scholes-style implied probability above the market price by
      >= edge_threshold, OR
  (2) become "achievable" right after a momentum spike (burst mode).

Run in its own screen session:
    screen -dmS strike bash -c \\
      'source .venv/bin/activate && python src/strike_scanner.py >> logs/strike.log 2>&1'

Real-money guardrails:
  - Per-ticket spend capped at ticket_size_usdc (default $1).
  - Daily spend capped at daily_spend_cap (default $10).
  - One bet per market_id (de-duplicated via state file).
  - Wallet balance preflight before every order.
  - VIRTUAL_MODE env var paper-trades without touching the wallet.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
from pathlib import Path
from typing import Optional

import httpx
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
_FILE  = Path(__file__).resolve()
_SRC   = _FILE.parent
_ROOT  = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load .env so KEY/FUNDER are visible to execution.order._get_clob_client
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    log = logging.getLogger("strike")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    fmt.converter = time.gmtime
    log_dir = _ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "strike.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)
    # No StreamHandler — when run inside a screen with >> redirect, the
    # FileHandler already captures everything; a second handler doubles every line.
    log.propagate = False
    # Suppress py_clob_client_v2's noisy "Could not create api key" error log —
    # it fires on every CLOB client call when an API key already exists, but the
    # underlying call succeeds via fallback. Pure noise.
    for name in ("py_clob_client_v2",
                 "py_clob_client_v2.http_helpers.helpers"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    return log


# ── Config ────────────────────────────────────────────────────────────────────
def _load_cfg() -> dict:
    path = _ROOT / "config" / "strike_params.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_CFG = _load_cfg()

LOOP_INTERVAL          = int(_CFG.get("loop_interval", 90))
MARKET_CACHE_TTL       = int(_CFG.get("market_cache_ttl", 300))
MIN_VOLUME_24H         = float(_CFG.get("min_volume_24h", 5000))
MIN_TICKET_PRICE       = float(_CFG.get("min_ticket_price", 0.02))
MAX_TICKET_PRICE       = float(_CFG.get("max_ticket_price", 0.30))
TICKET_SIZE_USDC       = float(_CFG.get("ticket_size_usdc", 1.0))
DAILY_SPEND_CAP        = float(_CFG.get("daily_spend_cap", 10.0))
EDGE_THRESHOLD         = float(_CFG.get("edge_threshold", 0.03))
MIN_HOURS_TO_CLOSE     = float(_CFG.get("min_hours_to_close", 0.5))
MAX_HOURS_TO_CLOSE     = float(_CFG.get("max_hours_to_close", 168))
VOL_TABLE = {
    "BTC": float(_CFG.get("volatility_btc", 0.60)),
    "ETH": float(_CFG.get("volatility_eth", 0.80)),
    "SOL": float(_CFG.get("volatility_sol", 1.00)),
    "XRP": float(_CFG.get("volatility_xrp", 0.80)),
    "DOGE": float(_CFG.get("volatility_doge", 1.20)),
}
BURST_ENABLED            = bool(_CFG.get("burst_enabled", True))
BURST_LOOKBACK_MIN       = int(_CFG.get("burst_lookback_min", 30))
BURST_MIN_MOVE_PCT       = float(_CFG.get("burst_min_move_pct", 0.015))
BURST_MAX_STRIKE_GAP     = float(_CFG.get("burst_max_strike_gap", 0.04))

VIRTUAL_MODE = (
    os.getenv("STRIKE_VIRTUAL_MODE", str(_CFG.get("virtual_mode", False))).lower()
    in ("true", "1", "yes")
)

# ── State ─────────────────────────────────────────────────────────────────────
_STATE_PATH = _ROOT / "data" / "strike_state.json"
_DEFAULT_STATE = {
    "tickets": [],          # list of {market_id, slug, asset, strike, direction,
                            #          end_iso, fill_price, ticket_usd, source,
                            #          created_iso, settled, payout_usd}
    "executed_ids": [],     # list of market_id we've already bet on (de-dup)
    "daily_spend": {},      # {YYYY-MM-DD: usd_spent_today}
}

def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return dict(_DEFAULT_STATE, tickets=[], executed_ids=[], daily_spend={})
    try:
        s = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        for k, v in _DEFAULT_STATE.items():
            s.setdefault(k, v if not isinstance(v, (list, dict)) else type(v)())
        return s
    except Exception:
        return dict(_DEFAULT_STATE, tickets=[], executed_ids=[], daily_spend={})

def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(_STATE_PATH)

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _spend_today(state: dict) -> float:
    return float(state.get("daily_spend", {}).get(_today_utc(), 0.0))

def _record_spend(state: dict, usd: float) -> None:
    d = state.setdefault("daily_spend", {})
    d[_today_utc()] = float(d.get(_today_utc(), 0.0)) + usd


# ── HTTP clients ──────────────────────────────────────────────────────────────
_HTTP = httpx.Client(timeout=10.0, headers={"User-Agent": "strike-scanner/1.0"})
_GAMMA_BASE   = "https://gamma-api.polymarket.com"
_BINANCE_BASE = "https://api.binance.com"


def _binance_price(symbol: str) -> Optional[float]:
    try:
        r = _HTTP.get(f"{_BINANCE_BASE}/api/v3/ticker/price",
                      params={"symbol": f"{symbol}USDT"})
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def _binance_klines(symbol: str, interval: str = "1m", limit: int = 60) -> list:
    """Return list of close prices for the last N intervals."""
    try:
        r = _HTTP.get(f"{_BINANCE_BASE}/api/v3/klines",
                      params={"symbol": f"{symbol}USDT", "interval": interval,
                              "limit": limit})
        r.raise_for_status()
        return [float(k[4]) for k in r.json()]
    except Exception:
        return []


# ── Strike-market parsing ─────────────────────────────────────────────────────
# Patterns we recognise:
#   "Will Bitcoin reach $X in May?"
#   "Will the price of Bitcoin be above $X on May 7?"
#   "Will Bitcoin reach $X May 4-10?"
#   "Will Bitcoin be below $X by July 1?"
_ASSET_MAP = {
    "BTC": "BTC", "BITCOIN": "BTC",
    "ETH": "ETH", "ETHEREUM": "ETH",
    "SOL": "SOL", "SOLANA": "SOL",
    "XRP": "XRP",
    "DOGE": "DOGE", "DOGECOIN": "DOGE",
}
_DIR_KEYWORDS_ABOVE = ("above", "reach", "exceed", "hit", "over", "cross")
_DIR_KEYWORDS_BELOW = ("below", "under", "dip to", "drop to")

_STRIKE_RE = re.compile(
    r"(\$?\s*)([\d,]+(?:\.\d+)?)\s*([kKmM]?)",
)


def _parse_market(m: dict) -> Optional[dict]:
    """Extract (asset, strike, direction) from a Gamma market, or None."""
    q = (m.get("question") or "").strip()
    if not q:
        return None
    qu = q.upper()
    asset = next((v for k, v in _ASSET_MAP.items() if k in qu), None)
    if asset is None:
        return None
    direction = None
    ql = q.lower()
    if any(k in ql for k in _DIR_KEYWORDS_BELOW):
        direction = "below"
    elif any(k in ql for k in _DIR_KEYWORDS_ABOVE):
        direction = "above"
    if direction is None:
        return None
    # find strike $X — first $-prefixed number wins
    found = None
    for match in re.finditer(r"\$\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)", q):
        amount_s, suffix = match.group(1).replace(",", ""), match.group(2)
        try:
            v = float(amount_s)
        except Exception:
            continue
        if suffix.lower() == "k":
            v *= 1_000
        elif suffix.lower() == "m":
            v *= 1_000_000
        found = v
        break
    if found is None:
        return None
    return {"asset": asset, "strike": found, "direction": direction}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _digital_prob(current: float, strike: float, direction: str,
                  vol_annual: float, hours_to_close: float) -> float:
    """
    Black-Scholes-style probability of finishing above/below strike at expiry.
    Drift = 0 (martingale assumption). Vol scales with sqrt(T).
    """
    if current <= 0 or strike <= 0 or hours_to_close <= 0:
        return 0.0
    T_years = hours_to_close / (24.0 * 365.0)
    sigma_T = vol_annual * math.sqrt(max(T_years, 1.0 / (24.0 * 365.0 * 60.0)))
    if sigma_T <= 1e-9:
        return 1.0 if (direction == "above" and current > strike) or \
                      (direction == "below" and current < strike) else 0.0
    d2 = (math.log(current / strike) - 0.5 * sigma_T * sigma_T) / sigma_T
    if direction == "above":
        return _normal_cdf(d2)
    return 1.0 - _normal_cdf(d2)


# ── ML model probability (shadow mode) ────────────────────────────────────────
# Trained on Binance historical OHLCV via scripts/train_strike_model.py.
# As of 2026-05-08 the model is logged side-by-side with B-S but NOT used for
# firing decisions — its log-loss on holdout is worse than B-S in the high-ask
# zone (overconfident at predicted prob > 0.30). Watching shadow output for
# regime drift before considering swap.
_MODEL_PATH = _ROOT / "models" / "strike_model.pkl"
_MODEL_PAYLOAD = None
_MODEL_LOAD_FAILED = False
_VOL_FEAT_CACHE: dict = {}      # asset → (timestamp, features_dict)
_VOL_FEAT_TTL = 60.0


def _get_model():
    """Lazy-load the joblib payload once. Returns None if unavailable."""
    global _MODEL_PAYLOAD, _MODEL_LOAD_FAILED
    if _MODEL_LOAD_FAILED:
        return None
    if _MODEL_PAYLOAD is not None:
        return _MODEL_PAYLOAD
    if not _MODEL_PATH.exists():
        _MODEL_LOAD_FAILED = True
        return None
    try:
        import joblib  # local import — only needed if model exists
        _MODEL_PAYLOAD = joblib.load(_MODEL_PATH)
        return _MODEL_PAYLOAD
    except Exception:
        _MODEL_LOAD_FAILED = True
        return None


def _vol_features(asset: str, log: logging.Logger) -> Optional[dict]:
    """Compute rv_1h/4h/24h/7d, recent returns, distance from 24h H/L, vol ratio
    from 7 days of Binance 1m klines. Cached per-asset for _VOL_FEAT_TTL seconds.
    """
    now = time.time()
    cached = _VOL_FEAT_CACHE.get(asset)
    if cached and now - cached[0] < _VOL_FEAT_TTL:
        return cached[1]

    bars: list = []
    end_ms = int(now * 1000)
    cursor = end_ms - 7 * 1440 * 60_000   # 7 days back
    while cursor < end_ms:
        try:
            r = _HTTP.get(f"{_BINANCE_BASE}/api/v3/klines",
                          params={"symbol": f"{asset}USDT", "interval": "1m",
                                  "startTime": cursor, "limit": 1000},
                          timeout=10)
            r.raise_for_status()
            batch = r.json()
        except Exception as exc:
            log.debug("vol-feat klines fetch failed for %s: %s", asset, exc)
            return None
        if not batch:
            break
        bars.extend(batch)
        cursor = batch[-1][0] + 60_000
    if len(bars) < 1440:
        return None

    closes = np.array([float(b[4]) for b in bars], dtype=np.float64)
    highs = np.array([float(b[2]) for b in bars], dtype=np.float64)
    lows = np.array([float(b[3]) for b in bars], dtype=np.float64)
    volumes = np.array([float(b[5]) for b in bars], dtype=np.float64)
    log_returns = np.diff(np.log(closes))
    ann = math.sqrt(60 * 24 * 365)

    n_lr = len(log_returns)
    feats = {
        "rv_1h":  float(np.std(log_returns[-60:])  * ann),
        "rv_4h":  float(np.std(log_returns[-240:]) * ann),
        "rv_24h": float(np.std(log_returns[-1440:]) * ann),
        "rv_7d":  float(np.std(log_returns[-min(10080, n_lr):]) * ann),
        "ret_1h":  float(np.log(closes[-1] / closes[-61]))   if len(closes) >= 61 else 0.0,
        "ret_4h":  float(np.log(closes[-1] / closes[-241]))  if len(closes) >= 241 else 0.0,
        "ret_24h": float(np.log(closes[-1] / closes[-1441])) if len(closes) >= 1441 else 0.0,
        "dist_from_high_24h": float((closes[-1] - np.max(highs[-1440:])) / closes[-1]),
        "dist_from_low_24h":  float((closes[-1] - np.min(lows[-1440:]))  / closes[-1]),
        "vol_ratio_1h_24h":   float(np.sum(volumes[-60:]) / max(np.sum(volumes[-1440:]) / 24, 1e-9)),
    }
    feats["vol_regime_4h_24h"] = feats["rv_4h"] / max(feats["rv_24h"], 1e-9)

    _VOL_FEAT_CACHE[asset] = (now, feats)
    return feats


def _model_prob(asset: str, current: float, strike: float, direction: str,
                hours_to_close: float, log: logging.Logger) -> Optional[float]:
    """Return calibrated probability from the strike model, or None on any failure
    (model missing, asset unsupported, klines unavailable, strike outside trained
    range). Always safe to call.

    Training distribution constraint: model only saw out-of-the-money strikes
    with |offset| >= 2%. Falls back to None outside that range so callers use B-S.
    """
    if asset not in ("BTC", "ETH", "SOL"):
        return None
    if current <= 0 or strike <= 0:
        return None
    # Reject in-the-money or near-money strikes (untrained territory)
    offset = strike / current - 1.0
    if direction == "above" and offset < 0.02:
        return None
    if direction == "below" and offset > -0.02:
        return None
    if hours_to_close < 1.0 or hours_to_close > 168.0:
        return None
    payload = _get_model()
    if payload is None:
        return None
    feats = _vol_features(asset, log)
    if feats is None:
        return None
    sym_btc = 1.0 if asset == "BTC" else 0.0
    sym_eth = 1.0 if asset == "ETH" else 0.0
    sym_sol = 1.0 if asset == "SOL" else 0.0
    is_above = 1.0 if direction == "above" else 0.0

    x = np.array([[
        math.log(strike / current),
        hours_to_close,
        feats["rv_1h"], feats["rv_4h"], feats["rv_24h"], feats["rv_7d"],
        feats["vol_regime_4h_24h"],
        feats["ret_1h"], feats["ret_4h"], feats["ret_24h"],
        feats["dist_from_high_24h"], feats["dist_from_low_24h"],
        feats["vol_ratio_1h_24h"],
        is_above,
        sym_btc, sym_eth, sym_sol,
    ]])
    try:
        raw = float(payload["model"].predict_proba(x)[0, 1])
        cal = float(payload["calibrator"].transform([raw])[0])
        return cal
    except Exception as exc:
        log.debug("model_prob failed for %s: %s", asset, exc)
        return None


# ── Market discovery ──────────────────────────────────────────────────────────
_market_cache: list = []
_market_cache_ts: float = 0.0


def _fetch_markets(log: logging.Logger) -> list:
    global _market_cache, _market_cache_ts
    if time.time() - _market_cache_ts < MARKET_CACHE_TTL and _market_cache:
        return _market_cache
    try:
        r = _HTTP.get(f"{_GAMMA_BASE}/markets",
                      params={"active": "true", "closed": "false",
                              "limit": 500, "order": "volume24hr",
                              "ascending": "false"},
                      timeout=15)
        r.raise_for_status()
        ms = r.json() or []
    except Exception as exc:
        log.warning("Gamma fetch failed: %s", exc)
        return _market_cache
    now_dt = datetime.now(timezone.utc)
    out = []
    for m in ms:
        end_s = m.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end_s.replace("Z", "+00:00"))
        except Exception:
            continue
        if end_dt <= now_dt:
            continue
        hours_to = (end_dt - now_dt).total_seconds() / 3600.0
        if hours_to < MIN_HOURS_TO_CLOSE or hours_to > MAX_HOURS_TO_CLOSE:
            continue
        if float(m.get("volume24hr") or 0) < MIN_VOLUME_24H:
            continue
        if not m.get("clobTokenIds"):
            continue
        # parse strike
        parsed = _parse_market(m)
        if parsed is None:
            continue
        m["_parsed"]      = parsed
        m["_hours_to"]    = hours_to
        out.append(m)
    _market_cache = out
    _market_cache_ts = time.time()
    log.info("Discovered %d strike-style markets (cache ttl=%ds)",
             len(out), MARKET_CACHE_TTL)
    return out


# ── Pricing ───────────────────────────────────────────────────────────────────
def _yes_token(m: dict) -> Optional[str]:
    ids = m.get("clobTokenIds")
    if isinstance(ids, str):
        try:
            ids = json.loads(ids)
        except Exception:
            return None
    if not ids:
        return None
    return str(ids[0])


def _yes_price(token_id: str) -> Optional[float]:
    """Return current best ask for the YES token. Real cost-to-buy."""
    try:
        r = _HTTP.get("https://clob.polymarket.com/book",
                      params={"token_id": token_id})
        r.raise_for_status()
        asks = r.json().get("asks") or []
        if not asks:
            return None
        return min(float(a["price"]) for a in asks)
    except Exception:
        return None


def _burst_signal(asset: str, log: logging.Logger) -> tuple[float, float]:
    """
    Return (current_price, recent_move_pct) over BURST_LOOKBACK_MIN minutes.
    move_pct is signed: positive = up, negative = down.
    """
    closes = _binance_klines(asset, "1m", BURST_LOOKBACK_MIN)
    if len(closes) < 2:
        cur = _binance_price(asset) or 0.0
        return cur, 0.0
    cur = closes[-1]
    return cur, (cur - closes[0]) / closes[0]


# ── Order placement (lazy import — only when live) ────────────────────────────
_cached_clob_client = None


def _get_client():
    global _cached_clob_client
    if _cached_clob_client is not None:
        return _cached_clob_client
    if VIRTUAL_MODE:
        return None
    from execution.order import _get_clob_client  # type: ignore
    _cached_clob_client = _get_clob_client()
    return _cached_clob_client


def _check_balance(log: logging.Logger) -> Optional[float]:
    if VIRTUAL_MODE:
        return 1000.0
    try:
        from crypto.execution import get_clob_balance  # type: ignore
        return get_clob_balance(log)
    except Exception as exc:
        log.warning("Balance check failed: %s", exc)
        return None


def _place_buy(token_id: str, price: float, ticket_usd: float,
               log: logging.Logger) -> Optional[dict]:
    """Place a GTC BUY at <price> for ticket_usd worth of tokens."""
    if VIRTUAL_MODE:
        log.info("[VIRTUAL] would buy token=%s... at %.4f for $%.2f",
                 token_id[:10], price, ticket_usd)
        return {"fill_price": price, "ticket_usd": ticket_usd, "virtual": True}
    client = _get_client()
    if client is None:
        log.warning("CLOB client unavailable — skip")
        return None
    try:
        from py_clob_client_v2.clob_types import OrderArgs, OrderType  # type: ignore
        from infra.http_client import call_with_timeout
        # Polymarket requires min $1 notional. Round UP and add 0.5% buffer
        # so size*price is comfortably above the minimum after their rounding.
        size = math.ceil(ticket_usd * 1.005 / price * 100) / 100
        if size <= 0:
            return None
        actual_usd = round(size * price, 4)
        args = OrderArgs(price=price, size=size, side="BUY",
                         token_id=token_id)
        signed = call_with_timeout(client.create_order, args)
        resp = call_with_timeout(client.post_order, signed, OrderType.GTC)
        if isinstance(resp, dict) and resp.get("success") in (True, "true"):
            log.info("ORDER OK token=%s... px=%.4f size=%.2f notional=$%.4f resp=%s",
                     token_id[:10], price, size, actual_usd,
                     resp.get("status", "?"))
            return {"fill_price": price, "ticket_usd": actual_usd,
                    "order_id": resp.get("orderID")}
        log.warning("Order rejected: %s", resp)
        return None
    except Exception as exc:
        log.warning("Order failed: %s", exc)
        return None


# ── Settlement ────────────────────────────────────────────────────────────────
def _settle_open_tickets(state: dict, log: logging.Logger) -> None:
    """Resolve tickets whose markets have closed."""
    now_dt = datetime.now(timezone.utc)
    changed = False
    for t in state.get("tickets", []):
        if t.get("settled"):
            continue
        try:
            end_dt = datetime.fromisoformat(t["end_iso"].replace("Z", "+00:00"))
        except Exception:
            continue
        if end_dt > now_dt:
            continue
        # market should be resolved — fetch outcome
        slug = t.get("slug") or ""
        try:
            r = _HTTP.get(f"{_GAMMA_BASE}/markets",
                          params={"slug": slug}, timeout=10)
            data = r.json()
            if isinstance(data, list) and data:
                m = data[0]
            else:
                m = data if isinstance(data, dict) else {}
            outcome_prices = m.get("outcomePrices")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            if not outcome_prices or len(outcome_prices) < 1:
                continue
            yes_outcome = float(outcome_prices[0])
            ticket_usd  = float(t.get("ticket_usd", 0.0))
            fill_price  = float(t.get("fill_price", 1.0))
            tokens      = ticket_usd / max(fill_price, 0.0001)
            payout      = tokens * yes_outcome
            t["settled"]      = True
            t["settled_iso"]  = now_dt.isoformat()
            t["yes_outcome"]  = yes_outcome
            t["payout_usd"]   = round(payout, 4)
            t["pnl_usd"]      = round(payout - ticket_usd, 4)
            changed = True
            log.info("SETTLE  %s  fill=%.4f outcome=%.2f payout=$%.2f pnl=$%+.2f",
                     slug[:60], fill_price, yes_outcome, payout, payout - ticket_usd)
        except Exception as exc:
            log.debug("Settle fetch failed for %s: %s", slug, exc)
    if changed:
        _save_state(state)


# ── Main scan logic ───────────────────────────────────────────────────────────
def _score_market(m: dict, current_prices: dict, log: logging.Logger
                  ) -> Optional[dict]:
    parsed = m["_parsed"]
    asset, strike, direction = parsed["asset"], parsed["strike"], parsed["direction"]
    cur = current_prices.get(asset)
    if cur is None:
        return None
    # already resolved by underlying?
    if direction == "above" and cur >= strike:
        return None       # in-the-money, will settle YES, no edge as a long-shot
    if direction == "below" and cur <= strike:
        return None
    hours_to = float(m.get("_hours_to") or 0.0)
    yes_token_id = _yes_token(m)
    if yes_token_id is None:
        return None
    yes_ask = _yes_price(yes_token_id)
    if yes_ask is None:
        return None
    if not (MIN_TICKET_PRICE <= yes_ask <= MAX_TICKET_PRICE):
        return None
    vol = VOL_TABLE.get(asset, 0.80)
    implied = _digital_prob(cur, strike, direction, vol, hours_to)
    edge_model = implied - yes_ask

    # ML model probability (SHADOW MODE — logged but not used for firing).
    # As of 2026-05-08 the model is overconfident at predicted prob > 0.30.
    # Watching shadow output before considering swap. See CLAUDE.md.
    ml_prob = _model_prob(asset, cur, strike, direction, hours_to, log)
    if ml_prob is not None:
        log.info("SHADOW %s %s ask=%.3f BS=%.3f ML=%.3f Δ=%+.3f hrs=%.1f  %s",
                 asset, direction, yes_ask, implied, ml_prob, ml_prob - implied,
                 hours_to, (m.get("question") or "")[:60])
    burst = False
    burst_move = 0.0
    if BURST_ENABLED:
        _, move = _burst_signal(asset, log)
        burst_move = move
        # Burst rule: if asset just moved by >= BURST_MIN_MOVE_PCT in the
        # favorable direction, and the strike is within BURST_MAX_STRIKE_GAP
        # of current → bet small (regardless of model edge).
        gap_pct = (strike - cur) / cur if direction == "above" else (cur - strike) / cur
        if (direction == "above" and move >= BURST_MIN_MOVE_PCT
            and 0 < gap_pct <= BURST_MAX_STRIKE_GAP):
            burst = True
        elif (direction == "below" and move <= -BURST_MIN_MOVE_PCT
              and 0 < gap_pct <= BURST_MAX_STRIKE_GAP):
            burst = True
    fire_model = edge_model >= EDGE_THRESHOLD
    if not (fire_model or burst):
        return None
    return {
        "market": m, "asset": asset, "strike": strike, "direction": direction,
        "yes_token_id": yes_token_id, "yes_ask": yes_ask, "current": cur,
        "implied": implied, "edge_model": edge_model, "hours_to": hours_to,
        "burst": burst, "burst_move": burst_move,
        "ml_prob": ml_prob,
        "source": "burst" if burst and not fire_model
                  else ("model" if fire_model else "burst+model"),
    }


def _scan_once(state: dict, log: logging.Logger) -> None:
    # Settle anything overdue first
    _settle_open_tickets(state, log)

    # Daily-spend gate
    spent = _spend_today(state)
    if spent >= DAILY_SPEND_CAP:
        log.info("Daily cap reached ($%.2f / $%.2f) — idle", spent, DAILY_SPEND_CAP)
        return

    markets = _fetch_markets(log)
    if not markets:
        return
    # Snapshot current prices once per scan
    assets_seen = {m["_parsed"]["asset"] for m in markets}
    cur_prices = {a: _binance_price(a) for a in assets_seen}
    cur_prices = {k: v for k, v in cur_prices.items() if v is not None}
    if not cur_prices:
        log.warning("No Binance prices — skipping scan")
        return

    executed_set = set(state.get("executed_ids", []))
    candidates = []
    for m in markets:
        if m.get("id") in executed_set or m.get("conditionId") in executed_set:
            continue
        scored = _score_market(m, cur_prices, log)
        if scored is None:
            continue
        candidates.append(scored)

    # Sort by best-edge first; break ties by shortest hours_to
    candidates.sort(key=lambda c: (-c["edge_model"], c["hours_to"]))
    log.info("Scan: %d markets / %d candidates / spent today $%.2f",
             len(markets), len(candidates), spent)

    # No balance check needed when there's nothing to fire on — saves a CLOB
    # round-trip (and its noisy api-key error) every empty scan.
    if not candidates:
        return

    bal = _check_balance(log)
    if bal is None or bal < 5.0:
        log.warning("Balance unavailable or too low ($%s) — skipping orders",
                    f"{bal:.2f}" if bal is not None else "?")
        return

    for c in candidates:
        if _spend_today(state) + TICKET_SIZE_USDC > DAILY_SPEND_CAP:
            log.info("Daily cap would be exceeded — stopping for today")
            break
        m = c["market"]
        market_id   = m.get("id") or m.get("conditionId")
        slug        = m.get("slug") or ""
        log.info(
            "FIRE [%s]  %s  ask=%.3f  implied=%.3f  edge=%+0.3f  burst=%s  "
            "%s strike=$%.0f cur=$%.0f T=%.1fh",
            c["source"], slug[:70], c["yes_ask"], c["implied"], c["edge_model"],
            "Y" if c["burst"] else "N",
            c["asset"], c["strike"], c["current"], c["hours_to"],
        )
        # Place
        result = _place_buy(c["yes_token_id"], c["yes_ask"],
                            TICKET_SIZE_USDC, log)
        if result is None:
            continue
        ticket = {
            "market_id":   market_id,
            "slug":        slug,
            "asset":       c["asset"],
            "strike":      c["strike"],
            "direction":   c["direction"],
            "end_iso":     m.get("endDate"),
            "fill_price":  result["fill_price"],
            "ticket_usd":  result["ticket_usd"],
            "source":      c["source"],
            "implied":     c["implied"],          # Black-Scholes prob (live pricer)
            "ml_prob":     c.get("ml_prob"),      # ★ shadow ML prob — for retrospective compare
            "edge_model":  c["edge_model"],
            "burst_move":  c.get("burst_move", 0.0),
            "current_at_entry": c["current"],
            "created_iso": datetime.now(timezone.utc).isoformat(),
            "settled":     False,
            "virtual":     bool(VIRTUAL_MODE or result.get("virtual")),
        }
        state.setdefault("tickets", []).append(ticket)
        state.setdefault("executed_ids", []).append(market_id)
        _record_spend(state, result["ticket_usd"])
        _save_state(state)


# ── Run loop ──────────────────────────────────────────────────────────────────
def run() -> None:
    log = _setup_logging()
    log.info("*** Strike scanner starting (mode=%s, ticket=$%.2f, daily_cap=$%.0f) ***",
             "VIRTUAL" if VIRTUAL_MODE else "LIVE",
             TICKET_SIZE_USDC, DAILY_SPEND_CAP)
    state = _load_state()
    while True:
        try:
            _scan_once(state, log)
        except KeyboardInterrupt:
            log.info("Strike scanner: keyboard interrupt — exiting")
            return
        except Exception as exc:
            log.exception("Scan iteration crashed: %s", exc)
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()
