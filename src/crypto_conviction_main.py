"""
Market Conviction Paper Trading Bot
=====================================

Entry point for simple "follow the crowd" strategy.
Discovers all crypto markets, reads implied probabilities, papers trades them.

Start:
  screen -S conviction
  source .venv/bin/activate
  python src/crypto_conviction_main.py >> logs/conviction.log 2>&1

Config: config/conviction_params.yaml (if you want to customize)
State: data/conviction_state.json (virtual portfolio)
Log: data/conviction_log.jsonl (all trades)
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    env_path = os.path.expanduser("~/poly-model/.env")
    if not os.path.exists(env_path):
        env_path = ".env"
    load_dotenv(env_path)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("conviction_main")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crypto.market_conviction import fetch_market, fetch_market_by_id, compute_conviction_signal, check_arb_opportunity, compute_market_skew
from crypto.clob_feed import clob_feed
from crypto.flow import _get_book as _flow_get_book  # WS-first + REST fallback + seed
from crypto.redeem import redeem_redeemable_positions

# Auto-redeem (AR) cadence: conviction trades fast (~30+ wins/h) so we sweep
# winning CTF tokens back to USDC every 20 min. The crypto bot's 12h sweep is
# too slow at this trade volume — without our own AR the wallet bleeds to ~$0
# in spendable USDC even while the bot is winning.
_AR_INTERVAL_SEC = 20 * 60

# Default to VIRTUAL mode (paper trading) unless explicitly set to false
_VIRTUAL_MODE = os.getenv("CONVICTION_VIRTUAL_MODE", "true").lower() != "false"


def _get_pending_ctf_value(wallet_addr: str, log) -> float:
    """Sum of (size × curPrice) for every CTF position in the wallet.

    Used by C2 so that cash → CTF-token conversions (the normal flow of every
    new entry) don't register as a loss. Includes winning positions (curPrice~1),
    losing/worthless (curPrice~0), and unresolved mid-life ones — exactly the
    mirror of what came out of `get_balance()` when we placed the order.
    Returns 0.0 on any API failure so C2 falls back to cash-only gracefully.
    """
    if not wallet_addr:
        return 0.0
    try:
        import requests
        url = "https://data-api.polymarket.com/positions"
        params = {"user": wallet_addr, "sizeThreshold": 0, "limit": 500, "offset": 0}
        total = 0.0
        for _ in range(10):  # 5000-position safety cap
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for p in batch:
                total += float(p.get("size", 0)) * float(p.get("curPrice", 0))
            if len(batch) < params["limit"]:
                break
            params["offset"] += params["limit"]
        return total
    except Exception as e:
        log.warning("CTF value fetch failed (C2 falls back to cash-only): %s", e)
        return 0.0


# 2026-05-13: restricted to mainstream after 1,036-trade analysis showed
#   Mainstream (BTC/ETH/SOL): +$14.35 / 523 trades / +2.8pp edge ✅
#   Mid-cap   (XRP/DOGE/BNB): -$36.41 / 515 trades / -7.1pp edge ❌
# Inversion rule didn't recover mid-cap (still -6.1pp inverted), so dropping entirely.
SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["5m", "15m", "1h"]
WINDOW_SECS = {"5m": 300, "15m": 900, "1h": 3600}

# Specific (symbol, timeframe) combinations to skip in LIVE mode only.
# Paper mode trades every mainstream combo so we can keep watching whether the
# blocked ones recover; live mode skips them to avoid -EV bleed.
# 2026-05-20: Only ETH 5m enabled live (BTC 15m / SOL 15m parked for now).
BLOCKED_SYMBOL_TF_LIVE = {("BTC", "5m"), ("BTC", "15m"), ("BTC", "1h"),
                          ("ETH", "15m"), ("ETH", "1h"),
                          ("SOL", "5m"), ("SOL", "15m"), ("SOL", "1h")}

# UTC hours to skip in LIVE mode. Set from paper-data analysis: UTC 11 (KST 20)
# was the worst-EV hour across BTC/ETH/SOL in both halves of a chronological
# train/test split — the one robust blacklist entry. Paper mode ignores this
# so we can keep grading the rule.
BLOCKED_HOURS_LIVE: set[int] = {11}

# ── Data-Collection Entry Config (2026-05-13) ──────────────────────────────
# New rule (replaces band filter):
#   conviction <  MIN_CONVICTION  → SKIP (wait — market may still enter band)
#   conviction >= MIN_CONVICTION  → ENTER once at first sight (one entry per market)
#   conviction >= 0.98            → already filtered by compute_market_skew
#
# Rationale: edge confirmed at conviction 0.52-0.54 (+$0.018/trade, 55.6% WR on 151
# trades). Above 0.54 is data-poor — small sample (21 trades in [0.54-0.56)) shows
# negative bias but n is too low. Goal of this phase: collect outcomes across the
# full price range so the optimal band/sizing can be re-fit from broader data.
# Re-entry is disabled: each market gets exactly one bet → one outcome data point.
STRATEGY_MODE = "SKEW"

# Enter when dominant side is in this range:
#   0.52 = market just started leaning (50/50 → 52/48)
#   0.55 = upper edge — anything past this is structurally negative EV (post-restart
#          data: skew 55-60% WR=42.9% -$0.15/trade; 60-65% WR=58.8% -$0.05/trade).
# Real-time WS observation via CLOBFeed catches every market exactly when it first
# crosses 0.52 instead of 5-10s later (poll latency).
MIN_SKEW = 0.52  # require at least some market bias
MAX_SKEW = 0.55  # stop entering when market is already strongly biased

# Per-combo entry bands for LIVE mode only (paper uses MIN_SKEW/MAX_SKEW globally
# so we keep collecting the full 0.52-0.55 distribution for re-fitting).
# Set from 2026-05-18 bucket analysis on 3,663 paper trades — each combo's
# profitable sub-band within the wider 0.52-0.55 range:
#   ETH 5m  [0.54-0.55): n=272, 59.6% WR, +$0.0317/trade EV
#   BTC 15m [0.52-0.53): n=128, 57.8% WR, +$0.0367/trade EV
#   SOL 15m [0.53-0.54): n=101, 58.4% WR, +$0.0251/trade EV
# Any (sym, tf) not in this dict and not in BLOCKED_SYMBOL_TF_LIVE falls back
# to the global MIN_SKEW/MAX_SKEW band.
LIVE_BANDS = {
    ("ETH", "5m"):  (0.54, 0.55),
    ("BTC", "15m"): (0.52, 0.53),
    ("SOL", "15m"): (0.53, 0.54),
}

# Per-tier asymmetric direction rule (2026-05-13 inversion analysis, n=256):
#   Mainstream (BTC/ETH/SOL, n=127): original direction +1.2pp edge ✅
#     → active books with real momentum signal, follow the crowd.
#   Mid-cap wide-gap (BNB/XRP/DOGE w/ entry-skew>0.015, n=115):
#     Original -16.6pp ❌  →  INVERTED +2.6pp ✅  swing +$21.71
#     → sparse books, dominant side is microstructure noise that mean-reverts.
#   Mid-cap tight-gap (n=19): original +12.7pp ✅  →  keep original
#     → tight spread = real direction signal even on mid-cap.
MAINSTREAM_SYMBOLS = {"BTC", "ETH", "SOL"}
SPREAD_GAP_INVERT_THRESHOLD = 0.015  # entry_price − skew above this on mid-cap → invert

# Loss-streak circuit breaker (live-only). After this many consecutive losses,
# pause new entries for the cooldown — almost certainly in an adversarial regime
# (sustained trend against our directional bets). Wait for it to pass.
# Counter resets to 0 on any win. Paper bot ignores this so we can measure impact.
LOSS_STREAK_THRESHOLD     = 3
LOSS_STREAK_COOLDOWN_SEC  = 20 * 60  # constant 20-min rest after each 3-loss streak


def _parse_token_ids(market: dict) -> list[str]:
    """Extract clobTokenIds as list of strings. Returns [] if missing/malformed."""
    raw = market.get("clobTokenIds", "[]")
    if isinstance(raw, str):
        try:
            ids = json.loads(raw)
        except Exception:
            return []
    else:
        ids = raw
    return [str(t) for t in ids] if ids else []


def _compute_clob_skew(market: dict) -> "dict | None":
    """
    Compute skew from live CLOB orderbook.

    Resolution order (matches flow.py::_get_book):
      1. WS cache via CLOBFeed (sub-second when active)
      2. REST /book fallback (ensures coverage for low-activity tokens that
         Polymarket WS never sends initial book events for) — seeds cache so
         future price_change deltas apply.

    Skew is derived from best ask of each side (the price we'd actually pay):
        up_skew = up_ask / (up_ask + down_ask)
    Returns None only when both WS and REST yield no asks.
    """
    tokens = _parse_token_ids(market)
    if len(tokens) < 2:
        return None
    up_token, down_token = tokens[0], tokens[1]

    _, up_asks   = _flow_get_book(up_token,   clob_feed)
    _, down_asks = _flow_get_book(down_token, clob_feed)
    if not up_asks or not down_asks:
        return None

    up_ask   = min(float(a["price"]) for a in up_asks)
    down_ask = min(float(a["price"]) for a in down_asks)

    # Skip near-resolved
    if up_ask >= 0.98 or up_ask <= 0.02 or down_ask >= 0.98 or down_ask <= 0.02:
        return None

    total = up_ask + down_ask
    if total <= 0:
        return None

    up_skew = up_ask / total
    direction = "UP" if up_skew > 0.5 else "DOWN"

    return {
        "up_skew":   up_skew,
        "down_skew": 1.0 - up_skew,
        "direction": direction,
        "yes_price": up_ask,
        "no_price":  down_ask,
        "total_cost": total,
        "symbol":    market.get("_symbol", "UNKNOWN"),
        "timeframe": market.get("_timeframe", "5m"),
        "market_id": market.get("id", ""),
        "slug":      market.get("slug", ""),
    }


def _window_starts_near_close(now_ts: int, window_secs: int, look_back: int = 3) -> list[int]:
    """Return window start timestamps for windows currently near close."""
    aligned = (now_ts // window_secs) * window_secs
    return [aligned - i * window_secs for i in range(look_back)]


def discover_markets(now_ts: int) -> list[dict]:
    """
    Discover active crypto markets using the crypto/flow.py discovery method.
    This ensures we get markets with proper clobTokenIds and symbol/timeframe data.
    """
    found = []
    seen = set()

    for timeframe in TIMEFRAMES:
        window_secs = WINDOW_SECS[timeframe]
        for ts in _window_starts_near_close(now_ts, window_secs, look_back=1):
            for sym in SYMBOLS:
                slug = f"{sym.lower()}-updown-{timeframe}-{ts}"
                m = fetch_market(slug)
                if m is None or m.get("id") in seen:
                    continue

                # Ensure market has clobTokenIds
                token_ids_raw = m.get("clobTokenIds")
                if not token_ids_raw:
                    continue
                try:
                    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
                    if len(token_ids) < 2:
                        continue
                except:
                    continue

                # Add symbol and timeframe if not already there
                if "_symbol" not in m:
                    m["_symbol"] = sym
                if "_timeframe" not in m:
                    m["_timeframe"] = timeframe

                seen.add(m["id"])
                found.append(m)

    return found


class ConvictionState:
    def __init__(self, state_path: str = "data/conviction_state.json"):
        self.state_path = state_path
        self.mode = "VIRTUAL"
        self.live_start_time = None
        self.arb_positions = {}
        self.closed_arb = []
        self.arb_pnl = 0.0
        self.load()

    def load(self):
        data = None
        if Path(self.state_path).exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
            except Exception as exc:
                # State file corrupt (e.g., truncated mid-write). Back it up
                # and start fresh rather than crashing.
                backup = self.state_path + ".corrupt"
                try:
                    Path(self.state_path).rename(backup)
                    log.warning("State file %s corrupt (%s); moved to %s, starting fresh",
                                self.state_path, exc, backup)
                except Exception:
                    log.warning("State file %s corrupt (%s); starting fresh", self.state_path, exc)
                data = None
        if data is not None:
            self.positions = data.get("positions", {})
            self.closed = data.get("closed_positions", [])
            self.pnl_total = data.get("pnl_total", 0.0)
            self.real_clob_balance = data.get("real_clob_balance", 0.0)
            self.start_of_day_balance = data.get("start_of_day_balance", 0.0)
            self.daily_loss = data.get("daily_loss", 0.0)
            self.mode = data.get("mode", "VIRTUAL")
            self.live_start_time = data.get("live_start_time", None)
            self.arb_positions = data.get("arb_positions", {})
            self.closed_arb = data.get("closed_arb", [])
            self.arb_pnl = data.get("arb_pnl", 0.0)
            self.consecutive_losses    = data.get("consecutive_losses", 0)
            self.circuit_breaker_until = data.get("circuit_breaker_until", 0.0)
        else:
            self.positions = {}
            self.closed = []
            self.pnl_total = 0.0
            self.real_clob_balance = 0.0
            self.start_of_day_balance = 0.0
            self.daily_loss = 0.0
            self.mode = "VIRTUAL"
            self.live_start_time = None
            self.arb_positions = {}
            self.closed_arb = []
            self.arb_pnl = 0.0
            self.consecutive_losses    = 0
            self.circuit_breaker_until = 0.0

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(
                {
                    "positions": self.positions,
                    "closed_positions": self.closed,
                    "pnl_total": self.pnl_total,
                    "real_clob_balance": self.real_clob_balance,
                    "start_of_day_balance": self.start_of_day_balance,
                    "daily_loss": self.daily_loss,
                    "mode": self.mode,
                    "live_start_time": self.live_start_time,
                    "arb_positions": self.arb_positions,
                    "closed_arb": self.closed_arb,
                    "arb_pnl": self.arb_pnl,
                    "consecutive_losses": self.consecutive_losses,
                    "circuit_breaker_until": self.circuit_breaker_until,
                },
                f,
            )

    def open_position(self, market_id: str, signal: dict):
        now = time.time()
        is_live = self.mode == "LIVE"
        self.positions[market_id] = {
            "symbol": signal["symbol"],
            "timeframe": signal["timeframe"],
            "direction": signal["direction"],
            "conviction": signal["conviction"],
            "entry_price": signal["entry_price"],
            "bet": signal["bet_size"],
            "opened_at": now,
            "is_live": is_live,
            "inverted": bool(signal.get("inverted", False)),
        }

    def close_position(self, market_id: str, resolved_direction: str, pnl: float):
        pos = self.positions.pop(market_id, None)
        if pos:
            pos["resolved_direction"] = resolved_direction
            pos["pnl"] = pnl
            pos["closed_at"] = time.time()
            # Inherit is_live flag if not already set
            if "is_live" not in pos:
                pos["is_live"] = self.mode == "LIVE"
            self.closed.append(pos)
            self.pnl_total += pnl

    def open_arb_position(self, market_id: str, arb: dict):
        now = time.time()
        is_live = self.mode == "LIVE"
        self.arb_positions[market_id] = {
            "symbol": arb["symbol"],
            "timeframe": arb["timeframe"],
            "yes_price": arb["yes_price"],
            "no_price": arb["no_price"],
            "total_cost": arb["total_cost"],
            "net_margin": arb["net_margin"],
            "bet": 1.0,
            "opened_at": now,
            "is_live": is_live,
        }

    def close_arb_position(self, market_id: str, pnl: float):
        pos = self.arb_positions.pop(market_id, None)
        if pos:
            pos["pnl"] = pnl
            pos["closed_at"] = time.time()
            # Inherit is_live flag if not already set
            if "is_live" not in pos:
                pos["is_live"] = self.mode == "LIVE"
            self.closed_arb.append(pos)
            self.arb_pnl += pnl


def _resolve_price(m: dict) -> tuple:
    """Extract (up_price, down_price) from market dict."""
    prices_raw = m.get("outcomePrices", "[0.5, 0.5]")
    if isinstance(prices_raw, str):
        prices = json.loads(prices_raw)
    else:
        prices = prices_raw
    return float(prices[0]), float(prices[1])


def _settle_one(state: ConvictionState, market_id: str, up_price: float, down_price: float):
    """Settle a single position if resolved. Returns True if settled."""
    if (up_price < 0.05 or up_price > 0.95) and (down_price < 0.05 or down_price > 0.95):
        pos = state.positions[market_id]
        resolved_dir = "UP" if up_price > 0.95 else "DOWN"
        bet = pos["bet"]
        entry = pos["entry_price"]
        won = pos["direction"] == resolved_dir
        if won:
            pnl = bet * (1.0 - entry - 0.02)
        else:
            pnl = -bet * entry
        state.close_position(market_id, resolved_dir, pnl)
        if not _VIRTUAL_MODE and pnl < 0:
            state.daily_loss += abs(pnl)

        # Loss-streak circuit breaker (live-only). Reset on win, increment on loss;
        # trigger pause when threshold is hit. Paper mode skips this so we can
        # measure the filter's impact against an unfiltered baseline.
        if not _VIRTUAL_MODE:
            if pnl > 0:
                if state.consecutive_losses > 0:
                    log.info("Loss streak ended (was %d) → win on %s %s",
                             state.consecutive_losses, pos["symbol"], pos["timeframe"])
                state.consecutive_losses = 0
            else:
                state.consecutive_losses += 1
                if state.consecutive_losses >= LOSS_STREAK_THRESHOLD and time.time() >= state.circuit_breaker_until:
                    state.circuit_breaker_until = time.time() + LOSS_STREAK_COOLDOWN_SEC
                    log.warning(
                        "CIRCUIT BREAKER tripped: %d consecutive losses — pausing new entries for %ds",
                        state.consecutive_losses, LOSS_STREAK_COOLDOWN_SEC
                    )

        log.info(
            "SETTLED %s %s @ %.3f → %s  pnl=%.2f  skew=%.1f%%",
            pos["symbol"], pos["timeframe"], entry, resolved_dir, pnl,
            pos.get("conviction", 0) * 100,
        )
        return True
    return False


def settle_positions(state: ConvictionState, markets: list[dict]):
    """Settle positions that appear in current discovered markets, then fetch
    any remaining stale open positions directly by ID (up to 20 per cycle)."""
    # Pass 1: settle from already-fetched markets
    for m in markets:
        market_id = m.get("id")
        if market_id not in state.positions:
            continue
        up_price, down_price = _resolve_price(m)
        _settle_one(state, market_id, up_price, down_price)

    # Pass 2: stale open positions not in current window — fetch directly
    current_ids = {m.get("id") for m in markets}
    stale = [mid for mid in list(state.positions) if mid not in current_ids]
    for market_id in stale[:20]:  # cap API calls per cycle
        pos = state.positions.get(market_id)
        if pos is None:
            continue
        m = fetch_market_by_id(market_id, pos.get("symbol", "?"), pos.get("timeframe", "5m"))
        if m is None:
            continue
        up_price, down_price = _resolve_price(m)
        _settle_one(state, market_id, up_price, down_price)


def settle_arb_positions(state: ConvictionState, markets: list[dict]):
    """Check if any arb positions have resolved; settle them."""
    for m in markets:
        market_id = m.get("id")
        if market_id not in state.arb_positions:
            continue

        prices_raw = m.get("outcomePrices", "[0.5, 0.5]")
        if isinstance(prices_raw, str):
            import json
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        up_price = float(prices[0])
        down_price = float(prices[1])

        # Check if resolved (one is ~0, one is ~1)
        if (up_price < 0.05 or up_price > 0.95) and (down_price < 0.05 or down_price > 0.95):
            pos = state.arb_positions[market_id]

            # For arb: you bought both YES and NO tokens totaling $total_cost
            # One wins and pays $1, other pays $0, minus 2% fee on the $1 payout
            # PnL = $1 - total_cost - 0.02
            total_cost = pos["total_cost"]
            pnl = 1.0 - total_cost - 0.02

            state.close_arb_position(market_id, pnl)

            log.info(
                "SETTLED ARB %s %s  cost=%.3f  pnl=%.4f",
                pos["symbol"],
                pos["timeframe"],
                total_cost,
                pnl,
            )


def main():
    log.info("=" * 60)
    log.info("Market Conviction Paper Trading Bot")
    log.info("Mode: %s", "LIVE" if not _VIRTUAL_MODE else "VIRTUAL")
    log.info("Strategy: %s", STRATEGY_MODE)
    if STRATEGY_MODE == "SKEW":
        log.info("Entry skew range (paper, fallback live): %.0f%% - %.0f%% (dominant side)", MIN_SKEW * 100, MAX_SKEW * 100)
        if not _VIRTUAL_MODE and LIVE_BANDS:
            log.info("Per-combo LIVE bands:")
            for (sym, tf), (lo, hi) in sorted(LIVE_BANDS.items()):
                log.info("  %s %s: %.0f%% - %.0f%%", sym, tf, lo * 100, hi * 100)
    log.info("Symbols: %s (%d total)", SYMBOLS, len(SYMBOLS))
    log.info("Timeframes: %s", TIMEFRAMES)
    if BLOCKED_SYMBOL_TF_LIVE:
        if _VIRTUAL_MODE:
            log.info("Block list (LIVE only, ignored in paper): %s", sorted(BLOCKED_SYMBOL_TF_LIVE))
        else:
            log.info("Blocked combos (LIVE): %s", sorted(BLOCKED_SYMBOL_TF_LIVE))
    if BLOCKED_HOURS_LIVE:
        tag = "LIVE only, ignored in paper" if _VIRTUAL_MODE else "LIVE"
        log.info("Blocked UTC hours (%s): %s", tag, sorted(BLOCKED_HOURS_LIVE))
    if _VIRTUAL_MODE:
        log.info("Loss-streak circuit breaker: DISABLED in paper mode (for A/B comparison)")
    else:
        log.info("Loss-streak circuit breaker: trigger=%d losses, cooldown=%ds",
                 LOSS_STREAK_THRESHOLD, LOSS_STREAK_COOLDOWN_SEC)
    log.info("Tick interval: 2s  |  discover every 30s  |  settle every 10s")
    log.info("Skew source: CLOBFeed WS + REST fallback (always-available book)")
    log.info("Direction: mainstream %s → original; mid-cap → invert when spread_gap > %.3f",
             sorted(MAINSTREAM_SYMBOLS), SPREAD_GAP_INVERT_THRESHOLD)
    log.info("=" * 60)

    # Start the live CLOB WebSocket feed — entry skew is read from this cache.
    clob_feed.start()

    # Use separate state files for live vs paper so histories don't mix.
    state_file = "data/conviction_real_state.json" if not _VIRTUAL_MODE else "data/conviction_state.json"
    state = ConvictionState(state_path=state_file)
    state.mode = "LIVE" if not _VIRTUAL_MODE else "VIRTUAL"
    log.info("State file: %s", state_file)
    # Record when transitioning to LIVE mode
    if state.mode == "LIVE" and state.live_start_time is None:
        state.live_start_time = time.time()
    log.info("Loaded state: %d open, %d closed, PnL=%.2f", len(state.positions), len(state.closed), state.pnl_total)

    # ── Initialize backend (live vs virtual) ──────────────────────────────
    from infra.backend import make_backend
    backend = make_backend(_VIRTUAL_MODE, log)

    # Live-mode wallet credentials (used by both C1 sync and AR)
    ar_wallet  = os.getenv("FUNDER", "")
    ar_key     = os.getenv("KEY", "")
    ar_enabled = (not _VIRTUAL_MODE) and bool(ar_wallet) and bool(ar_key)

    # ── C1: Balance sync at startup (live mode) ────────────────────────────
    # `start_of_day_balance` is total wallet value = cash + open/winning CTF
    # tokens. C2 reads the same total later, so cash → CTF conversion (every
    # new entry) nets to zero. Only real wins/losses (token resolves to $1/$0)
    # change the total — exactly what we want the circuit breaker to watch.
    if not _VIRTUAL_MODE:
        real_bal = backend.get_balance(log)
        if real_bal is None:
            log.error("Cannot fetch wallet balance — refusing to start in live mode.")
            return

        ctf_value = _get_pending_ctf_value(ar_wallet, log)
        total_value = real_bal + ctf_value

        state.real_clob_balance = real_bal
        state.start_of_day_balance = total_value
        state.daily_loss = 0.0
        state.save()
        if real_bal < 1.0:
            log.error("INSUFFICIENT BALANCE: %.2f < $1.00 minimum. Exiting.", real_bal)
            return
        log.info("Wallet synced: cash=$%.2f  CTF=$%.2f  total=$%.2f", real_bal, ctf_value, total_value)

    log_file = Path("data/conviction_log.jsonl")
    last_balance_sync = time.time()
    last_discover    = 0.0
    last_settle      = 0.0
    last_ar          = 0.0  # auto-redeem timer
    markets: list[dict] = []

    if ar_enabled:
        log.info("Auto-redeem (AR) enabled: every %d min", _AR_INTERVAL_SEC // 60)
    elif not _VIRTUAL_MODE:
        log.warning("Auto-redeem DISABLED: KEY/FUNDER missing from env")

    while True:
        try:
            now = time.time()
            now_ts = int(now)

            # ── Discover every 30s (market list changes slowly; subscribe new tokens) ─
            if now - last_discover >= 30.0:
                markets = discover_markets(now_ts)
                new_tokens: list[str] = []
                for m in markets:
                    tokens = _parse_token_ids(m)
                    if len(tokens) >= 2:
                        new_tokens.extend(tokens[:2])
                if new_tokens:
                    clob_feed.subscribe(new_tokens)
                last_discover = now

            # ── Settle every 10s (Gamma resolution is slow; no need for fast tick) ───
            if now - last_settle >= 10.0:
                settle_positions(state, markets)
                settle_arb_positions(state, markets)
                last_settle = now

                # Log when the loss-streak circuit breaker expires (live only)
                if (not _VIRTUAL_MODE) and state.circuit_breaker_until > 0 and now >= state.circuit_breaker_until:
                    log.info("Circuit breaker expired — resuming entries (loss count stays at %d until first win)",
                             state.consecutive_losses)
                    state.circuit_breaker_until = 0.0

            # ── Auto-redeem winning CTF tokens (live mode only) ────────────────
            #
            # The bot wins faster than the crypto bot's 12h AR sweep, so without
            # our own AR the wallet's spendable USDC stays near $0 while wins
            # pile up as redeemable CTF tokens. AR sweeps all winning positions
            # in the wallet (regardless of which bot opened them) and converts
            # them back to USDC.e in one call.
            if ar_enabled and (now - last_ar) >= _AR_INTERVAL_SEC:
                try:
                    redeemed = redeem_redeemable_positions(ar_wallet, ar_key, log)
                    if redeemed > 0:
                        log.info("AR: redeemed $%.2f USDC.e back to wallet", redeemed)
                        # Re-sync CLOB balance after redemption so C2 sees new funds
                        new_bal = backend.get_balance(log)
                        if new_bal is not None:
                            state.real_clob_balance = new_bal
                            state.save()
                except Exception as ar_exc:
                    log.error("AR failed: %s", ar_exc, exc_info=True)
                last_ar = now

            # ── C2: Daily loss circuit breaker (live mode) ────────────────────
            # Compares total wallet value (cash + CTF) to start-of-day total so
            # that cash → CTF conversion on every entry doesn't register as a
            # loss. Only real outcomes (winning token → $1, losing token → $0)
            # move the total.
            if not _VIRTUAL_MODE and time.time() - last_balance_sync > 300:  # Every 5 min
                now_bal = backend.get_balance(log)
                if now_bal is not None:
                    now_ctf = _get_pending_ctf_value(ar_wallet, log)
                    now_total = now_bal + now_ctf
                    daily_unrealized = state.start_of_day_balance - now_total
                    state.daily_loss = max(0, daily_unrealized)
                    state.real_clob_balance = now_bal

                    daily_loss_limit = state.start_of_day_balance * 0.50
                    if state.daily_loss > daily_loss_limit:
                        log.error(
                            "DAILY LOSS LIMIT HIT: total=$%.2f (cash=$%.2f + CTF=$%.2f) vs start=$%.2f → loss $%.2f (%.1f%%) > limit $%.2f (50%%). Halting trades.",
                            now_total, now_bal, now_ctf, state.start_of_day_balance,
                            state.daily_loss, state.daily_loss / state.start_of_day_balance * 100, daily_loss_limit
                        )
                        state.save()
                        time.sleep(30)
                        continue

                last_balance_sync = time.time()

            # Check for new signals
            for m in markets:
                market_id = m.get("id")
                if market_id in state.positions or market_id in state.arb_positions:
                    continue

                # Skip explicitly-blocked (symbol, timeframe) combos — LIVE only.
                # Paper mode keeps trading them so we have running data on every combo.
                if (not _VIRTUAL_MODE) and (m.get("_symbol"), m.get("_timeframe")) in BLOCKED_SYMBOL_TF_LIVE:
                    continue

                # Skip blocked UTC hours — LIVE only.
                if (not _VIRTUAL_MODE) and datetime.now(timezone.utc).hour in BLOCKED_HOURS_LIVE:
                    continue

                # Loss-streak circuit breaker — LIVE only.
                # If we just took N losses in a row, pause new entries. Paper bot
                # keeps trading so we can A/B the filter's impact.
                if (not _VIRTUAL_MODE) and time.time() < state.circuit_breaker_until:
                    continue

                # ARB CHECK FIRST
                arb = check_arb_opportunity(m)
                if arb:
                    if not _VIRTUAL_MODE:
                        # Live mode: place parallel YES + NO orders
                        # For now, just virtual mode
                        pass
                    else:
                        # Virtual trade
                        state.open_arb_position(market_id, arb)

                    log.info(
                        "ARB %s %s  yes=%.3f no=%.3f  cost=%.3f  margin=%.2f%%",
                        arb["symbol"],
                        arb["timeframe"],
                        arb["yes_price"],
                        arb["no_price"],
                        arb["total_cost"],
                        arb["margin_pct"],
                    )

                    # Log to JSONL
                    with open(log_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "market_id": market_id,
                                    "type": "ARB",
                                    "symbol": arb["symbol"],
                                    "timeframe": arb["timeframe"],
                                    "yes_price": arb["yes_price"],
                                    "no_price": arb["no_price"],
                                    "total_cost": arb["total_cost"],
                                    "net_margin": arb["net_margin"],
                                    "margin_pct": arb["margin_pct"],
                                    "mode": "LIVE" if not _VIRTUAL_MODE else "VIRTUAL",
                                }
                            )
                            + "\n"
                        )
                    continue  # DO NOT fall through to skew check

                # SKEW-BASED ENTRY (only if no arb)
                if STRATEGY_MODE == "SKEW":
                    # Read skew from live CLOB orderbook (sub-second freshness).
                    # Returns None when WS book is stale → silently skip this tick;
                    # we'll re-check on the next 2s tick once data arrives.
                    skew = _compute_clob_skew(m)
                    if skew is None:
                        continue

                    conviction_value = max(skew["up_skew"], skew["down_skew"])
                    # Per-combo band in live mode (each profitable sub-band per analysis);
                    # paper keeps the wider MIN_SKEW/MAX_SKEW range for ongoing data collection.
                    if not _VIRTUAL_MODE:
                        _band = LIVE_BANDS.get((skew["symbol"], skew["timeframe"]),
                                               (MIN_SKEW, MAX_SKEW))
                    else:
                        _band = (MIN_SKEW, MAX_SKEW)
                    if not (_band[0] <= conviction_value <= _band[1]):
                        continue

                    # ── Per-tier direction rule ───────────────────────────────────
                    original_direction = skew["direction"]
                    dominant_ask  = skew["yes_price"] if original_direction == "UP"   else skew["no_price"]
                    minority_ask  = skew["no_price"]  if original_direction == "UP"   else skew["yes_price"]
                    spread_gap    = dominant_ask - conviction_value
                    sym           = skew["symbol"]

                    if sym in MAINSTREAM_SYMBOLS:
                        invert = False
                    else:
                        # Mid-cap: invert when book is sparse (wide gap between ask and skew)
                        invert = spread_gap > SPREAD_GAP_INVERT_THRESHOLD

                    if invert:
                        final_direction = "DOWN" if original_direction == "UP" else "UP"
                        final_entry     = minority_ask
                    else:
                        final_direction = original_direction
                        final_entry     = dominant_ask

                    signal = {
                        "symbol":      sym,
                        "timeframe":   skew["timeframe"],
                        "direction":   final_direction,
                        "conviction":  conviction_value,
                        "entry_price": final_entry,
                        "bet_size":    1.0,
                        "inverted":    invert,
                    }
                else:
                    # OLD: CONVICTION-BASED (disabled, kept for reference)
                    signal = compute_conviction_signal(m)
                    if signal is None:
                        continue

                # ── Live vs Virtual execution ──────────────────────────────
                if not _VIRTUAL_MODE:
                    # Get token ID from market (clobTokenIds is a JSON string)
                    import json as json_lib
                    token_ids_raw = m.get("clobTokenIds", "[]")
                    if isinstance(token_ids_raw, str):
                        try:
                            token_ids = json_lib.loads(token_ids_raw)
                        except:
                            token_ids = []
                    else:
                        token_ids = token_ids_raw

                    direction_idx = 0 if signal["direction"] == "UP" else 1
                    token_id = token_ids[direction_idx] if len(token_ids) > direction_idx else None

                    if not token_id:
                        log.warning("Missing token ID for %s", market_id)
                        continue

                    # Place real order using backend ($1 fixed)
                    from infra.types import ClobTokenId, ConditionId
                    result = backend.place_order(
                        token_id=ClobTokenId(token_id),
                        bet_size_usdc=1.0,
                        market_id=ConditionId(market_id),
                        log=log,
                        price_hint=signal["entry_price"],
                    )

                    # Backend contract (infra/backend.py):
                    #   status: "FILLED" | "NO_FILL", fill_price: float,
                    #   filled_usdc: float, order_id: str
                    if not result or result.get("status") != "FILLED":
                        continue

                    # CLOB minimum order is 5 tokens; an intended $1 bet at
                    # 0.59 ask becomes ~$2.95 spent on 5 tokens. Record `bet`
                    # as token count (not USDC) so the PnL formula
                    #   win  = bet * (1 - entry - 0.02)
                    #   loss = -bet * entry
                    # reflects real wallet movement instead of the intended $1.
                    fill_price  = float(result.get("fill_price", signal["entry_price"]))
                    filled_usdc = float(result.get("filled_usdc", 1.0))
                    tokens      = filled_usdc / fill_price if fill_price > 0 else 0.0

                    pos = {
                        "symbol":      signal["symbol"],
                        "timeframe":   signal["timeframe"],
                        "direction":   signal["direction"],
                        "conviction":  signal["conviction"],
                        "entry_price": fill_price,
                        "bet":         tokens,
                        "opened_at":   time.time(),
                        "is_live":     True,
                        "inverted":    bool(signal.get("inverted", False)),
                        "order_id":    result.get("order_id", ""),
                        "fill_type":   result.get("fill_type", ""),
                        "filled_usdc": filled_usdc,
                    }
                    state.positions[market_id] = pos
                else:
                    # Virtual trade
                    state.open_position(market_id, signal)

                entry_type = "SKEW" if STRATEGY_MODE == "SKEW" else "CONVICTION"
                flip_tag   = " INVERT" if signal.get("inverted") else ""
                log.info(
                    "%s%s %s %s skew=%.0f%% direction=%s  bet=$1.00  @ %.3f",
                    entry_type,
                    flip_tag,
                    signal["symbol"],
                    signal["timeframe"],
                    signal["conviction"] * 100,
                    signal["direction"],
                    signal["entry_price"],
                )

                # Log to JSONL
                with open(log_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "market_id": market_id,
                                "symbol": signal["symbol"],
                                "timeframe": signal["timeframe"],
                                "direction": signal["direction"],
                                "conviction": signal["conviction"],
                                "entry_price": signal["entry_price"],
                                "bet": 1.0 if not _VIRTUAL_MODE else signal["bet_size"],
                                "mode": "LIVE" if not _VIRTUAL_MODE else "VIRTUAL",
                            }
                        )
                        + "\n"
                    )

            state.save()
            time.sleep(2)

        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
