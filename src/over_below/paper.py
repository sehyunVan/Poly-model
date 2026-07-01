"""
over_below/paper.py — Paper trade tracker for over/below signals.

Each trade is persisted to data/over_below_paper.jsonl.
Settlement: when yes_price approaches 0.97 (YES resolved) or 0.03 (NO resolved)
on the next scan, we mark the trade as settled and compute PnL.

PnL formula (same as swarm):
  Win:  pnl = bet * (1 - entry_ask) / entry_ask
  Loss: pnl = -bet
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("over_below.paper")

_PAPER_FILE  = Path("data/over_below_paper.jsonl")
_EXEC_FILE   = Path("data/over_below_executed.json")
RESOLVED_HI  = 0.97
RESOLVED_LO  = 0.03
FEE_RATE     = 0.02            # Polymarket taker fee (2% of bet notional)
_EXEC_TTL_S  = 7 * 24 * 3600   # 7-day dedup TTL

_HTTP = httpx.Client(timeout=10.0)
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"


def _clob_ask(token_id: str) -> Optional[float]:
    """Best (lowest) executable ask for a token — what a taker actually pays.
    Gamma yes/no prices are lagging mids; we fill against the CLOB book."""
    if not token_id:
        return None
    try:
        b = _HTTP.get(f"{CLOB_BASE}/book", params={"token_id": str(token_id)}, timeout=8).json()
        asks = [float(a["price"]) for a in b.get("asks", [])]
        return min(asks) if asks else None
    except Exception:
        return None


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class OBTrade:
    # Identity
    market_id:    str
    condition_id: str
    question:     str
    symbol:       str
    direction_market: str   # "above" or "below" — what the market asks
    threshold:    float
    # Entry
    bet_direction: str      # "YES" or "NO" — what we bet
    entry_ts:     float     # unix timestamp
    entry_ask:    float     # price paid (yes_price for YES bet, no_price for NO bet)
    bet_size:     float
    # LLM scores
    avg_prob:     float
    market_prob:  float
    edge:         float
    model_1:      str
    model_2:      str
    prob_1:       float
    prob_2:       float
    reasoning_1:  str
    reasoning_2:  str
    # Settlement
    settled:      bool           = False
    outcome:      Optional[str]  = None   # "WIN" or "LOSS"
    exit_price:   Optional[float] = None
    exit_ts:      Optional[float] = None
    pnl:          Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OBTrade":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Persistence ────────────────────────────────────────────────────────────────

def load_trades() -> list[OBTrade]:
    if not _PAPER_FILE.exists():
        return []
    trades = []
    with open(_PAPER_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(OBTrade.from_dict(json.loads(line)))
                except Exception as exc:
                    log.warning("Bad trade record: %s", exc)
    return trades


def save_trades(trades: list[OBTrade]) -> None:
    _PAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PAPER_FILE, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_executed() -> dict[str, float]:
    """Load {market_id: timestamp} dedup dict."""
    if not _EXEC_FILE.exists():
        return {}
    try:
        return json.loads(_EXEC_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_executed(executed: dict[str, float]) -> None:
    _EXEC_FILE.parent.mkdir(parents=True, exist_ok=True)
    _EXEC_FILE.write_text(json.dumps(executed, indent=2), encoding="utf-8")


def prune_executed(executed: dict[str, float]) -> dict[str, float]:
    """Remove entries older than TTL."""
    now = time.time()
    return {k: v for k, v in executed.items() if now - v < _EXEC_TTL_S}


# ── Record new trade ───────────────────────────────────────────────────────────

def record_trade(signal, bet_size: float) -> OBTrade:
    """Create an OBTrade from a scanner OBSignal and persist it."""
    from over_below.scanner import OBSignal
    m = signal.market

    # Entry price = executable CLOB ask, not the Gamma mid (which lags the book).
    bet_token = m.yes_token_id if signal.bet_direction == "YES" else m.no_token_id
    entry_ask = _clob_ask(bet_token) or (
        m.yes_price if signal.bet_direction == "YES" else m.no_price)

    trade = OBTrade(
        market_id=m.market_id,
        condition_id=m.condition_id,
        question=m.question,
        symbol=m.symbol,
        direction_market=m.direction,
        threshold=m.threshold,
        bet_direction=signal.bet_direction,
        entry_ts=time.time(),
        entry_ask=entry_ask,
        bet_size=bet_size,
        avg_prob=signal.avg_prob,
        market_prob=signal.market_prob,
        edge=signal.edge,
        model_1=signal.model_1_name,
        model_2=signal.model_2_name,
        prob_1=signal.llm_prob_1,
        prob_2=signal.llm_prob_2,
        reasoning_1=signal.reasoning_1,
        reasoning_2=signal.reasoning_2,
    )

    # Append to JSONL
    _PAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PAPER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade.to_dict()) + "\n")

    log.info(
        "PAPER RECORD %s %s | ask=%.3f bet=$%.2f edge=%+.3f | %s",
        trade.bet_direction, trade.question[:55],
        entry_ask, bet_size, signal.edge, trade.condition_id[:12],
    )
    return trade


# ── Settlement ──────────────────────────────────────────────────────────────────

def _fetch_yes_price(market_id: str) -> Optional[float]:
    """Fetch the resolved YES price from Gamma for settlement check.

    Uses the /markets/{id} path. The previous ?conditionId= query was broken —
    Gamma ignores the param and returns an unrelated market, so trades never
    settled. Only returns a price once the market is closed (resolved).
    """
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/markets/{market_id}", timeout=8)
        r.raise_for_status()
        m = r.json()
        if isinstance(m, list):
            m = m[0] if m else None
        if not m or not m.get("closed", False):
            return None
        raw = m.get("outcomePrices", "[]")
        prices = json.loads(raw) if isinstance(raw, str) else raw
        if prices:
            return float(prices[0])
    except Exception as exc:
        log.warning("Settlement check failed (%s): %s", market_id, exc)
    return None


def settle_open_trades(trades: list[OBTrade]) -> int:
    """
    Check open trades for resolution; mutates trades in place.
    Returns number newly settled.
    """
    open_trades = [t for t in trades if not t.settled]
    if not open_trades:
        return 0

    settled_count = 0
    for trade in open_trades:
        yes_price = _fetch_yes_price(trade.market_id)
        if yes_price is None:
            continue

        resolved_yes = yes_price >= RESOLVED_HI
        resolved_no  = yes_price <= RESOLVED_LO
        if not resolved_yes and not resolved_no:
            continue

        trade.exit_price = yes_price
        trade.exit_ts    = time.time()
        trade.settled    = True

        # Win condition: if we bet YES and it resolved YES, or bet NO and resolved NO
        fee = FEE_RATE * trade.bet_size   # 2% taker fee on entry notional
        if (trade.bet_direction == "YES" and resolved_yes) or \
           (trade.bet_direction == "NO"  and resolved_no):
            trade.outcome = "WIN"
            trade.pnl = round(trade.bet_size * (1.0 - trade.entry_ask) / trade.entry_ask - fee, 4)
        else:
            trade.outcome = "LOSS"
            trade.pnl = round(-trade.bet_size - fee, 4)

        settled_count += 1
        log.info(
            "SETTLED %s %s → %s  pnl=%+.2f  %s",
            trade.bet_direction, trade.question[:50],
            trade.outcome, trade.pnl, trade.condition_id[:12],
        )

    return settled_count


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(trades: list[OBTrade]) -> None:
    settled  = [t for t in trades if t.settled]
    open_    = [t for t in trades if not t.settled]
    wins     = [t for t in settled if t.outcome == "WIN"]
    total_pnl = sum(t.pnl or 0 for t in settled)
    wr = len(wins) / len(settled) * 100 if settled else 0

    log.info(
        "=== OB Paper Summary ===  settled=%d (%.0f%% WR / $%.2f pnl)  open=%d",
        len(settled), wr, total_pnl, len(open_),
    )

    # Per-symbol breakdown
    for sym in sorted({t.symbol for t in settled}):
        st = [t for t in settled if t.symbol == sym]
        w  = [t for t in st if t.outcome == "WIN"]
        p  = sum(t.pnl or 0 for t in st)
        log.info("  %s: %d trades  %.0f%% WR  $%.2f pnl",
                 sym, len(st), len(w) / len(st) * 100 if st else 0, p)
