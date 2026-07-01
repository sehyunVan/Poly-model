"""
llm_updown/paper.py — Paper tracker for LLM-primary up/down strategy.

Settles when the Polymarket window resolves (up_price → 0.97 or 0.03).
PnL: Win = bet*(1-ask)/ask, Loss = -bet  (same as swarm/over_below).
"""
from __future__ import annotations

import json, logging, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("llm_updown.paper")

_PAPER_FILE = Path("data/llm_updown_paper.jsonl")
_EXEC_FILE  = Path("data/llm_updown_executed.json")
RESOLVED_HI = 0.97
RESOLVED_LO = 0.03
FEE_RATE    = 0.02   # Polymarket taker fee (2% of bet notional), charged on entry
_EXEC_TTL   = 3600   # 1h — windows are 5 min, so 1h is plenty for dedup
GAMMA_BASE  = "https://gamma-api.polymarket.com"
_HTTP       = httpx.Client(timeout=10.0)


@dataclass
class UDTrade:
    market_id:    str
    condition_id: str
    slug:         str
    question:     str
    symbol:       str
    timeframe:    str       # "5m", "15m", "4h"
    bet_direction: str      # "YES" or "NO"
    entry_ts:     float
    entry_ask:    float     # price paid
    bet_size:     float
    window_elapsed: float
    avg_prob:     float
    market_prob:  float
    edge:         float
    model_1:      str
    model_2:      str
    prob_1:       float
    prob_2:       float
    reasoning_1:  str
    reasoning_2:  str
    current_price: float
    open_price:   float
    settled:      bool          = False
    outcome:      Optional[str] = None
    exit_price:   Optional[float] = None
    exit_ts:      Optional[float] = None
    pnl:          Optional[float] = None

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        d.setdefault("timeframe", "5m")   # backfill legacy records
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_trades() -> list[UDTrade]:
    if not _PAPER_FILE.exists(): return []
    trades = []
    for line in open(_PAPER_FILE, encoding="utf-8"):
        if line.strip():
            try: trades.append(UDTrade.from_dict(json.loads(line)))
            except Exception as e: log.warning("Bad record: %s", e)
    return trades


def save_trades(trades: list[UDTrade]) -> None:
    _PAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PAPER_FILE, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_executed() -> dict[str, float]:
    if not _EXEC_FILE.exists(): return {}
    try: return json.loads(_EXEC_FILE.read_text(encoding="utf-8"))
    except: return {}


def save_executed(ex: dict[str, float]) -> None:
    _EXEC_FILE.parent.mkdir(parents=True, exist_ok=True)
    _EXEC_FILE.write_text(json.dumps(ex, indent=2), encoding="utf-8")


def prune_executed(ex: dict[str, float]) -> dict[str, float]:
    now = time.time()
    return {k: v for k, v in ex.items() if now - v < _EXEC_TTL}


def record_trade(signal, bet_size: float, fill_ask: Optional[float] = None) -> UDTrade:
    from llm_updown.scanner import UDSignal
    m = signal.market
    # Entry price = executable CLOB ask the signal was built on (live passes the
    # actual fill). Falls back to the Gamma mid only if no ask is available.
    ask = fill_ask if fill_ask is not None else getattr(signal, "entry_ask", 0.0)
    if not ask:
        ask = m.up_price if signal.bet_direction == "YES" else m.down_price
    trade = UDTrade(
        market_id=m.market_id, condition_id=m.condition_id,
        slug=m.slug, question=m.question, symbol=m.symbol,
        timeframe=m.timeframe,
        bet_direction=signal.bet_direction,
        entry_ts=time.time(), entry_ask=ask, bet_size=bet_size,
        window_elapsed=m.window_elapsed,
        avg_prob=signal.avg_prob, market_prob=signal.market_prob, edge=signal.edge,
        model_1=signal.model_1_name, model_2=signal.model_2_name,
        prob_1=signal.llm_prob_1, prob_2=signal.llm_prob_2,
        reasoning_1=signal.reasoning_1, reasoning_2=signal.reasoning_2,
        current_price=signal.current_price, open_price=signal.open_price,
    )
    _PAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PAPER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade.to_dict()) + "\n")
    log.info("PAPER RECORD %s %s[%s] | ask=%.3f bet=$%.2f edge=%+.3f elapsed=%.0fs | %s",
             trade.bet_direction, trade.symbol, trade.timeframe,
             ask, bet_size, signal.edge, m.window_elapsed, m.question[:55])
    return trade


def settle_open_trades(trades: list[UDTrade], max_per_call: int = 60) -> int:
    """Settle resolved windows. Capped at max_per_call per invocation to avoid stalling."""
    open_t = [t for t in trades if not t.settled]
    if not open_t: return 0
    settled = 0
    for trade in open_t[:max_per_call]:
        try:
            # Use slug-based event lookup — conditionId query returns wrong sub-markets
            r = _HTTP.get(f"{GAMMA_BASE}/events",
                          params={"slug": trade.slug}, timeout=8)
            events = r.json()
            if not isinstance(events, list) or not events:
                continue
            ev = events[0]
            if not ev.get("closed", False):
                continue   # window still open — check next cycle
            markets = ev.get("markets", [])
            if not markets:
                continue
            p = markets[0].get("outcomePrices", "[]")
            if isinstance(p, str): p = json.loads(p)
            if not p: continue
            yp = float(p[0])   # UP token price: 1.0 if UP won, 0.0 if DOWN won
            if yp > RESOLVED_HI or yp < RESOLVED_LO:
                trade.exit_price = yp
                trade.exit_ts    = time.time()
                trade.settled    = True
                won = (trade.bet_direction == "YES" and yp > RESOLVED_HI) or \
                      (trade.bet_direction == "NO"  and yp < RESOLVED_LO)
                trade.outcome = "WIN" if won else "LOSS"
                gross = trade.bet_size*(1-trade.entry_ask)/trade.entry_ask if won else -trade.bet_size
                trade.pnl = round(gross - FEE_RATE * trade.bet_size, 4)   # 2% taker fee
                settled += 1
                log.info("SETTLED %s %s[%s] → %s pnl=%+.2f yp=%.2f | %s",
                         trade.bet_direction, trade.symbol, trade.timeframe,
                         trade.outcome, trade.pnl, yp, trade.question[:50])
        except Exception as e:
            log.debug("Settlement check failed (%s): %s", trade.slug, e)
    return settled


def print_summary(trades: list[UDTrade]) -> None:
    s = [t for t in trades if t.settled]
    o = [t for t in trades if not t.settled]
    w = [t for t in s if t.outcome == "WIN"]
    pnl = sum(t.pnl or 0 for t in s)
    wr  = len(w)/len(s)*100 if s else 0
    log.info("=== LLM UpDown Summary === settled=%d WR=%.0f%% pnl=$%.2f open=%d",
             len(s), wr, pnl, len(o))
