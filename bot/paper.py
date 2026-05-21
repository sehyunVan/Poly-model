"""
bot/paper.py -- Paper trading tracker.

Every cycle:
  1. Settle any open picks whose market has resolved (price >= 0.97 or <= 0.03)
  2. Save new picks from the current swarm cycle
  3. Print a running summary (win rate, PnL if we had bet $10/pick)

State is persisted to data/paper_trades.jsonl (one JSON record per line).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from bot.swarm.consensus import MarketConsensus

log = logging.getLogger(__name__)

PAPER_FILE        = Path("data/paper_trades.jsonl")
REAL_TRADES_FILE  = Path("data/swarm_real_trades.jsonl")
SWARM_STATE       = Path("data/swarm_state.json")
SIMULATED_BET     = 10.0    # hypothetical $ per pick for PnL calculation
RESOLVED_HI       = 0.97    # price threshold to call YES resolved
RESOLVED_LO       = 0.03    # price threshold to call NO resolved


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    # Identity
    market_id:    str
    question:     str
    direction:    str          # "YES" or "NO"
    # Entry
    entry_ts:     str          # ISO timestamp
    entry_price:  float        # YES price at time of pick
    score:        float
    yes_votes:    int
    no_votes:     int
    avg_conf:     float
    # Settlement
    settled:      bool         = False
    outcome:      Optional[str] = None   # "WIN" or "LOSS"
    exit_price:   Optional[float] = None
    exit_ts:      Optional[str]   = None
    pnl:          Optional[float] = None  # based on SIMULATED_BET
    # Per-model votes: [{model, decision, confidence, reasoning}]
    model_votes:  list = field(default_factory=list)
    # Per-model correctness filled on settlement: {model_name: True/False/None(abstain)}
    model_correct: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaperTrade":
        # Handle legacy records that predate model_votes/model_correct fields
        d.setdefault("model_votes", [])
        d.setdefault("model_correct", {})
        return cls(**d)


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_trades() -> list[PaperTrade]:
    if not PAPER_FILE.exists():
        return []
    trades = []
    with open(PAPER_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(PaperTrade.from_dict(json.loads(line)))
                except Exception:
                    pass
    return trades


def _save_trades(trades: list[PaperTrade]) -> None:
    PAPER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PAPER_FILE, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t.to_dict()) + "\n")


# ── Settlement ────────────────────────────────────────────────────────────────

def _fetch_yes_price(market_id: str) -> Optional[float]:
    """Fetch current YES price from Gamma API."""
    try:
        resp = httpx.get(
            f"https://gamma-api.polymarket.com/markets/{market_id}",
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        import json as _json
        raw = data.get("outcomePrices", "[0.5,0.5]")
        prices = _json.loads(raw) if isinstance(raw, str) else raw
        return float(prices[0])
    except Exception as e:
        log.debug("Price fetch failed for %s: %s", market_id, e)
        return None


def _settle(trade: PaperTrade) -> PaperTrade:
    """Check if a trade has resolved and mark it settled."""
    price = _fetch_yes_price(trade.market_id)
    if price is None:
        return trade

    yes_resolved = price >= RESOLVED_HI
    no_resolved  = price <= RESOLVED_LO

    if not yes_resolved and not no_resolved:
        return trade  # still open

    # Determine outcome
    if trade.direction == "YES":
        won = yes_resolved
        exit_price = price
    else:  # NO
        won = no_resolved
        exit_price = price

    # PnL: bought YES at entry_price, pays $1 on win
    if trade.direction == "YES":
        pnl = (1.0 - trade.entry_price) * SIMULATED_BET if won else -trade.entry_price * SIMULATED_BET
    else:
        pnl = trade.entry_price * SIMULATED_BET if won else -(1.0 - trade.entry_price) * SIMULATED_BET

    trade.settled    = True
    trade.outcome    = "WIN" if won else "LOSS"
    trade.exit_price = round(exit_price, 4)
    trade.exit_ts    = datetime.now(tz=timezone.utc).isoformat()
    trade.pnl        = round(pnl, 3)

    # Score each model: correct if it voted for the winning side
    actual_winner = trade.direction if won else ("NO" if trade.direction == "YES" else "YES")
    for vote in trade.model_votes:
        model = vote.get("model", "unknown")
        decision = vote.get("decision", "NO_TRADE")
        if decision == "NO_TRADE":
            trade.model_correct[model] = None  # abstained — not counted
        else:
            trade.model_correct[model] = (decision == actual_winner)

    return trade


# ── Summary ───────────────────────────────────────────────────────────────────

def _model_accuracy(settled: list[PaperTrade]) -> dict[str, dict]:
    """
    Aggregate per-model correctness across all settled trades.
    Returns {model_name: {correct, wrong, abstain, accuracy_pct}}.
    """
    stats: dict[str, dict] = {}
    for t in settled:
        for model, correct in (t.model_correct or {}).items():
            s = stats.setdefault(model, {"correct": 0, "wrong": 0, "abstain": 0})
            if correct is None:
                s["abstain"] += 1
            elif correct:
                s["correct"] += 1
            else:
                s["wrong"] += 1
    for s in stats.values():
        total = s["correct"] + s["wrong"]
        s["accuracy_pct"] = round(s["correct"] / total * 100, 1) if total else None
        s["voted"] = total
    return stats


def _print_summary(trades: list[PaperTrade]) -> None:
    settled = [t for t in trades if t.settled]
    open_t  = [t for t in trades if not t.settled]
    wins    = [t for t in settled if t.outcome == "WIN"]
    total_pnl = sum(t.pnl for t in settled if t.pnl is not None)
    wr = len(wins) / len(settled) * 100 if settled else 0.0

    print(f"\n  [PAPER] open={len(open_t)}  settled={len(settled)}  "
          f"WR={wr:.1f}%  PnL=${total_pnl:+.2f}  (@ ${SIMULATED_BET}/trade)")

    if settled:
        recent = sorted(settled, key=lambda t: t.exit_ts or "", reverse=True)[:5]
        for t in recent:
            mark = "WIN " if t.outcome == "WIN" else "LOSS"
            print(f"    {mark}  {t.direction:<3}  pnl=${t.pnl:+.2f}  {t.question[:60]}")

    # Per-model leaderboard
    model_stats = _model_accuracy(settled)
    if model_stats:
        print("  [MODEL ACCURACY]")
        ranked = sorted(model_stats.items(), key=lambda x: x[1].get("accuracy_pct") or 0, reverse=True)
        for name, s in ranked:
            acc = s["accuracy_pct"]
            acc_str = f"{acc:.1f}%" if acc is not None else "n/a"
            print(f"    {name:<20} {acc_str:>6}  ({s['correct']}W/{s['wrong']}L  +{s['abstain']} abstain)")


# ── Real trade tracker ───────────────────────────────────────────────────────

def record_real_execution(pick, ask: float, bet: float, direction: str | None = None) -> None:
    """
    Append one real executed trade to REAL_TRADES_FILE.
    Called from main.py immediately after execute_pick() succeeds.
    `direction` should be fill["direction"] so that YES→NO or NO→YES flips are recorded
    as the actually-executed direction, not the original consensus direction.
    """
    actual_direction = direction if direction is not None else pick.direction.value
    record = {
        "market_id":   pick.market.id,
        "question":    pick.market.question,
        "direction":   actual_direction,
        "entry_ts":    datetime.now(tz=timezone.utc).isoformat(),
        "ask":         round(ask, 4),    # actual CLOB fill price
        "bet":         round(bet, 4),    # real USDC spent
        "yes_price":   round(pick.market.yes_price, 4),
        "score":       round(pick.score, 4),
        "synthesis_confidence": pick.synthesis_confidence,  # deepseek calibrated confidence
        "avg_confidence":       round(pick.avg_confidence, 1) if pick.avg_confidence else None,
        "yes_votes":    pick.yes_votes,
        "no_votes":     pick.no_votes,
        "whale_strength": pick.whale_strength if hasattr(pick, 'whale_strength') else None,
        "settled":     False,
        "outcome":     None,
        "exit_price":  None,
        "exit_ts":     None,
        "pnl":         None,
    }
    REAL_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REAL_TRADES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    flip_note = " [FLIPPED from %s]" % pick.direction.value if actual_direction != pick.direction.value else ""
    log.info("Real trade recorded: %s%s %s ask=%.3f bet=$%.2f",
             actual_direction, flip_note, pick.market.question[:50], ask, bet)


def _load_real_trades() -> list[dict]:
    if not REAL_TRADES_FILE.exists():
        return []
    trades = []
    with open(REAL_TRADES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except Exception:
                    pass
    return trades


def _save_real_trades(trades: list[dict]) -> None:
    REAL_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REAL_TRADES_FILE, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def _settle_real_trades(trades: list[dict]) -> list[dict]:
    """Check open real trades for resolution and compute actual PnL."""
    n_settled = 0
    for t in trades:
        if t.get("settled"):
            continue
        price = _fetch_yes_price(t["market_id"])
        if price is None:
            continue
        yes_resolved = price >= RESOLVED_HI
        no_resolved  = price <= RESOLVED_LO
        if not yes_resolved and not no_resolved:
            continue

        direction = t["direction"]
        ask       = float(t["ask"])
        bet       = float(t["bet"])

        if direction == "YES":
            won = yes_resolved
        else:
            won = no_resolved

        # Real PnL: bought tokens at ask price, each pays $1 on win
        # tokens = bet / ask
        # win:  pnl = bet/ask - bet = bet * (1 - ask) / ask
        # loss: pnl = -bet
        if won:
            pnl = round(bet * (1.0 - ask) / ask, 4)
        else:
            pnl = round(-bet, 4)

        t["settled"]    = True
        t["outcome"]    = "WIN" if won else "LOSS"
        t["exit_price"] = round(price, 4)
        t["exit_ts"]    = datetime.now(tz=timezone.utc).isoformat()
        t["pnl"]        = pnl
        n_settled += 1
        log.info("Real trade settled: %s %s pnl=$%+.4f  %s",
                 t["outcome"], direction, pnl, t["question"][:55])

    return trades


# ── Swarm state writer ────────────────────────────────────────────────────────

def _write_swarm_state(trades: list[PaperTrade], last_picks: list[MarketConsensus]) -> None:
    """Write a structured swarm_state.json for the dashboard to read."""
    from datetime import date as _date
    settled = [t for t in trades if t.settled]
    open_t  = [t for t in trades if not t.settled]
    wins    = [t for t in settled if t.outcome == "WIN"]
    total_pnl = round(sum(t.pnl for t in settled if t.pnl is not None), 2)
    today = _date.today().isoformat()
    daily_pnl = round(sum(
        t.pnl for t in settled
        if t.pnl is not None and (t.exit_ts or "")[:10] == today
    ), 2)
    wr = round(len(wins) / len(settled) * 100, 1) if settled else None

    # ── Real trade stats ──────────────────────────────────────────────────────
    real_trades = _load_real_trades()
    real_settled = [t for t in real_trades if t.get("settled")]
    real_open    = [t for t in real_trades if not t.get("settled")]
    real_wins    = [t for t in real_settled if t.get("outcome") == "WIN"]
    real_pnl     = round(sum(t.get("pnl", 0) for t in real_settled), 4)
    real_daily   = round(sum(
        t.get("pnl", 0) for t in real_settled
        if (t.get("exit_ts") or "")[:10] == today
    ), 4)
    real_wr      = round(len(real_wins) / len(real_settled) * 100, 1) if real_settled else None

    # Last cycle's top picks (for the "current picks" panel)
    cycle_picks = []
    for p in last_picks:
        cycle_picks.append({
            "market_id":  p.market.id,
            "question":   p.market.question,
            "direction":  p.direction.value,
            "score":      p.score,
            "yes_votes":  p.yes_votes,
            "no_votes":   p.no_votes,
            "avg_conf":   p.avg_confidence,
            "yes_price":  p.market.yes_price,
            "vol_24h":    p.market.volume_24h,
            "whale":      {
                "direction": p.whale.direction,
                "strength":  p.whale.strength,
                "has_signal": p.whale.has_signal,
            } if p.whale else None,
        })

    model_stats = _model_accuracy(settled)
    model_leaderboard = sorted(
        [{"model": k, **v} for k, v in model_stats.items()],
        key=lambda x: x.get("accuracy_pct") or 0,
        reverse=True,
    )

    # ── Mirror tracker stats ──────────────────────────────────────────────────
    try:
        from bot.paper_mirror import mirror_stats
        mirror = mirror_stats()
    except Exception as e:
        log.warning("mirror_stats failed: %s", e)
        mirror = {}

    state = {
        "last_updated":   datetime.now(tz=timezone.utc).isoformat(),
        "open_count":     len(open_t),
        "settled_count":  len(settled),
        "win_rate":       wr,
        "total_pnl":      total_pnl,
        "daily_pnl":      daily_pnl,
        "simulated_bet":  SIMULATED_BET,
        "model_leaderboard": model_leaderboard,
        "last_cycle_picks": cycle_picks,
        "open_picks": [t.to_dict() for t in open_t[-20:]],
        "closed_picks": sorted(
            [t.to_dict() for t in settled],
            key=lambda d: d.get("exit_ts") or "",
            reverse=True,
        )[:30],
        # Real executed trades (actual $1 bets placed on CLOB)
        "real_open_count":    len(real_open),
        "real_settled_count": len(real_settled),
        "real_win_rate":      real_wr,
        "real_pnl":           real_pnl,
        "real_daily_pnl":     real_daily,
        "real_open_picks":    real_open[-20:],
        "real_closed_picks":  sorted(
            real_settled, key=lambda d: d.get("exit_ts") or "", reverse=True
        )[:30],
        # Strategy-mirror trades (mirrors all real gates, $10 sim, no execution friction)
        **mirror,
    }

    SWARM_STATE.parent.mkdir(parents=True, exist_ok=True)
    SWARM_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Public API ────────────────────────────────────────────────────────────────

def flush_state(last_picks: list[MarketConsensus]) -> None:
    """
    Write swarm_state.json — always loads current paper trades from disk so
    real trades recorded after record_picks() are included immediately.
    Call at end of every cycle, after real execution is done.
    """
    trades = _load_trades()
    _write_swarm_state(trades, last_picks)


def record_picks(picks: list[MarketConsensus]) -> tuple:
    """
    Called each cycle.
    1. Load all existing trades.
    2. Settle any open ones that have resolved.
    3. Add new picks (skip markets already tracked).
    4. Save and print summary.
    Returns (trades, picks) so caller can call flush_state() after real execution.
    """
    trades = _load_trades()
    existing_ids = {t.market_id for t in trades}

    # Settle open real trades
    real_trades = _load_real_trades()
    if any(not t.get("settled") for t in real_trades):
        real_trades = _settle_real_trades(real_trades)
        _save_real_trades(real_trades)

    # Settle open paper trades
    n_settled = 0
    for i, t in enumerate(trades):
        if not t.settled:
            updated = _settle(t)
            if updated.settled:
                n_settled += 1
                trades[i] = updated
                log.info(
                    "Paper settled: %s  %s  pnl=$%+.2f  %s",
                    updated.outcome, updated.direction,
                    updated.pnl or 0, updated.question[:55],
                )

    # Record new picks
    n_new = 0
    ts = datetime.now(tz=timezone.utc).isoformat()
    for pick in picks:
        if pick.market.id in existing_ids:
            continue
        trades.append(PaperTrade(
            market_id=pick.market.id,
            question=pick.market.question,
            direction=pick.direction.value,
            entry_ts=ts,
            entry_price=pick.market.yes_price,
            score=pick.score,
            yes_votes=pick.yes_votes,
            no_votes=pick.no_votes,
            avg_conf=pick.avg_confidence,
            model_votes=[
                {
                    "model":      v.model_name,
                    "decision":   v.decision.value,
                    "confidence": v.confidence,
                    "reasoning":  v.reasoning,
                }
                for v in pick.verdicts
            ],
        ))
        n_new += 1
        log.info(
            "Paper recorded: %s  score=%.3f  %s",
            pick.direction.value, pick.score, pick.market.question[:55],
        )

    if n_new or n_settled:
        _save_trades(trades)

    _print_summary(trades)
    return trades, picks
