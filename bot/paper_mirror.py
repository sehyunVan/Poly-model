"""
bot/paper_mirror.py — Strategy-mirror paper tracker.

Logs only picks that would have passed every real-bot execution gate. Runs
alongside the broad paper.py tracker so the model-accuracy leaderboard keeps
its full sample while this stream isolates strategy PnL from execution
friction (NegRisk allowance failures, balance issues, etc.).

Mirrored gates (all applied per pick):
  main.py:    score >= EXEC_SCORE_MIN, ai_agree_frac >= AI_AGREE_MIN, whale.has_signal,
              48h market dedup, 24h question dedup
  execution:  CLOB best-ask available, YES divergence guard (ask <= yes_price * 1.5),
              flat $10 bet, min 5 tokens

State files:
  data/swarm_paper_mirror.jsonl           — one record per simulated trade
  data/swarm_paper_mirror_executed.json   — 48h dedup keyed by market_id
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

MIRROR_FILE      = Path("data/swarm_paper_mirror.jsonl")
MIRROR_EXECUTED  = Path("data/swarm_paper_mirror_executed.json")
SIM_BET          = 10.0    # matches execution.EXEC_FLAT_BET
MIN_TOKENS       = 5.0     # matches execution.EXEC_MIN_TOKENS
MAKER_OFFSET     = 0.012   # matches execution._SWARM_MAKER_OFFSET — mirror books fills at ask-offset
EXEC_TTL_H       = 48
QUESTION_DEDUP_H = 24
RESOLVED_HI      = 0.97
RESOLVED_LO      = 0.03
CLOB_HOST        = "https://clob.polymarket.com"


# ── State persistence ─────────────────────────────────────────────────────────

def _load_trades() -> list[dict]:
    if not MIRROR_FILE.exists():
        return []
    out = []
    with open(MIRROR_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _save_trades(trades: list[dict]) -> None:
    MIRROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MIRROR_FILE, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def _load_executed() -> dict[str, dict]:
    if not MIRROR_EXECUTED.exists():
        return {}
    try:
        raw = json.loads(MIRROR_EXECUTED.read_text())
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=EXEC_TTL_H)
        out: dict[str, dict] = {}
        for mid, info in raw.items():
            try:
                if datetime.fromisoformat(info["ts"]) > cutoff:
                    out[mid] = info
            except (KeyError, ValueError):
                pass
        return out
    except Exception:
        return {}


def _save_executed(records: dict[str, dict]) -> None:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=EXEC_TTL_H)
    pruned = {}
    for mid, info in records.items():
        try:
            if datetime.fromisoformat(info["ts"]) > cutoff:
                pruned[mid] = info
        except (KeyError, ValueError):
            pass
    MIRROR_EXECUTED.parent.mkdir(parents=True, exist_ok=True)
    MIRROR_EXECUTED.write_text(json.dumps(pruned, indent=2))


def _recent_questions(trades: list[dict]) -> set[str]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=QUESTION_DEDUP_H)
    out: set[str] = set()
    for t in trades:
        ts_str = t.get("entry_ts", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts > cutoff and t.get("question"):
                out.add(t["question"])
        except ValueError:
            pass
    return out


# ── CLOB / Gamma fetches ──────────────────────────────────────────────────────

async def _get_best_ask(token_id: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=8) as h:
            r = await h.get(f"{CLOB_HOST}/book", params={"token_id": token_id})
            r.raise_for_status()
            asks = r.json().get("asks", [])
            return min(float(a["price"]) for a in asks) if asks else None
    except Exception:
        return None


async def _fetch_yes_price(market_id: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=8) as h:
            r = await h.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
            r.raise_for_status()
            raw = r.json().get("outcomePrices", "[0.5,0.5]")
            prices = json.loads(raw) if isinstance(raw, str) else raw
            return float(prices[0])
    except Exception:
        return None


# ── Settlement ────────────────────────────────────────────────────────────────

async def _settle_pending(trades: list[dict]) -> int:
    n = 0
    for t in trades:
        if t.get("settled"):
            continue
        price = await _fetch_yes_price(t["market_id"])
        if price is None:
            continue
        yes_resolved = price >= RESOLVED_HI
        no_resolved  = price <= RESOLVED_LO
        if not yes_resolved and not no_resolved:
            continue
        direction = t["direction"]
        ask = float(t["ask"])
        bet = float(t["bet"])
        won = yes_resolved if direction == "YES" else no_resolved
        pnl = round(bet * (1.0 - ask) / ask, 4) if won else round(-bet, 4)
        t["settled"]    = True
        t["outcome"]    = "WIN" if won else "LOSS"
        t["exit_price"] = round(price, 4)
        t["exit_ts"]    = datetime.now(tz=timezone.utc).isoformat()
        t["pnl"]        = pnl
        n += 1
        log.info("Mirror settled: %s %s pnl=$%+.4f  %s",
                 t["outcome"], direction, pnl, t["question"][:55])
    return n


# ── Gate evaluation ───────────────────────────────────────────────────────────

def _gates_pass(pick) -> tuple[bool, str]:
    from bot.swarm.consensus import EXEC_SCORE_MIN, AI_AGREE_MIN
    if pick.score < EXEC_SCORE_MIN:
        return False, "score"
    if pick.ai_agree_frac < AI_AGREE_MIN:
        return False, "agree"
    if not (pick.whale and pick.whale.has_signal):
        return False, "whale"
    return True, ""


# ── Public API ────────────────────────────────────────────────────────────────

async def record_paper_mirror(picks) -> None:
    """Apply real-bot gates and log $10-simulated trades for the picks that pass."""
    from bot.swarm.models import Decision

    trades   = _load_trades()
    executed = _load_executed()
    n_settled = await _settle_pending(trades)
    recent_qs = _recent_questions(trades)

    n_new = 0
    now = datetime.now(tz=timezone.utc)

    for pick in picks:
        mid = pick.market.id
        if mid in executed:
            continue
        if pick.market.question in recent_qs:
            continue
        ok, _ = _gates_pass(pick)
        if not ok:
            continue

        # Mirror live YES direction block (execution.py:209, re-instated 2026-05-09).
        if pick.direction == Decision.YES:
            continue

        token_id = (pick.market.no_token_id if pick.direction == Decision.NO
                    else pick.market.yes_token_id)
        if not token_id:
            continue
        ask = await _get_best_ask(token_id)
        if ask is None:
            continue
        if pick.direction == Decision.YES and ask > pick.market.yes_price * 1.5:
            continue

        # Mirror live maker offset (execution.py:271): live posts a maker bid at
        # ask - MAKER_OFFSET first. Assume it fills — overstates slightly when the
        # maker times out and falls back to taker, but matches the design target.
        entry_price = round(ask - MAKER_OFFSET, 4)
        if entry_price <= 0:
            continue

        bet  = SIM_BET
        size = math.ceil(bet / entry_price * 10000) / 10000
        if size < MIN_TOKENS:
            continue

        rec = {
            "market_id":   mid,
            "question":    pick.market.question,
            "direction":   pick.direction.value,
            "entry_ts":    now.isoformat(),
            "ask":         entry_price,
            "bet":         round(bet, 4),
            "yes_price":   round(pick.market.yes_price, 4),
            "score":       round(pick.score, 4),
            "synthesis_confidence": getattr(pick, "synthesis_confidence", None),
            "avg_confidence": round(pick.avg_confidence, 1) if pick.avg_confidence else None,
            "yes_votes":   pick.yes_votes,
            "no_votes":    pick.no_votes,
            "whale_strength": pick.whale.strength if pick.whale else None,
            "settled":     False,
            "outcome":     None,
            "exit_price":  None,
            "exit_ts":     None,
            "pnl":         None,
        }
        trades.append(rec)
        executed[mid] = {"ts": now.isoformat(), "ask": entry_price, "count": 1}
        n_new += 1
        log.info("Mirror recorded: %s ask=%.3f bet=$%.2f  %s",
                 pick.direction.value, entry_price, bet, pick.market.question[:55])

    if n_new or n_settled:
        _save_trades(trades)
        _save_executed(executed)


def mirror_stats() -> dict:
    """Return mirror_* fields for swarm_state.json."""
    from datetime import date as _date
    trades = _load_trades()
    settled = [t for t in trades if t.get("settled")]
    open_t  = [t for t in trades if not t.get("settled")]
    wins    = [t for t in settled if t.get("outcome") == "WIN"]
    pnl     = round(sum(t.get("pnl", 0) for t in settled), 4)
    today   = _date.today().isoformat()
    daily   = round(sum(
        t.get("pnl", 0) for t in settled
        if (t.get("exit_ts") or "")[:10] == today
    ), 4)
    wr = round(len(wins) / len(settled) * 100, 1) if settled else None
    return {
        "mirror_open_count":    len(open_t),
        "mirror_settled_count": len(settled),
        "mirror_win_rate":      wr,
        "mirror_pnl":           pnl,
        "mirror_daily_pnl":     daily,
        "mirror_open_picks":    open_t[-20:],
        "mirror_closed_picks":  sorted(
            settled, key=lambda d: d.get("exit_ts") or "", reverse=True
        )[:30],
    }
