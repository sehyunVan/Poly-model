"""Ghost log for out-of-band high-conviction NO picks (2026-06-11).

Tests the "timed early-entry" hypothesis at ZERO capital risk.

The swarm is whale-triggered and engine-zone-gated, so a NO pick that passes
every conviction gate (whale, alignment, AI-agree, avg-conf, score, deepseek
veto, NO direction) but whose CLOB NO ask is outside [0.70, 0.78) is currently
SKIPPED. This module records each such skip and then snapshots its NO-ask each
cycle until it settles, answering:

  - Of skipped picks, how many later pass through the engine zone [0.70, 0.78)?
  - From which direction — rising in from below (momentum-confirmed, good) or
    falling in from above (thesis weakening = adverse selection, bad)?
  - How would a flat $40 NO bet placed at that first zone-touch have settled?

If the data says rising-into-zone picks win, building a live timed-entry monitor
is justified. If not, we've learned it for free.

Hooks:
  - bot/execution.py  → record_ghost(pick, ask)  at the two band-reject points
  - bot/main.py       → await update_ghosts()     once per cycle
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

GHOST_FILE = Path(__file__).resolve().parent.parent / "data" / "swarm_ghost_band.jsonl"

# Engine zone (must match the band gate in bot/execution.py).
ZONE_LO = 0.70
ZONE_HI = 0.78
GHOST_BET = 40.0          # hypothetical flat bet for PnL-if-entered
_MAX_TRAJ = 300           # cap stored trajectory points (min/max preserved separately)
_MAX_PER_CYCLE = 40       # safety cap on ghosts touched per cycle (volume is low)

# Settlement thresholds — mirror bot/paper.py.
_RESOLVED_HI = 0.97
_RESOLVED_LO = 0.03


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load() -> list[dict]:
    if not GHOST_FILE.exists():
        return []
    out = []
    for line in GHOST_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _save(rows: list[dict]) -> None:
    GHOST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GHOST_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record_ghost(pick, ask: float) -> None:
    """Record one out-of-band high-conviction NO pick. Deduped by market_id while
    an entry for that market is still unsettled."""
    try:
        mkt = pick.market
        rows = _load()
        for r in rows:
            if r.get("market_id") == mkt.id and not r.get("settled"):
                return  # already tracking this market
        reason = "below_zone" if ask < ZONE_LO else "above_zone"
        row = {
            "market_id":      mkt.id,
            "condition_id":   getattr(mkt, "condition_id", None),
            "question":       mkt.question,
            "no_token_id":    getattr(mkt, "no_token_id", "") or "",
            "first_seen_ts":  _now(),
            "first_ask":      round(ask, 4),
            "skip_reason":    reason,
            "hours_to_close": round(mkt.hours_to_close, 2) if getattr(mkt, "hours_to_close", None) is not None else None,
            "score":          round(pick.score, 4) if getattr(pick, "score", None) is not None else None,
            "avg_confidence": round(pick.avg_confidence, 1) if getattr(pick, "avg_confidence", None) else None,
            "yes_votes":      getattr(pick, "yes_votes", None),
            "no_votes":       getattr(pick, "no_votes", None),
            "whale_strength": getattr(pick, "whale_strength", None),
            # trajectory + zone-entry tracking
            "min_ask":        round(ask, 4),
            "max_ask":        round(ask, 4),
            "trajectory":     [{"ts": _now(), "ask": round(ask, 4)}],
            "last_update":    _now(),
            "entered_zone":   ZONE_LO <= ask < ZONE_HI,  # should be False (it was rejected), guarded anyway
            "entered_ts":     None,
            "entered_ask":    None,
            "entered_from":   None,
            # settlement
            "settled":        False,
            "outcome":        None,        # WIN/LOSS for a hypothetical NO bet
            "exit_price":     None,
            "exit_ts":        None,
            "hypo_pnl":       None,        # $40 NO bet at entered_ask, if entered
        }
        rows.append(row)
        _save(rows)
        log.info("GHOST record: NO ask=%.3f (%s) htc=%sh  %s",
                 ask, reason, row["hours_to_close"], mkt.question[:50])
    except Exception as e:
        log.warning("ghost record failed: %s", e)


async def update_ghosts() -> None:
    """Once per cycle: snapshot CLOB NO ask for pre-entry ghosts, detect zone
    entry + direction, and settle resolved markets. Bounded, best-effort —
    never raises into the swarm loop."""
    try:
        from bot.execution import _get_best_ask   # lazy import (avoids cycle)
        from bot.paper import _fetch_yes_price
    except Exception as e:
        log.warning("ghost update import failed: %s", e)
        return

    rows = _load()
    open_rows = [r for r in rows if not r.get("settled")]
    if not open_rows:
        return

    # Round-robin fairness: oldest last_update first, capped per cycle.
    open_rows.sort(key=lambda r: r.get("last_update") or "")
    touched = 0
    changed = False

    for r in open_rows:
        if touched >= _MAX_PER_CYCLE:
            break
        touched += 1
        r["last_update"] = _now()
        changed = True

        # 1) Snapshot ask + detect zone entry (only while not yet entered).
        if not r.get("entered_zone") and r.get("no_token_id"):
            try:
                ask = await _get_best_ask(r["no_token_id"])
            except Exception:
                ask = None
            if ask is not None:
                traj = r.setdefault("trajectory", [])
                traj.append({"ts": _now(), "ask": round(ask, 4)})
                if len(traj) > _MAX_TRAJ:
                    del traj[: len(traj) - _MAX_TRAJ]
                r["min_ask"] = round(min(r.get("min_ask", ask), ask), 4)
                r["max_ask"] = round(max(r.get("max_ask", ask), ask), 4)
                if ZONE_LO <= ask < ZONE_HI:
                    r["entered_zone"] = True
                    r["entered_ts"]   = _now()
                    r["entered_ask"]  = round(ask, 4)
                    # below_zone skip rises INTO the band (momentum); above_zone falls in (adverse)
                    r["entered_from"] = "below" if r.get("skip_reason") == "below_zone" else "above"
                    log.info("GHOST entered zone from %s: ask=%.3f  %s",
                             r["entered_from"], ask, r["question"][:50])

        # 2) Settlement check.
        price = None
        try:
            price = _fetch_yes_price(r["market_id"])
        except Exception:
            price = None
        if price is None:
            continue
        if price >= _RESOLVED_HI:
            no_won = False   # YES resolved
        elif price <= _RESOLVED_LO:
            no_won = True    # NO resolved
        else:
            continue         # not resolved yet

        r["settled"]    = True
        r["outcome"]    = "WIN" if no_won else "LOSS"   # for a NO bet
        r["exit_price"] = round(price, 4)
        r["exit_ts"]    = _now()
        if r.get("entered_zone") and r.get("entered_ask"):
            ea = float(r["entered_ask"])
            r["hypo_pnl"] = round(GHOST_BET * (1.0 - ea) / ea, 4) if no_won else round(-GHOST_BET, 4)
        log.info("GHOST settled: NO %s entered=%s hypo_pnl=%s  %s",
                 r["outcome"], r.get("entered_zone"), r.get("hypo_pnl"), r["question"][:45])

    if changed:
        _save(rows)
