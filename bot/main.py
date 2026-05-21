"""
Main loop — runs every CYCLE_SECONDS.

Cycle:
  1. Fetch active markets from Gamma API
  2. Filter (price band, volume, time, category)
  3. Run AI swarm on filtered candidates
  4. Print top-N consensus picks
  5. (execution hook — wired in later)
  6. Sleep until next cycle
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot.config import ACTIVE_MODELS, CONSENSUS_TOP_N, CYCLE_SECONDS
from bot.data.filter import apply_filters
from bot.data.markets import fetch_markets
from bot.data.whale import whale_feed
from bot.execution import EXEC_MIN_SCORE, execute_pick
from bot.paper import flush_state, record_picks, record_real_execution
from bot.paper_mirror import record_paper_mirror
from bot.swarm.consensus import AI_AGREE_MIN, EXEC_SCORE_MIN, MarketConsensus, run_swarm

# Force UTF-8 stdout on Windows (Korean locale uses cp949 by default)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bot.main")

_EXECUTED_FILE    = Path("data/swarm_executed.json")
_EXECUTED_TTL_H   = 48   # prune entries older than this many hours
_REENTRY_MIN_S    = 45 * 60  # minimum seconds between first and second entry on same market
_REENTRY_MAX_COUNT = 2       # max total entries per market within TTL window

# Markets where execution was declined this session (YES blocked, NO ask > ceiling, etc.)
# Prevents re-running expensive synthesis every cycle for the same unexecutable market.
# In-memory only — intentionally short-lived so a market can retry next session if conditions change.
_DECLINED: dict[str, float] = {}   # market_id → monotonic time of decline
_DECLINED_TTL_S = 2 * 3600         # 2 hours

# Question-level dedup: prevents betting the same matchup twice in consecutive days when they share
# the same question text but different market IDs (e.g. "Nationals vs Pirates" Apr 15 and Apr 16).
_REAL_TRADES_FILE = Path("data/swarm_real_trades.jsonl")
_QUESTION_DEDUP_H = 24  # block same question for 24h after any execution (win or loss)


def _load_recent_questions() -> set[str]:
    """Return the set of question strings executed within the past QUESTION_DEDUP_H hours."""
    if not _REAL_TRADES_FILE.exists():
        return set()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_QUESTION_DEDUP_H)
    recent: set[str] = set()
    try:
        for line in _REAL_TRADES_FILE.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            ts_str = t.get("entry_ts", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if ts > cutoff:
                q = t.get("question", "")
                if q:
                    recent.add(q)
    except Exception as e:
        log.warning("Could not load recent questions for dedup: %s", e)
    return recent


# ── Executed-market persistence ───────────────────────────────────────────────

def _load_executed() -> dict[str, dict]:
    """
    Load executed records from disk.
    Returns {market_id: {"ts": iso_str, "ask": float|None, "count": int}}.
    Handles both the old format {market_id: timestamp_str} and the new rich format.
    """
    if not _EXECUTED_FILE.exists():
        return {}
    try:
        raw = json.loads(_EXECUTED_FILE.read_text())
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_EXECUTED_TTL_H)
        result: dict[str, dict] = {}
        for mid, val in raw.items():
            if isinstance(val, str):
                # Backward compat: old format stored only a timestamp string
                info = {"ts": val, "ask": None, "count": 1}
            else:
                info = val
            try:
                ts = datetime.fromisoformat(info["ts"])
                if ts > cutoff:
                    result[mid] = info
            except (KeyError, ValueError):
                pass
        return result
    except Exception as e:
        log.warning("Could not load executed markets: %s", e)
        return {}


def _save_executed(records: dict[str, dict]) -> None:
    """Persist executed records to disk, pruning entries older than TTL."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_EXECUTED_TTL_H)
    pruned = {}
    for mid, info in records.items():
        try:
            if datetime.fromisoformat(info["ts"]) > cutoff:
                pruned[mid] = info
        except (KeyError, ValueError):
            pass
    try:
        _EXECUTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        _EXECUTED_FILE.write_text(json.dumps(pruned, indent=2))
    except Exception as e:
        log.warning("Could not save executed markets: %s", e)


# ── Display ───────────────────────────────────────────────────────────────────

def _print_picks(picks: list[MarketConsensus], cycle: int) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*70}")
    print(f"  CYCLE {cycle}  |  {ts}  |  {len(ACTIVE_MODELS)} models")
    print(f"{'='*70}")

    if not picks:
        print("  No consensus picks this cycle.")
        print(f"{'='*70}\n")
        return

    for rank, p in enumerate(picks, 1):
        m = p.market
        side_str = "YES ^" if p.direction.value == "YES" else "NO  v"
        print(
            f"\n  #{rank}  {side_str}  score={p.score:.3f}  "
            f"whale={p.whale_score:.2f}  ai={p.ai_score:.2f}  "
            f"agree={p.ai_agree_frac:.0%}  conf={p.avg_confidence:.0f}%"
        )
        print(f"      {m.question}")
        print(
            f"      YES={m.yes_price:.3f}  NO={m.no_price:.3f}  "
            f"vol24h=${m.volume_24h:,.0f}  closes_in={m.hours_to_close:.1f}h"
        )
        w = p.whale
        if w:
            print(
                f"      [whale] {w.direction}  YES_buy=${w.yes_flow_usdc:,.0f}  "
                f"NO_buy=${w.no_flow_usdc:,.0f}  "
                f"strength={w.strength:.2f}  {w.note}  [{w.source}]"
            )
        for v in p.verdicts:
            flag = "+" if v.decision == p.direction else ("." if v.decision.value == "NO_TRADE" else "x")
            print(f"      {flag} {v.model_name:<14}  {v.decision.value:<8}  {v.confidence:>3}%  {v.reasoning[:60]}")

    print(f"\n{'='*70}\n")


# ── One cycle ─────────────────────────────────────────────────────────────────

async def run_cycle(cycle: int, executed_records: dict[str, dict]) -> None:
    # Load question-level dedup set fresh each cycle so it sees trades placed this session
    recent_questions = _load_recent_questions()

    # 1. Fetch
    all_markets = await fetch_markets(limit=200)
    if not all_markets:
        log.warning("No markets fetched -- skipping cycle.")
        return

    # 2. Filter
    candidates = apply_filters(all_markets)
    if not candidates:
        log.info("No markets passed filters this cycle.")
        return

    # 3. Swarm
    picks = await run_swarm(candidates)

    # 4. Display
    _print_picks(picks, cycle)

    # 5. Paper trading -- settle resolved markets and record new picks
    if picks:
        record_picks(picks)
        # Strategy-mirror tracker: applies all real-bot gates so its PnL isolates
        # strategy quality from execution friction (NegRisk allowance, balance, etc.)
        await record_paper_mirror(picks)

    # 6. Real execution — whale-triggered, AI-validated.
    #    First entry: standard flow (once per market per question within TTL).
    #    Re-entry: allowed once more if ask improved ≥4% and ≥45 min elapsed.
    now_mono = time.monotonic()
    now_dt   = datetime.now(tz=timezone.utc)
    for pick in picks:
        mid        = pick.market.id
        entry_info = executed_records.get(mid)
        reentry_info: dict | None = None

        if entry_info is not None:
            count = entry_info.get("count", 1)
            if count >= _REENTRY_MAX_COUNT:
                log.info("SKIP exec: max %d entries reached for %s", _REENTRY_MAX_COUNT, mid[:16])
                continue
            # count == 1 and initial ask was recorded: check re-entry eligibility
            if entry_info.get("ask") is None:
                # Old-format entry — no ask stored, can't check improvement; treat as done.
                log.info("SKIP exec: already traded %s (no ask for re-entry check)", mid[:16])
                continue
            elapsed = (now_dt - datetime.fromisoformat(entry_info["ts"])).total_seconds()
            if elapsed < _REENTRY_MIN_S:
                log.info(
                    "SKIP re-entry: only %.0fm elapsed (min %.0fm)  %s",
                    elapsed / 60, _REENTRY_MIN_S / 60, mid[:16],
                )
                continue
            log.info(
                "RE-ENTRY ELIGIBLE: market=%s  initial_ask=%.3f  elapsed=%.0fm",
                mid[:16], entry_info["ask"], elapsed / 60,
            )
            reentry_info = entry_info
        else:
            # First entry: apply question-level dedup (same matchup on consecutive days).
            mkt_question = pick.market.question
            if mkt_question in recent_questions:
                log.info(
                    "SKIP exec: question traded in last %dh (dedup) — %s",
                    _QUESTION_DEDUP_H, mkt_question[:55],
                )
                continue

        # Skip markets recently declined by execution.py (ceiling, balance, etc.).
        if mid in _DECLINED and now_mono - _DECLINED[mid] < _DECLINED_TTL_S:
            log.info("SKIP exec: declined recently, cooling off  %s", mid[:16])
            continue

        gate_score = pick.score >= EXEC_SCORE_MIN
        gate_agree = pick.ai_agree_frac >= AI_AGREE_MIN
        gate_whale = pick.whale and pick.whale.has_signal
        if gate_score and gate_agree and gate_whale:
            action = "RE-ENTRY" if reentry_info else "EXECUTING"
            log.info(
                "%s: %s  score=%.3f  agree=%.0f%%  whale=%s(%.2f)  %s",
                action, pick.direction.value, pick.score, pick.ai_agree_frac * 100,
                pick.whale.direction, pick.whale.strength, pick.market.question[:50],
            )
            fill = await execute_pick(pick, reentry_info=reentry_info)
            if fill:
                prev_count = entry_info.get("count", 0) if entry_info else 0
                executed_records[mid] = {
                    "ts":    now_dt.isoformat(),
                    "ask":   fill["ask"],
                    "count": prev_count + 1,
                }
                _save_executed(executed_records)
                record_real_execution(pick, fill["ask"], fill["bet"], direction=fill.get("direction"))
            else:
                # Declined — cool off only for first entries (re-entry declines are
                # normally "ask didn't improve", which may change next cycle — no cooloff).
                if reentry_info is None:
                    _DECLINED[mid] = now_mono
                    log.info("EXEC declined for %s — cooling off 2h", mid[:16])
        else:
            log.info(
                "SKIP exec: score=%.3f agree=%.0f%% whale=%s  %s",
                pick.score, pick.ai_agree_frac * 100,
                pick.whale.direction if pick.whale else "NONE",
                pick.market.question[:50],
            )

    # 7. Write swarm_state.json AFTER both paper tracking and real execution
    # flush_state loads paper trades fresh from disk so real trades are always included
    flush_state(picks)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("Bot starting -- %d active models: %s",
             len(ACTIVE_MODELS), [m["name"] for m in ACTIVE_MODELS])

    if not ACTIVE_MODELS:
        log.error("No API keys configured. Set at least ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    # Load previously executed markets from disk (survives restarts).
    # Format: {market_id: {"ts": iso_str, "ask": float, "count": int}}
    # Re-entry is allowed once per market when ask improves ≥4% and ≥45 min elapsed.
    _executed_records: dict[str, dict] = _load_executed()
    log.info("Loaded %d previously executed markets (within %dh TTL)",
             len(_executed_records), _EXECUTED_TTL_H)

    # Start the persistent Polymarket CLOB WebSocket trade feed
    whale_feed.start()
    log.info("WhaleFeed started — will subscribe to candidate tokens each cycle")

    cycle = 0
    while True:
        cycle += 1
        t0 = time.monotonic()
        try:
            await run_cycle(cycle, _executed_records)
        except Exception as e:
            log.exception("Cycle %d crashed: %s", cycle, e)

        elapsed = time.monotonic() - t0
        sleep   = max(0, CYCLE_SECONDS - elapsed)
        log.info("Cycle %d done in %.1fs -- sleeping %.0fs", cycle, elapsed, sleep)
        await asyncio.sleep(sleep)


if __name__ == "__main__":
    asyncio.run(main())
