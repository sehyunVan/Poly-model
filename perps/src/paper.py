"""
perps/src/paper.py — paper-trading harness for HL perp signal validation.

A "paper trade" is opened in memory when the signal fires, held for a fixed
duration (hold_seconds), and closed at the prevailing mid. Entry and exit
both pay fee + slippage. Funding accrues proportional to hold duration.

State (kept in data/perps_state.json):
  - open positions: [{trade_id, coin, side, entry_ts, entry_mid, entry_fill, ...}]
  - closed positions counter + cumulative PnL
  - last_close_ts per coin (for cooldown)

Each entry+exit also appends one line to data/perps_paper.jsonl for later analysis.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger("perps.paper")


@dataclass
class PaperPosition:
    trade_id:        str
    coin:            str
    side:            str        # "LONG" or "SHORT"
    entry_ts:        float
    entry_mid:       float
    entry_fill:      float      # mid + slippage*tick (LONG pays up) or mid - slippage*tick (SHORT receives less)
    notional_usd:    float
    qty:             float      # notional / entry_fill
    score:           float
    components:      dict       # per-component scores at entry (drift, ob, cvd, lag, fund, mom30)
    funding_rate:    float      # snapshot at entry
    hold_seconds:    int        # configured hold duration


@dataclass
class PaperState:
    open_positions:  list[PaperPosition] = field(default_factory=list)
    closed_count:    int   = 0
    win_count:       int   = 0
    cumulative_pnl:  float = 0.0
    last_close_ts:   dict  = field(default_factory=dict)   # coin → ts
    next_trade_id:   int   = 1


# ── State persistence ─────────────────────────────────────────────────────

def _state_to_dict(s: PaperState) -> dict:
    return {
        "open_positions": [asdict(p) for p in s.open_positions],
        "closed_count":   s.closed_count,
        "win_count":      s.win_count,
        "cumulative_pnl": s.cumulative_pnl,
        "last_close_ts":  s.last_close_ts,
        "next_trade_id":  s.next_trade_id,
    }


def _dict_to_state(d: dict) -> PaperState:
    s = PaperState()
    s.open_positions  = [PaperPosition(**p) for p in d.get("open_positions", [])]
    s.closed_count    = int(d.get("closed_count", 0))
    s.win_count       = int(d.get("win_count", 0))
    s.cumulative_pnl  = float(d.get("cumulative_pnl", 0.0))
    s.last_close_ts   = dict(d.get("last_close_ts", {}))
    s.next_trade_id   = int(d.get("next_trade_id", 1))
    return s


def load_state(path: str) -> PaperState:
    p = Path(path)
    if not p.exists():
        return PaperState()
    try:
        return _dict_to_state(json.loads(p.read_text()))
    except Exception as exc:
        _log.warning("Could not load state %s — starting fresh (%s)", path, exc)
        return PaperState()


def save_state(state: PaperState, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_state_to_dict(state), indent=2))
    os.replace(tmp, p)


# ── Paper trade lifecycle ──────────────────────────────────────────────────

def can_open(
    state: PaperState,
    coin: str,
    now: float,
    cooldown_seconds: int,
    max_concurrent: int,
) -> tuple[bool, str]:
    """Return (allowed, reason). Reason is a short human-readable code."""
    open_for_coin = [p for p in state.open_positions if p.coin == coin]
    if open_for_coin:
        return False, "already_open"
    if len(state.open_positions) >= max_concurrent:
        return False, "concurrent_cap"
    last = state.last_close_ts.get(coin, 0.0)
    if now - last < cooldown_seconds:
        return False, "cooldown"
    return True, "ok"


def open_position(
    state: PaperState,
    *,
    coin: str,
    side: str,
    score: float,
    components: dict,
    mid: float,
    funding_rate: float,
    now: float,
    notional_usd: float,
    slippage_ticks: int,
    tick_size: float,
    fee_rate: float,
    hold_seconds: int,
    log_path: str,
) -> PaperPosition:
    """Open a paper position. Entry fill is mid worsened by slippage_ticks."""
    direction_sign = +1 if side == "LONG" else -1
    entry_fill = mid + direction_sign * slippage_ticks * tick_size
    qty = notional_usd / entry_fill if entry_fill > 0 else 0.0
    trade_id = f"t{state.next_trade_id:06d}"
    state.next_trade_id += 1

    pos = PaperPosition(
        trade_id      = trade_id,
        coin          = coin,
        side          = side,
        entry_ts      = now,
        entry_mid     = mid,
        entry_fill    = entry_fill,
        notional_usd  = notional_usd,
        qty           = qty,
        score         = score,
        components    = components,
        funding_rate  = funding_rate,
        hold_seconds  = hold_seconds,
    )
    state.open_positions.append(pos)

    _log.info(
        "OPEN  %s %s %s qty=%.6f mid=$%.2f fill=$%.2f score=%+.3f",
        trade_id, side, coin, qty, mid, entry_fill, score,
    )
    _append_log(log_path, {
        "event":         "open",
        "ts":            now,
        "trade_id":      trade_id,
        "coin":          coin,
        "side":          side,
        "score":         score,
        "components":    components,
        "mid":           mid,
        "entry_fill":    entry_fill,
        "qty":           qty,
        "notional_usd":  notional_usd,
        "funding_rate":  funding_rate,
        "hold_seconds":  hold_seconds,
    })
    return pos


def should_close(pos: PaperPosition, now: float) -> bool:
    return (now - pos.entry_ts) >= pos.hold_seconds


def close_position(
    state: PaperState,
    pos: PaperPosition,
    *,
    mid: float,
    now: float,
    slippage_ticks: int,
    tick_size: float,
    fee_rate: float,
    funding_per_hour: bool,
    log_path: str,
) -> float:
    """Close a paper position at the prevailing mid. Returns net PnL."""
    direction_sign = +1 if pos.side == "LONG" else -1
    # exit fill: LONG sells at bid (worse than mid), SHORT buys at ask (worse than mid)
    exit_fill = mid - direction_sign * slippage_ticks * tick_size

    # Notional PnL: qty * (exit - entry) * sign
    price_pnl = pos.qty * (exit_fill - pos.entry_fill) * direction_sign

    # Fees on both legs (taker, paid on notional)
    fee_entry = pos.notional_usd * fee_rate
    fee_exit  = pos.qty * exit_fill * fee_rate
    fee_total = fee_entry + fee_exit

    # Funding cost: paid by LONG when funding > 0, paid to LONG when < 0.
    # HL is hourly funding (`funding_per_hour=true` in config).
    hours_held = (now - pos.entry_ts) / 3600.0
    # funding_rate is per-period; on HL that's per-hour. SHORT gets paid when funding positive.
    funding_pnl = -direction_sign * pos.notional_usd * pos.funding_rate * hours_held

    net_pnl = price_pnl - fee_total + funding_pnl

    state.open_positions = [p for p in state.open_positions if p.trade_id != pos.trade_id]
    state.closed_count += 1
    if net_pnl > 0:
        state.win_count += 1
    state.cumulative_pnl += net_pnl
    state.last_close_ts[pos.coin] = now

    _log.info(
        "CLOSE %s %s %s held=%.0fs mid=$%.2f fill=$%.2f price_pnl=%+.4f fee=%.4f funding=%+.4f NET=%+.4f  cum=%+.2f wr=%.1f%% (n=%d)",
        pos.trade_id, pos.side, pos.coin, now - pos.entry_ts,
        mid, exit_fill, price_pnl, fee_total, funding_pnl, net_pnl,
        state.cumulative_pnl,
        (state.win_count / state.closed_count * 100) if state.closed_count else 0.0,
        state.closed_count,
    )
    _append_log(log_path, {
        "event":          "close",
        "ts":             now,
        "trade_id":       pos.trade_id,
        "coin":           pos.coin,
        "side":           pos.side,
        "entry_ts":       pos.entry_ts,
        "entry_fill":     pos.entry_fill,
        "exit_mid":       mid,
        "exit_fill":      exit_fill,
        "qty":            pos.qty,
        "notional_usd":   pos.notional_usd,
        "hours_held":     hours_held,
        "price_pnl":      price_pnl,
        "fee_total":      fee_total,
        "funding_pnl":    funding_pnl,
        "net_pnl":        net_pnl,
        "score":          pos.score,
        "components":     pos.components,
    })
    return net_pnl


# ── Internal ───────────────────────────────────────────────────────────────

def _append_log(path: str, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
