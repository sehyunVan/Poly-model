#!/usr/bin/env python3
"""
src/dashboard.py

Lightweight trading dashboard — serves a live HTML page with charts
showing portfolio state, budget allocation, category breakdown, and PnL metrics.

Works in both VIRTUAL_MODE and real mode (reads data/virtual_state.json).

Usage:
    python src/dashboard.py              # http://localhost:8765
    python src/dashboard.py --port 9000
    python src/dashboard.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent

try:
    from dotenv import load_dotenv
    for _p in [_ROOT, _ROOT / "polymarket-mcp-main" / "polymarket-mcp-main"]:
        if (_p / ".env").exists():
            load_dotenv(_p / ".env")
            break
except ImportError:
    pass

_VIRTUAL_MODE = os.getenv("VIRTUAL_MODE", "true").lower() != "false"
_VIRTUAL_STATE_PATH = os.getenv("VIRTUAL_STATE_PATH", "data/virtual_state.json")
_REAL_STATE_PATH    = "data/real_state.json"
_1H_REAL_STATE_PATH = "data/1h_real_state.json"
_1H_VIRTUAL_PATH    = "data/1h_virtual_state.json"
_SWARM_STATE_PATH   = "data/swarm_state.json"
_ARB_STATE_PATH     = "funding_arb/data/arb_state.json"
_ARB_LOG_PATH       = "funding_arb/logs/arb.log"
_CONVICTION_STATE_PATH = "data/conviction_state.json"
_CONVICTION_LOG_PATH   = "data/conviction_log.jsonl"
_STRIKE_STATE_PATH  = "data/strike_state.json"
_STRIKE_LOG_PATH    = "logs/strike.log"
_STRIKE_CFG_PATH    = "config/strike_params.yaml"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_raw_state() -> dict:
    """Load raw portfolio state from disk."""
    path = _ROOT / (_VIRTUAL_STATE_PATH if _VIRTUAL_MODE else _REAL_STATE_PATH)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": f"State file not found: {path}"}


def _load_swarm_state() -> dict:
    """Load swarm bot state from disk, enriched with pending CTF value AND on-chain
    positions cross-referenced with the swarm_real_trades.jsonl metadata."""
    path = _ROOT / _SWARM_STATE_PATH
    if not path.exists():
        return {"error": "Swarm state not found — swarm bot may not be running yet."}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}

    # ── On-chain enrichment ───────────────────────────────────────────────────
    # Open positions on-chain are the TRUTH; swarm_real_trades.jsonl is incomplete
    # because record_real_execution() only fires on confirmed fills (~20% of attempts).
    wallet = _load_wallet_state()

    # pending_ctf_usdc = won_redeemable + won_pending from live on-chain data.
    # The crypto loop's real_state.json:pending_ctf_usdc only refreshes on the 12h AR
    # cycle, so it goes stale between cycles and inflates net PnL. Use the live wallet
    # value when available; fall back to the stale state value if web3 is unreachable.
    if "error" not in wallet:
        pending_ctf = float(wallet.get("won_redeemable", 0.0)) + float(wallet.get("won_pending", 0.0))
    else:
        pending_ctf = 0.0
        try:
            rsp = _ROOT / _REAL_STATE_PATH
            if rsp.exists():
                rs = json.loads(rsp.read_text(encoding="utf-8"))
                pending_ctf = float(rs.get("pending_ctf_usdc", 0.0))
        except Exception:
            pass
    data["pending_ctf_usdc"] = round(pending_ctf, 2)
    data["real_net_pnl"] = round((data.get("real_pnl") or 0.0) + pending_ctf, 2)

    # Era split — always computed from JSONL regardless of wallet availability.
    # Splits cumulative real PnL into (a) live strategy = NO direction (current policy)
    # and (b) legacy YES era (frozen since 2026-04-11 YES-block). Also computes a
    # 14-day NO-only trend so recent performance is visible without legacy drag.
    jsonl_all: list[dict] = []
    jp = _ROOT / "data/swarm_real_trades.jsonl"
    if jp.exists():
        for line in jp.read_text(encoding="utf-8").splitlines():
            try:
                jsonl_all.append(json.loads(line))
            except Exception:
                pass

    def _bucket(rows: list[dict]) -> dict:
        settled = [r for r in rows if r.get("settled")]
        if not settled:
            return {"n": 0, "wr": None, "pnl": 0.0}
        wins = sum(1 for r in settled if (r.get("pnl") or 0) > 0)
        pnl  = sum((r.get("pnl") or 0) for r in settled)
        return {
            "n": len(settled),
            "wr": round(wins / len(settled) * 100, 1),
            "pnl": round(pnl, 2),
        }

    live = _bucket([r for r in jsonl_all if (r.get("direction") or "").upper() == "NO"])
    legacy = _bucket([r for r in jsonl_all if (r.get("direction") or "").upper() == "YES"])
    cutoff_14d = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    recent = _bucket([
        r for r in jsonl_all
        if (r.get("direction") or "").upper() == "NO"
        and (r.get("exit_ts") or "") >= cutoff_14d
    ])
    data["live_strategy"] = live
    data["legacy_yes"]    = legacy
    data["recent_14d_no"] = recent

    if "error" not in wallet:
        # Build a {question_lower: jsonl_record} index from JSONL for metadata lookup
        jsonl: list[dict] = jsonl_all
        # Latest record per (question, direction)
        meta_by_q: dict[tuple[str, str], dict] = {}
        for r in jsonl:
            q = (r.get("question") or "").strip().lower()
            d = (r.get("direction") or "").upper()
            if not q or not d:
                continue
            prev = meta_by_q.get((q, d))
            if prev is None or (r.get("entry_ts") or "") > (prev.get("entry_ts") or ""):
                meta_by_q[(q, d)] = r

        def enrich(items: list[dict]) -> list[dict]:
            out = []
            for it in items:
                q  = (it.get("title") or "").strip().lower()
                outcome = (it.get("outcome") or "").upper()
                meta = meta_by_q.get((q, outcome)) or meta_by_q.get((q, "YES")) or meta_by_q.get((q, "NO"))
                rec = dict(it)
                rec["tracked"] = bool(meta)
                if meta:
                    rec["entry_ts"]   = meta.get("entry_ts")
                    rec["entry_ask"]  = meta.get("ask")
                    rec["bet_usd"]    = meta.get("bet")
                    rec["score"]      = meta.get("score")
                    rec["synth_conf"] = meta.get("synthesis_confidence")
                    rec["yes_votes"]  = meta.get("yes_votes")
                    rec["no_votes"]   = meta.get("no_votes")
                # PnL "if win now": current value minus what we paid.
                # If untracked, we estimate cost as size × 0.5 (no entry record).
                bet = float(meta.get("bet", 0)) if meta else round(float(it.get("size", 0)) * 0.5, 2)
                rec["est_cost"]      = round(bet, 2)
                rec["est_pnl_now"]   = round(float(it.get("current_value", 0)) - bet, 2)
                out.append(rec)
            return out

        data["on_chain_in_play"]      = enrich(wallet.get("top_in_play", []))
        data["on_chain_won_pending"]  = enrich(wallet.get("top_won_pending", []))
        data["on_chain_in_play_count"]    = wallet.get("counts", {}).get("in_play", 0)
        data["on_chain_won_pending_count"] = wallet.get("counts", {}).get("won_pending", 0)
        data["on_chain_in_play_curval"]    = wallet.get("in_play_curval", 0)
        data["on_chain_won_pending_value"] = wallet.get("won_pending", 0)

    return data


def _load_1h_crypto_state() -> dict:
    """Load 1H crypto loop state from disk."""
    path = _ROOT / _1H_REAL_STATE_PATH
    if not path.exists():
        path = _ROOT / _1H_VIRTUAL_PATH
    if not path.exists():
        return {"error": "1H crypto state not found — loop may not be running yet."}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        closed = data.get("closed_positions", [])
        wins = sum(1 for p in closed if p.get("realized_pnl", 0) > 0)
        pnl = sum(p.get("realized_pnl", 0) for p in closed)
        wr = round(100 * wins / len(closed), 1) if closed else None
        return {
            "closed_trades": len(closed),
            "wins": wins,
            "hit_rate": wr,
            "pnl": round(pnl, 2),
            "mode": "LIVE" if path.name == "1h_real_state.json" else "VIRTUAL",
        }
    except Exception as exc:
        return {"error": str(exc)}


def _load_arb_state() -> dict:
    """Load funding rate arb state from disk."""
    path = _ROOT / _ARB_STATE_PATH
    if not path.exists():
        return {"error": "Arb state not found — arb bot may not be running yet."}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}

    # Derive status from positions dict (new multi-position schema)
    positions = raw.get("positions", {})
    active_symbols = list(positions.keys())
    status = "ACTIVE" if active_symbols else "IDLE"
    first_pos = positions[active_symbols[0]] if active_symbols else {}
    active_symbol = active_symbols[0] if active_symbols else ""

    # current_rates saved directly in state by main.py each cycle
    current_rates = raw.get("current_rates", {})

    # Compute pending_funding from state data (no log parsing needed)
    pending_funding = 0.0
    now = time.time()
    for sym, pos in positions.items():
        rate_data = current_rates.get(sym, {})
        rate = rate_data.get("rate", 0.0)
        elapsed = (now - pos.get("last_funding_collected_time", now)) / (8 * 3600)
        pending_funding += pos.get("position_usdt", 0.0) * rate * max(elapsed, 0)

    current_price = current_rates.get(active_symbol, {}).get("mark_price", 0.0)

    # Read config for virtual_mode and entry threshold
    arb_cfg_path = _ROOT / "funding_arb" / "config" / "arb_params.yaml"
    arb_virtual = True
    entry_threshold_apy = 5.48
    if arb_cfg_path.exists():
        try:
            cfg_text = arb_cfg_path.read_text(encoding="utf-8")
            m = re.search(r'^\s*virtual_mode\s*:\s*(true|false)', cfg_text, re.MULTILINE | re.IGNORECASE)
            arb_virtual = (m is None) or (m.group(1).lower() == "true")
            m2 = re.search(r'^\s*entry_funding_rate\s*:\s*([\d.]+)', cfg_text, re.MULTILINE)
            if m2:
                entry_threshold_apy = round(float(m2.group(1)) * 3 * 365 * 100, 2)
        except Exception:
            pass

    # Recent log: only event lines (skip the status block unicode art)
    recent_log: list[str] = []
    log_path = _ROOT / _ARB_LOG_PATH
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            keywords = ("ENTRY", "EXIT", "FUNDING", "APPROACH", "ERROR", "CRITICAL",
                        "Resuming", "====", "mode=")
            event_lines = [l for l in lines if any(k in l for k in keywords)]
            recent_log = event_lines[-15:]
        except Exception:
            pass

    return {
        "status":                   status,
        "active_symbol":            active_symbol,
        "mode":                     "VIRTUAL" if arb_virtual else "LIVE",
        "entry_price":              first_pos.get("entry_price", 0.0),
        "spot_qty":                 first_pos.get("spot_qty", 0.0),
        "futures_qty":              first_pos.get("futures_qty", 0.0),
        "position_usdt":            first_pos.get("position_usdt", 0.0),
        "entry_time":               first_pos.get("entry_time", 0.0),
        "positions":                positions,
        "total_funding_collected":  round(raw.get("total_funding_collected", 0.0), 6),
        "total_realized_pnl":       round(raw.get("total_realized_pnl", 0.0), 4),
        "trade_count":              raw.get("trade_count", 0),
        "error_count":              raw.get("error_count", 0),
        "current_rates":            current_rates,
        "current_price":            current_price,
        "pending_funding":          round(pending_funding, 6),
        "entry_threshold_apy":      entry_threshold_apy,
        "recent_log":               recent_log,
    }


def _load_conviction_state() -> dict:
    """Load market conviction bot state from disk.

    Live trades come from data/conviction_real_state.json (separate file, written
    by the live bot). Paper trades come from data/conviction_state.json.
    The live trade record (open + closed) is returned with full per-trade detail
    (order_id, fill_type, filled_usdc, etc.) — this replaces the arb area in the
    dashboard since the arb path barely fires.
    """
    # Paper state — may be corrupt (e.g., truncated mid-write); keep going either way.
    paper_path = _ROOT / _CONVICTION_STATE_PATH
    state: dict = {}
    paper_load_err = None
    if paper_path.exists():
        try:
            state = json.loads(paper_path.read_text(encoding="utf-8"))
        except Exception as exc:
            paper_load_err = str(exc)
            state = {}

    # Live state lives in a separate file so live PnL isn't polluted by paper history.
    real_state_path = _ROOT / "data" / "conviction_real_state.json"
    live_state: dict = {}
    if real_state_path.exists():
        try:
            live_state = json.loads(real_state_path.read_text(encoding="utf-8"))
        except Exception:
            live_state = {}

    if not state and not live_state:
        return {"error": "Conviction bot state not found — bot may not be running yet."}

    # Summarize ARB state (prioritized over conviction)
    arb_open = state.get("arb_positions", {})
    arb_closed = state.get("closed_arb", [])
    arb_pnl_total = float(state.get("arb_pnl", 0))
    # Mode / balance come from live state first (live bot writes there), then paper as fallback.
    mode = live_state.get("mode") or state.get("mode", "VIRTUAL")
    live_start_time = live_state.get("live_start_time") or state.get("live_start_time")
    real_clob_balance = live_state.get("real_clob_balance") or state.get("real_clob_balance", 0)

    # Separate live and paper arb trades
    arb_paper_closed = [p for p in arb_closed if not p.get("is_live", False)]
    arb_live_closed = [p for p in arb_closed if p.get("is_live", False)]

    # Compute stats on paper arb trades
    arb_paper_wins = sum(1 for p in arb_paper_closed if float(p.get("pnl", 0)) > 0)
    arb_paper_wr = round(100 * arb_paper_wins / len(arb_paper_closed), 1) if arb_paper_closed else None
    arb_paper_pnl = sum(float(p.get("pnl", 0)) for p in arb_paper_closed)

    # Compute stats on live arb trades
    arb_live_wins = sum(1 for p in arb_live_closed if float(p.get("pnl", 0)) > 0)
    arb_live_wr = round(100 * arb_live_wins / len(arb_live_closed), 1) if arb_live_closed else None
    arb_live_pnl = sum(float(p.get("pnl", 0)) for p in arb_live_closed)

    # Separate open arb positions by mode
    arb_open_paper = [p for p in arb_open.values() if not p.get("is_live", False)]
    arb_open_live = [p for p in arb_open.values() if p.get("is_live", False)]

    # Read conviction log to get recent trades (now includes ARB type and CONVICTION type)
    log_entries = []
    log_path = _ROOT / _CONVICTION_LOG_PATH
    if log_path.exists():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    log_entries.append(json.loads(line))
                except Exception:
                    pass
        except Exception:
            pass

    # Separate arb and conviction entries
    arb_entries = [e for e in log_entries if e.get("type") == "ARB"]
    conviction_entries = [e for e in log_entries if e.get("type") != "ARB" or "type" not in e]
    recent_arb = sorted(arb_entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:50]
    recent_conviction = sorted(conviction_entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:50]

    # CONVICTION SIGNAL STATS
    # Paper: from data/conviction_state.json (the legacy mixed file — paper trades only here)
    conviction_open_paper_src = state.get("positions", {})
    conviction_closed_paper_src = state.get("closed_positions", [])
    conviction_pnl_total = float(state.get("pnl_total", 0))

    # Live: from data/conviction_real_state.json (separate file written by live bot)
    conviction_open_live_src = live_state.get("positions", {})
    conviction_closed_live_src = live_state.get("closed_positions", [])

    # Filter paper file to genuinely paper-only (some old rows may have is_live=True remnants)
    conviction_paper_closed = [p for p in conviction_closed_paper_src if not p.get("is_live", False)]
    conviction_paper_open   = [p for p in conviction_open_paper_src.values() if not p.get("is_live", False)]
    conviction_live_closed  = list(conviction_closed_live_src)
    conviction_live_open    = list(conviction_open_live_src.values())

    # Conviction stats
    conv_paper_wins = sum(1 for p in conviction_paper_closed if float(p.get("pnl", 0)) > 0)
    conv_paper_wr = round(100 * conv_paper_wins / len(conviction_paper_closed), 1) if conviction_paper_closed else None
    conv_paper_pnl = sum(float(p.get("pnl", 0)) for p in conviction_paper_closed)

    conv_live_wins = sum(1 for p in conviction_live_closed if float(p.get("pnl", 0)) > 0)
    conv_live_wr = round(100 * conv_live_wins / len(conviction_live_closed), 1) if conviction_live_closed else None
    conv_live_pnl = sum(float(p.get("pnl", 0)) for p in conviction_live_closed)

    # ── Detailed live trade record: open + closed, newest first ──────────────
    def _trade_row(p, status):
        # Filled USDC: prefer the recorded field, else estimate as bet × entry
        bet   = float(p.get("bet", 0) or 0)
        entry = float(p.get("entry_price", 0) or 0)
        filled = p.get("filled_usdc")
        filled = float(filled) if filled is not None else bet * entry
        return {
            "status":      status,
            "symbol":      p.get("symbol", "?"),
            "timeframe":   p.get("timeframe", "?"),
            "direction":   p.get("direction", "?"),
            "conviction":  float(p.get("conviction", 0) or 0),
            "entry_price": entry,
            "tokens":      bet,
            "filled_usdc": filled,
            "inverted":    bool(p.get("inverted", False)),
            "fill_type":   p.get("fill_type", ""),
            "order_id":    p.get("order_id", ""),
            "pnl":         float(p.get("pnl", 0) or 0),
            "opened_at":   float(p.get("opened_at", 0) or 0),
            "closed_at":   float(p.get("closed_at", 0) or 0),
            "resolved_direction": p.get("resolved_direction", ""),
        }

    live_record = []
    for p in conviction_live_open:
        live_record.append(_trade_row(p, "OPEN"))
    for p in conviction_live_closed:
        won = float(p.get("pnl", 0)) > 0
        live_record.append(_trade_row(p, "WIN" if won else "LOSS"))
    # Sort newest first (closed_at if present else opened_at)
    live_record.sort(key=lambda r: r["closed_at"] or r["opened_at"], reverse=True)

    # Per-symbol breakdown of live conviction trades
    live_by_symbol = {}
    for r in live_record:
        if r["status"] == "OPEN":
            continue
        key = f"{r['symbol']} {r['timeframe']}"
        b = live_by_symbol.setdefault(key, {"trades": 0, "wins": 0, "pnl": 0.0})
        b["trades"] += 1
        if r["status"] == "WIN":
            b["wins"] += 1
        b["pnl"] += r["pnl"]

    # Per-symbol breakdown of paper conviction trades (all settled, lifetime).
    # Mirrors the live table so paper performance is broken out by combo
    # instead of just a recent-20 sample.
    paper_by_symbol = {}
    for p in conviction_paper_closed:
        sym = p.get("symbol", "?")
        tf  = p.get("timeframe", "?")
        key = f"{sym} {tf}"
        b = paper_by_symbol.setdefault(key, {"trades": 0, "wins": 0, "pnl": 0.0})
        b["trades"] += 1
        pnl = float(p.get("pnl", 0) or 0)
        if pnl > 0:
            b["wins"] += 1
        b["pnl"] += pnl

    # ── Detailed paper trade record: open + closed, newest first ──────────────
    # Mirror of live_record but for paper trades, so dashboard shows paper results
    # in the same format as live (WIN/LOSS/OPEN status, details, etc.)
    paper_record = []
    for p in conviction_paper_open:
        paper_record.append(_trade_row(p, "OPEN"))
    for p in conviction_paper_closed:
        won = float(p.get("pnl", 0)) > 0
        paper_record.append(_trade_row(p, "WIN" if won else "LOSS"))
    # Sort newest first (closed_at if present else opened_at)
    paper_record.sort(key=lambda r: r["closed_at"] or r["opened_at"], reverse=True)

    # Total USDC committed in open live positions (real wallet exposure)
    live_open_usdc = sum(r["filled_usdc"] for r in live_record if r["status"] == "OPEN")

    # Group arb trades by symbol and timeframe — SEPARATE by mode
    arb_paper_by_symbol_tf = {}
    arb_live_by_symbol_tf = {}

    for pos in arb_paper_closed:
        symbol = pos.get("symbol", "?")
        tf = pos.get("timeframe", "?")
        key = f"{symbol} {tf}"
        if key not in arb_paper_by_symbol_tf:
            arb_paper_by_symbol_tf[key] = {"trades": 0, "pnl": 0.0, "wins": 0}
        arb_paper_by_symbol_tf[key]["trades"] += 1
        pnl = float(pos.get("pnl", 0))
        arb_paper_by_symbol_tf[key]["pnl"] += pnl
        if pnl > 0:
            arb_paper_by_symbol_tf[key]["wins"] += 1

    for pos in arb_live_closed:
        symbol = pos.get("symbol", "?")
        tf = pos.get("timeframe", "?")
        key = f"{symbol} {tf}"
        if key not in arb_live_by_symbol_tf:
            arb_live_by_symbol_tf[key] = {"trades": 0, "pnl": 0.0, "wins": 0}
        arb_live_by_symbol_tf[key]["trades"] += 1
        pnl = float(pos.get("pnl", 0))
        arb_live_by_symbol_tf[key]["pnl"] += pnl
        if pnl > 0:
            arb_live_by_symbol_tf[key]["wins"] += 1

    return {
        "mode": mode,
        "live_start_time": live_start_time,
        "real_clob_balance": round(float(real_clob_balance or 0), 2),
        "paper_load_error": paper_load_err,

        # Arbitrage stats
        "arb_live_open_count": len(arb_open_live),
        "arb_live_closed_count": len(arb_live_closed),
        "arb_live_hits": arb_live_wins,
        "arb_live_hit_rate": arb_live_wr,
        "arb_live_pnl": round(arb_live_pnl, 2),
        "arb_live_open_positions": list(arb_open_live)[:20],
        "arb_live_by_symbol_tf": arb_live_by_symbol_tf,

        "arb_paper_open_count": len(arb_open_paper),
        "arb_paper_closed_count": len(arb_paper_closed),
        "arb_paper_hits": arb_paper_wins,
        "arb_paper_hit_rate": arb_paper_wr,
        "arb_paper_pnl": round(arb_paper_pnl, 2),
        "arb_paper_open_positions": list(arb_open_paper)[:20],
        "arb_paper_by_symbol_tf": arb_paper_by_symbol_tf,
        "arb_total_pnl": round(arb_pnl_total, 2),
        "arb_recent_trades": recent_arb[:20],

        # Conviction signal stats
        "conv_live_open_count": len(conviction_live_open),
        "conv_live_closed_count": len(conviction_live_closed),
        "conv_live_hits": conv_live_wins,
        "conv_live_hit_rate": conv_live_wr,
        "conv_live_pnl": round(conv_live_pnl, 2),
        "conv_live_open_positions": conviction_live_open[:20],
        "conv_live_open_usdc": round(live_open_usdc, 2),
        "conv_live_trade_record": live_record[:50],   # newest first (open + settled)
        "conv_live_by_symbol_tf": live_by_symbol,

        "conv_paper_open_count": len(conviction_paper_open),
        "conv_paper_closed_count": len(conviction_paper_closed),
        "conv_paper_hits": conv_paper_wins,
        "conv_paper_hit_rate": conv_paper_wr,
        "conv_paper_pnl": round(conv_paper_pnl, 2),
        "conv_paper_trade_record": paper_record[:50],   # newest first (open + settled)
        "conv_paper_by_symbol_tf": paper_by_symbol,
        "conv_total_pnl": round(conviction_pnl_total, 2),
        "conv_recent_signals": recent_conviction[:20],

        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_strike_state() -> dict:
    """Load crypto-strike lottery scanner state from disk."""
    path = _ROOT / _STRIKE_STATE_PATH
    if not path.exists():
        return {"error": "Strike state not found — strike scanner may not have run yet."}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}

    tickets = raw.get("tickets", []) or []
    daily   = raw.get("daily_spend", {}) or {}
    today   = datetime.now(timezone.utc).date().isoformat()
    spent_today = float(daily.get(today, 0.0))
    spent_total = float(sum(daily.values()))

    # Read config for cap + mode
    daily_cap     = 10.0
    ticket_size   = 1.0
    cfg_virtual   = False
    cfg_path      = _ROOT / _STRIKE_CFG_PATH
    if cfg_path.exists():
        try:
            cfg = cfg_path.read_text(encoding="utf-8")
            m = re.search(r"^\s*daily_spend_cap\s*:\s*([\d.]+)", cfg, re.MULTILINE)
            if m: daily_cap = float(m.group(1))
            m = re.search(r"^\s*ticket_size_usdc\s*:\s*([\d.]+)", cfg, re.MULTILINE)
            if m: ticket_size = float(m.group(1))
            m = re.search(r"^\s*virtual_mode\s*:\s*(true|false)", cfg, re.MULTILINE | re.IGNORECASE)
            if m: cfg_virtual = (m.group(1).lower() == "true")
        except Exception:
            pass
    env_virtual = os.getenv("STRIKE_VIRTUAL_MODE", "").lower() in ("true", "1", "yes")
    mode = "VIRTUAL" if (cfg_virtual or env_virtual) else "LIVE"

    open_tix    = [t for t in tickets if not t.get("settled")]
    settled_tix = [t for t in tickets if t.get("settled")]

    # Realized PnL from settled tickets (full-payout model)
    realized_pnl = round(sum(float(t.get("pnl_usd", 0.0)) for t in settled_tix), 4)
    wins         = [t for t in settled_tix if float(t.get("pnl_usd", 0.0)) > 0]
    win_rate     = round(len(wins) / len(settled_tix) * 100, 1) if settled_tix else None

    # Open potential = sum of (ticket_usd / fill_price - ticket_usd) — payout if all win
    def _max_payout(t):
        try:
            return float(t["ticket_usd"]) / max(float(t["fill_price"]), 0.0001)
        except Exception:
            return 0.0
    open_max_payout = round(sum(_max_payout(t) for t in open_tix), 2)
    open_cost       = round(sum(float(t.get("ticket_usd", 0.0)) for t in open_tix), 2)
    open_max_pnl    = round(open_max_payout - open_cost, 2)

    # Build display rows
    def _ticket_row(t):
        cost = float(t.get("ticket_usd", 0.0))
        fp   = float(t.get("fill_price", 0.0))
        max_payout = cost / max(fp, 0.0001)
        return {
            "asset":      t.get("asset", "?"),
            "strike":     float(t.get("strike", 0.0)),
            "direction":  t.get("direction", "above"),
            "fill_price": round(fp, 4),
            "ticket_usd": round(cost, 4),
            "max_payout": round(max_payout, 2),
            "max_x":      round(max_payout / max(cost, 0.0001), 1),
            "source":     t.get("source", "?"),
            "edge_model": round(float(t.get("edge_model", 0.0)), 3),
            "implied":    round(float(t.get("implied", 0.0)), 3),
            "current_at_entry": round(float(t.get("current_at_entry", 0.0)), 2),
            "end_iso":    t.get("end_iso", ""),
            "created_iso": t.get("created_iso", ""),
            "slug":       t.get("slug", ""),
            "pnl_usd":    round(float(t.get("pnl_usd", 0.0)), 4) if t.get("settled") else None,
            "yes_outcome": t.get("yes_outcome"),
            "settled":    bool(t.get("settled")),
        }

    open_rows    = sorted([_ticket_row(t) for t in open_tix],
                          key=lambda r: r.get("end_iso") or "")
    # latest 25 settled
    settled_rows = sorted([_ticket_row(t) for t in settled_tix],
                          key=lambda r: r.get("created_iso") or "",
                          reverse=True)[:25]

    # Recent log lines (FIRE / ORDER OK / SETTLE / WARNING)
    recent_log: list[str] = []
    log_path = _ROOT / _STRIKE_LOG_PATH
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            keywords = ("FIRE", "ORDER OK", "SETTLE", "WARNING", "Daily cap",
                        "Strike scanner starting", "Discovered")
            evt = [l for l in lines if any(k in l for k in keywords)]
            recent_log = evt[-25:]
        except Exception:
            pass

    return {
        "mode":             mode,
        "ticket_size":      ticket_size,
        "daily_cap":        daily_cap,
        "spent_today":      round(spent_today, 4),
        "spent_total":      round(spent_total, 4),
        "open_count":       len(open_tix),
        "settled_count":    len(settled_tix),
        "wins":             len(wins),
        "win_rate":         win_rate,
        "realized_pnl":     realized_pnl,
        "open_cost":        open_cost,
        "open_max_payout":  open_max_payout,
        "open_max_pnl":     open_max_pnl,
        "open_tickets":     open_rows,
        "settled_tickets":  settled_rows,
        "recent_log":       recent_log,
    }


# ── On-chain wallet state (60s cache) ────────────────────────────────────────
_wallet_cache: dict | None = None
_wallet_cache_ts: float    = 0.0
_WALLET_TTL_SEC            = 60


def _load_wallet_state() -> dict:
    """Fetch USDC.e cash + Polymarket CTF position breakdown from on-chain.

    Cached 60s. Categories:
      - cash:           liquid USDC.e in wallet
      - won_pending:    won markets, redeemable=False (UMA finalising)
      - won_redeemable: won markets, redeemable=True  (AR will sweep next cycle)
      - in_play:        live positions, 0.05 < curPrice < 0.97
      - lost:           curPrice <= 0.05 (gone)

    Reports both face value (size) and current market value where applicable.
    """
    global _wallet_cache, _wallet_cache_ts
    if _wallet_cache and (time.time() - _wallet_cache_ts) < _WALLET_TTL_SEC:
        return _wallet_cache

    try:
        import requests
        from web3 import Web3
    except ImportError:
        return {"error": "web3/requests not installed"}

    key = os.getenv("KEY")
    if not key:
        return {"error": "KEY env var not set"}

    try:
        w3 = Web3(Web3.HTTPProvider("https://rpc-mainnet.matic.quiknode.pro"))
        addr = w3.eth.account.from_key(key).address
        usdc_e = w3.eth.contract(
            w3.to_checksum_address("0x2791bca1f2de4661ed88a30c99a7a9449aa84174"),
            abi=[{"inputs":[{"name":"a","type":"address"}],"name":"balanceOf",
                  "outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}],
        ).functions.balanceOf(addr).call() / 1e6
        pol = float(w3.from_wei(w3.eth.get_balance(addr), "ether"))
    except Exception as exc:
        return {"error": f"web3 call failed: {exc}"}

    try:
        r = requests.get("https://data-api.polymarket.com/positions",
                         params={"user": addr, "limit": 200}, timeout=12)
        positions = r.json() or []
    except Exception as exc:
        return {"error": f"data-api fetch failed: {exc}"}

    won_redeemable: list[dict] = []
    won_pending:    list[dict] = []
    in_play:        list[dict] = []
    lost:           list[dict] = []
    for p in positions:
        try:
            cp = float(p.get("curPrice") or 0)
        except Exception:
            cp = 0.0
        if p.get("redeemable") and cp >= 0.97:
            won_redeemable.append(p)
        elif (not p.get("redeemable")) and cp >= 0.97:
            won_pending.append(p)
        elif cp <= 0.05:
            lost.append(p)
        else:
            in_play.append(p)

    def sumf(items, key):
        return round(sum(float(p.get(key) or 0) for p in items), 2)

    def top_items(items, n=10):
        ranked = sorted(items, key=lambda p: -float(p.get("size") or 0))[:n]
        return [{
            "title":         (p.get("title") or "")[:80],
            "market_id":     p.get("conditionId") or p.get("market") or "",
            "size":          round(float(p.get("size") or 0), 2),
            "current_value": round(float(p.get("currentValue") or 0), 2),
            "cur_price":     round(float(p.get("curPrice") or 0), 4),
            "outcome":       p.get("outcome"),
            "outcome_idx":   p.get("outcomeIndex"),
            "redeemable":    bool(p.get("redeemable")),
            "end_date":      p.get("endDate") or p.get("end_date") or "",
        } for p in ranked]

    cash               = round(usdc_e, 2)
    won_redeem_val     = sumf(won_redeemable, "size")          # face value (size = cash equivalent at $1 per token)
    won_pending_val    = sumf(won_pending, "size")
    in_play_face       = sumf(in_play, "size")
    in_play_curval     = sumf(in_play, "currentValue")
    lost_face          = sumf(lost, "size")
    cumulative_invested = sumf(positions, "size")

    total_recoverable  = round(cash + won_redeem_val + won_pending_val + in_play_curval, 2)

    out = {
        "wallet_addr":      addr,
        "cash_usdc":        cash,
        "pol_balance":      round(pol, 4),
        "won_redeemable":   won_redeem_val,
        "won_pending":      won_pending_val,
        "in_play_face":     in_play_face,
        "in_play_curval":   in_play_curval,
        "lost_face":        lost_face,
        "cumulative_invested": cumulative_invested,
        "total_recoverable": total_recoverable,
        "counts": {
            "won_redeemable": len(won_redeemable),
            "won_pending":    len(won_pending),
            "in_play":        len(in_play),
            "lost":           len(lost),
        },
        "top_in_play":      top_items(in_play, 50),
        "top_won_pending":  top_items(won_pending, 50),
        "fetched_at":       datetime.now(timezone.utc).isoformat(),
    }
    _wallet_cache = out
    _wallet_cache_ts = time.time()
    return out


def _build_api_response(raw: dict) -> dict:
    """Compute derived metrics from raw state dict."""
    if "error" in raw:
        return raw

    positions   = raw.get("positions", [])
    closed      = raw.get("closed_positions", [])
    pnl_history = raw.get("pnl_history", [])

    available         = float(raw.get("available_usdc", 0))
    initial_budget    = float(raw.get("initial_budget", 1000))
    deployed          = sum(float(p.get("size_usdc", 0)) for p in positions)
    # real_clob_balance: actual on-chain USDC.e, synced at startup (C1) and every 10 min (AR).
    # Use this as the ground-truth wallet figure — available_usdc drifts between syncs.
    real_clob_balance = float(raw.get("real_clob_balance", available))
    # pending_ctf_usdc: value of won positions whose CTF tokens have not been redeemed yet.
    # Live source: on-chain wallet (won_redeemable + won_pending), 60s cache. Fallback to
    # raw state's pending_ctf_usdc only if web3 is unreachable — that field only refreshes
    # on the 12h AR cycle and goes stale otherwise (over-counts redemptions already done).
    _wallet = _load_wallet_state()
    if "error" not in _wallet:
        unredeemed_ctf = round(
            float(_wallet.get("won_redeemable", 0.0))
            + float(_wallet.get("won_pending", 0.0)),
            2,
        )
    else:
        unredeemed_ctf = round(float(raw.get("pending_ctf_usdc", 0.0)), 2)
    # total: everything we own — liquid CLOB + won-but-not-yet-redeemed CTF tokens + open bets.
    total             = round(real_clob_balance + unredeemed_ctf + deployed, 2)

    # ── Category breakdown ────────────────────────────────────────────────────
    cat_deployed: dict[str, float] = {}
    cat_count:    dict[str, int]   = {}
    for p in positions:
        cat = p.get("category", "other")
        cat_deployed[cat] = cat_deployed.get(cat, 0.0) + float(p.get("size_usdc", 0))
        cat_count[cat]    = cat_count.get(cat, 0) + 1

    # ── PnL ───────────────────────────────────────────────────────────────────
    # pnl_history is not updated by the crypto loop (entries stop 2026-03-18).
    # Compute cumulative and daily PnL directly from closed_positions instead.
    settled  = [p for p in closed if p.get("realized_pnl") is not None]
    cumulative_pnl = sum(float(p.get("realized_pnl", 0)) for p in settled)

    # Polymarket taker fee (2%) is deducted from available_usdc at bet placement but NOT
    # stored per-position, so cumulative_pnl is gross. Estimate total fees from bet sizes.
    total_fees_paid = round(sum(float(p.get("size_usdc", 0)) * 0.02 for p in settled), 2)
    net_pnl = round(cumulative_pnl - total_fees_paid, 2)

    today = datetime.now(timezone.utc).date().isoformat()
    daily_pnl = sum(
        float(p.get("realized_pnl", 0))
        for p in settled
        if (p.get("fill_time") or "")[:10] == today
    )
    daily_fees = round(sum(
        float(p.get("size_usdc", 0)) * 0.02
        for p in settled
        if (p.get("fill_time") or "")[:10] == today
    ), 2)
    daily_net_pnl = round(daily_pnl - daily_fees, 2)

    # implied_deposited: total capital ever put into the wallet, estimated from current balance
    # plus cumulative trading losses. Accounts for deposits made after launch that the bot
    # doesn't track. If user deposited more, this increases automatically.
    implied_deposited = round(max(initial_budget, total - net_pnl), 2)

    # ROI: net pnl (after fees) vs total deposited.
    # deploy_pct: open bets relative to real CLOB balance.
    roi_pct    = (net_pnl / implied_deposited * 100) if implied_deposited > 0 else 0.0

    # Real ROI: based on wallet balance change (real_pnl_all_time), not truncated trade history.
    # This is accurate even when closed_positions was lost due to state corruption.
    # Uses live unredeemed_ctf (on-chain) instead of stale raw.pending_ctf_usdc.
    _real_pnl_all_time = round(
        float(raw.get("real_clob_balance", real_clob_balance))
        + unredeemed_ctf
        - float(raw.get("initial_real_clob_balance", 0.0))
        - float(raw.get("total_detected_deposits", 0.0)),
        2,
    )
    _total_deposited = float(raw.get("initial_real_clob_balance", 0.0)) + float(raw.get("total_detected_deposits", 0.0))
    real_roi_pct = round(_real_pnl_all_time / _total_deposited * 100, 1) if _total_deposited > 0 else 0.0
    deploy_pct = (deployed / real_clob_balance * 100) if real_clob_balance > 0 else 0.0

    # ── Hit rate from settled positions ──────────────────────────────────────
    wins     = [p for p in settled if float(p.get("realized_pnl", 0)) > 0]
    hit_rate = round(len(wins) / len(settled) * 100, 1) if settled else None

    # ── Real-time PnL chart — one point per settled trade ────────────────────
    sorted_closed = sorted(settled, key=lambda p: p.get("fill_time", ""))
    rt_chart = []
    running = 0.0
    for p in sorted_closed:
        pnl = float(p.get("realized_pnl", 0))
        running += pnl
        ft = p.get("fill_time", "")
        try:
            dt = datetime.fromisoformat(ft.replace("Z", "+00:00"))
            label = dt.strftime("%m/%d %H:%M")
        except Exception:
            label = ft[:16]
        rt_chart.append({
            "label":     label,
            "title":     (p.get("title", "") or "")[:35],
            "direction": p.get("direction", ""),
            "category":  p.get("category", "other"),
            "trade_pnl": round(pnl, 2),
            "cumulative": round(running, 2),
        })

    # ── Position age (days since fill) ───────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    def _age_days(fill_time: str) -> float:
        try:
            ft = datetime.fromisoformat(fill_time.replace("Z", "+00:00"))
            return round((datetime.now(timezone.utc) - ft).total_seconds() / 86400, 1)
        except Exception:
            return 0.0

    open_positions = sorted(positions, key=lambda p: -float(p.get("size_usdc", 0)))

    return {
        "mode":            "VIRTUAL" if _VIRTUAL_MODE else "REAL",
        "initial_budget":  initial_budget,
        "available":          round(available, 2),
        "real_clob_balance":  round(real_clob_balance, 2),
        "unredeemed_ctf":     unredeemed_ctf,
        "deployed":           round(deployed, 2),
        "total":              round(total, 2),
        "deploy_pct":         round(deploy_pct, 1),
        "cumulative_pnl":  round(cumulative_pnl, 2),
        "total_fees_paid": total_fees_paid,
        "net_pnl":         net_pnl,
        "realized_pnl":    round(cumulative_pnl, 2),
        "daily_pnl":       round(daily_pnl, 2),
        "daily_net_pnl":   daily_net_pnl,
        "implied_deposited": implied_deposited,
        "roi_pct":         round(roi_pct, 2),
        "real_roi_pct":    real_roi_pct,
        "hit_rate":        hit_rate,
        "position_count":  len(positions),
        "closed_count":    len(settled),
        "cat_deployed":    {k: round(v, 2) for k, v in cat_deployed.items()},
        "cat_count":       cat_count,
        "pnl_chart":       rt_chart,
        "positions":       [
            {**p, "age_days": _age_days(p.get("fill_time", ""))}
            for p in open_positions
        ],
        "closed_positions": sorted(
            settled,
            key=lambda p: p.get("fill_time", ""),
            reverse=True
        )[:20],
        "last_updated":    raw.get("last_updated", ""),
        "start_date":      raw.get("start_date", ""),
        "server_time":     now_iso,
        # ── Real PnL tracking ────────────────────────────────────────────────
        "real_pnl_all_time":       _real_pnl_all_time,
        "initial_real_clob":       float(raw.get("initial_real_clob_balance", 0.0)),
        "total_detected_deposits": float(raw.get("total_detected_deposits", 0.0)),
    }


# ── HTML template ─────────────────────────────────────────────────────────────
# Note: uses vanilla JS — no build step, no external deps beyond Chart.js CDN

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Polymarket Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script>
    window.onerror = function(msg, src, line, col, err) {
      var el = document.getElementById('global-js-error');
      if (el) {
        el.style.display = 'block';
        el.textContent = 'JS Error: ' + msg + ' (line ' + line + (src ? ', ' + src.split('/').pop() : '') + ')';
      }
      return false;
    };
    window.addEventListener('unhandledrejection', function(e) {
      var el = document.getElementById('global-js-error');
      if (el) {
        el.style.display = 'block';
        el.textContent = 'Unhandled Promise: ' + (e.reason && e.reason.message ? e.reason.message : String(e.reason));
      }
    });
  </script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      #0d1117;
      --surface: #161b22;
      --border:  #30363d;
      --text:    #e6edf3;
      --muted:   #8b949e;
      --green:   #3fb950;
      --red:     #f85149;
      --blue:    #58a6ff;
      --amber:   #d29922;
      --purple:  #a371f7;
      --pink:    #f778ba;
      --teal:    #39d353;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      min-height: 100vh;
    }

    /* ── Header ── */
    .header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      gap: 12px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .header h1 { font-size: 18px; font-weight: 600; flex: 1; }
    .badge {
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
    }
    .badge-virtual { background: #1c3055; color: var(--blue); border: 1px solid var(--blue); }
    .badge-real    { background: #1c3020; color: var(--green); border: 1px solid var(--green); }
    .badge-error   { background: #3d1c1c; color: var(--red); border: 1px solid var(--red); }
    .refresh-info  { font-size: 12px; color: var(--muted); }

    /* ── Layout ── */
    .main { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }

    /* ── Card groups ── */
    .card-group-label {
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 8px;
    }
    .card-group { margin-bottom: 20px; }

    /* ── Metric cards ── */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }
    /* Operational cards (wallet balance) are visually quieter */
    .card-dim {
      background: #0f1318;
      border: 1px solid #252b33;
    }
    .card-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
    .card-value { font-size: 22px; font-weight: 700; }
    .card-sub   { font-size: 11px; color: var(--muted); margin-top: 3px; }
    .card-tag   { font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 8px; margin-left: 6px; vertical-align: middle; }
    .tag-liquid { background: rgba(88,166,255,0.15); color: var(--blue); }
    .tag-locked { background: rgba(210,153,34,0.15); color: var(--amber); }
    .pos  { color: var(--green); }
    .neg  { color: var(--red); }
    .neu  { color: var(--text); }
    .dim  { color: var(--muted); }

    /* ── Chart grid ── */
    .charts {
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    @media (max-width: 700px) {
      .charts { grid-template-columns: 1fr; }
    }
    .chart-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .chart-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 12px;
    }

    /* ── PnL chart (full width) ── */
    .pnl-chart-wrap {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .hidden { display: none; }

    /* ── Tables ── */
    .section-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin: 0 0 10px;
    }
    .table-wrap {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 20px;
    }
    table { width: 100%; border-collapse: collapse; }
    thead th {
      background: #1c2330;
      padding: 8px 12px;
      text-align: left;
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      white-space: nowrap;
    }
    tbody tr { border-top: 1px solid var(--border); }
    tbody tr:hover { background: rgba(255,255,255,0.03); }
    tbody td { padding: 8px 12px; font-size: 13px; }
    .dir-yes { color: var(--green); font-weight: 600; }
    .dir-no  { color: var(--red);   font-weight: 600; }
    .title-cell { max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* category pills */
    .cat {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 10px;
      font-size: 11px;
      font-weight: 600;
    }
    .cat-politics { background: #1c2f55; color: var(--blue); }
    .cat-crypto   { background: #2d2200; color: var(--amber); }
    .cat-sports   { background: #2a1030; color: var(--pink); }
    .cat-other    { background: #1e1a2e; color: var(--purple); }

    /* error state */
    .error-box {
      background: #2d1c1c;
      border: 1px solid var(--red);
      border-radius: 8px;
      padding: 20px;
      color: var(--red);
      margin-bottom: 20px;
    }

    /* ── Tabs ── */
    .tabs {
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 20px;
    }
    .tab-btn {
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      padding: 10px 18px;
      letter-spacing: 0.02em;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active { color: var(--text); border-bottom-color: var(--blue); }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* ── Swarm picks table ── */
    .score-bar {
      display: inline-block;
      height: 8px;
      border-radius: 4px;
      background: var(--blue);
      vertical-align: middle;
      margin-right: 6px;
    }
    .pick-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 10px;
    }
    .pick-card:hover { border-color: var(--blue); }
    .pick-q { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
    .pick-meta { font-size: 12px; color: var(--muted); display: flex; gap: 14px; flex-wrap: wrap; }
    .pick-meta span { white-space: nowrap; }
    .whale-tag {
      display: inline-block;
      padding: 1px 7px;
      border-radius: 8px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(210,153,34,0.20);
      color: var(--amber);
    }
  </style>
</head>
<body>

<div id="global-js-error" style="display:none;background:#f85149;color:#fff;padding:10px 16px;font-family:monospace;font-size:13px;position:fixed;top:0;left:0;right:0;z-index:9999;word-break:break-all"></div>

<div class="header">
  <h1>Polymarket Dashboard</h1>
  <span id="mode-badge" class="badge badge-virtual">VIRTUAL</span>
  <span class="refresh-info" id="refresh-info">Refreshing in <span id="countdown">30</span>s</span>
  <span class="refresh-info" id="updated-at"></span>
</div>

<div class="main">
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('crypto')">Crypto Loop</button>
    <button class="tab-btn" onclick="switchTab('swarm')">AI Swarm</button>
    <button class="tab-btn" onclick="switchTab('arb')">Funding Arb</button>
    <button class="tab-btn" onclick="switchTab('conviction')">Arbitrage (Conviction)</button>
  </div>

  <div id="error-box" class="error-box hidden"></div>

  <div id="tab-crypto" class="tab-pane active">

  <!-- ── True wallet view (on-chain) ──────────────────────────────────── -->
  <div class="card-group">
    <p class="card-group-label">True Wallet (on-chain) — what's actually recoverable right now</p>
    <div class="cards">
      <div class="card">
        <div class="card-label">Total Recoverable</div>
        <div class="card-value pos" id="w-total">—</div>
        <div class="card-sub" id="w-total-sub">cash + pending + in-play current</div>
      </div>
      <div class="card">
        <div class="card-label">Liquid Cash</div>
        <div class="card-value" id="w-cash">—</div>
        <div class="card-sub" id="w-pol">— POL gas</div>
      </div>
      <div class="card">
        <div class="card-label">Won, Pending UMA</div>
        <div class="card-value pos" id="w-pending">—</div>
        <div class="card-sub" id="w-pending-cnt">— markets · resolved, awaiting finalize</div>
      </div>
      <div class="card">
        <div class="card-label">Won, Redeemable</div>
        <div class="card-value pos" id="w-redeem">—</div>
        <div class="card-sub" id="w-redeem-cnt">— markets · AR sweeps next cycle</div>
      </div>
      <div class="card">
        <div class="card-label">In-Play (current)</div>
        <div class="card-value neu" id="w-inplay-cur">—</div>
        <div class="card-sub" id="w-inplay-face">face $— · — markets</div>
      </div>
      <div class="card card-dim">
        <div class="card-label">Lost (cumulative)</div>
        <div class="card-value dim" id="w-lost">—</div>
        <div class="card-sub" id="w-lost-cnt">— markets · written off</div>
      </div>
    </div>
  </div>

  <!-- Logical portfolio group -->
  <div class="card-group">
    <p class="card-group-label">Logical Portfolio — what we own</p>
    <div class="cards">
      <div class="card">
        <div class="card-label">Total Portfolio</div>
        <div class="card-value" id="c-budget">—</div>
        <div class="card-sub" id="c-budget-breakdown">—</div>
      </div>
      <div class="card">
        <div class="card-label">True PnL (cash)</div>
        <div class="card-value" id="c-realpnl">—</div>
        <div class="card-sub" id="c-realpnl-sub">—</div>
      </div>
      <div class="card">
        <div class="card-label">Portfolio ROI</div>
        <div class="card-value" id="c-cpnl">—</div>
        <div class="card-sub" id="c-roi">—</div>
      </div>
      <div class="card">
        <div class="card-label">Today PnL</div>
        <div class="card-value" id="c-dpnl">—</div>
      </div>
      <div class="card">
        <div class="card-label">Hit Rate</div>
        <div class="card-value" id="c-hit">—</div>
        <div class="card-sub" id="c-closed">—</div>
      </div>
      <div class="card">
        <div class="card-label">Open Positions</div>
        <div class="card-value neu" id="c-pos">—</div>
        <div class="card-sub" id="c-pos-usdc">—</div>
      </div>
    </div>
  </div>

  <!-- Real wallet group -->
  <div class="card-group">
    <p class="card-group-label">Real Wallet — liquid USDC (operational context)</p>
    <div class="cards">
      <div class="card card-dim">
        <div class="card-label">Wallet Balance <span class="card-tag tag-liquid">LIQUID</span></div>
        <div class="card-value" id="c-avail">—</div>
        <div class="card-sub" id="c-avail-pct">—</div>
      </div>
      <div class="card card-dim">
        <div class="card-label">In Open Bets <span class="card-tag tag-locked">LOCKED</span></div>
        <div class="card-value dim" id="c-deployed">—</div>
        <div class="card-sub" id="c-deploy-pct">—</div>
      </div>
      <div class="card card-dim">
        <div class="card-label">Est. Total Deposited</div>
        <div class="card-value dim" id="c-total">—</div>
        <div class="card-sub">current balance − net PnL</div>
      </div>
    </div>
  </div>

  <!-- Budget doughnut + Category bar -->
  <div class="charts">
    <div class="chart-box">
      <div class="chart-title">Portfolio Breakdown</div>
      <canvas id="budget-chart" width="260" height="260"></canvas>
    </div>
    <div class="chart-box">
      <div class="chart-title">Deployed by Category (USDC)</div>
      <canvas id="cat-chart" height="180"></canvas>
    </div>
  </div>

  <!-- PnL History -->
  <div class="pnl-chart-wrap" id="pnl-section">
    <div class="chart-title" style="display:flex;justify-content:space-between;align-items:center;">
      <span>Real-time PnL — per trade</span>
      <span id="pnl-trade-count" style="font-size:11px;color:var(--muted)"></span>
    </div>
    <canvas id="pnl-chart" height="120"></canvas>
  </div>

  <!-- Open Positions -->
  <p class="section-title" id="pos-title">Open Positions</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Market</th>
          <th>Cat</th>
          <th>Dir</th>
          <th>Size (USDC)</th>
          <th>Fill Price</th>
          <th>Age (d)</th>
        </tr>
      </thead>
      <tbody id="pos-body"></tbody>
    </table>
  </div>

  <!-- Closed Positions -->
  <div id="closed-section" class="hidden">
    <p class="section-title">Settled Positions (last 20)</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Market</th>
            <th>Dir</th>
            <th>Bet</th>
            <th>Fill</th>
            <th>Outcome</th>
            <th>PnL</th>
            <th>Closed At</th>
          </tr>
        </thead>
        <tbody id="closed-body"></tbody>
      </table>
    </div>
  </div>
  </div><!-- /tab-crypto -->

  <!-- ── Swarm tab ─────────────────────────────────────────────────────────── -->
  <div id="tab-swarm" class="tab-pane">
    <div id="swarm-error" class="error-box hidden"></div>

    <!-- Era split (decision-relevant: NO-only is the live policy) -->
    <div class="card-group">
      <p class="card-group-label">Live Strategy (NO only — post 2026-04-11 YES-block)</p>
      <p style="font-size:11px;color:var(--muted);margin-top:-6px;margin-bottom:10px">
        These cards show only NO-direction trades, which is the only direction the bot
        can take today. YES bets are frozen and listed separately below as legacy.
      </p>
      <div class="cards">
        <div class="card">
          <div class="card-label">Live WR (NO)</div>
          <div class="card-value" id="sr-live-wr">—</div>
          <div class="card-sub" id="sr-live-settled">— settled</div>
        </div>
        <div class="card">
          <div class="card-label">Live PnL (NO)</div>
          <div class="card-value" id="sr-live-pnl">—</div>
          <div class="card-sub">cumulative cash, JSONL only</div>
        </div>
        <div class="card">
          <div class="card-label">Last 14d WR (NO)</div>
          <div class="card-value" id="sr-14d-wr">—</div>
          <div class="card-sub" id="sr-14d-settled">— settled</div>
        </div>
        <div class="card">
          <div class="card-label">Last 14d PnL (NO)</div>
          <div class="card-value" id="sr-14d-pnl">—</div>
          <div class="card-sub">recent trend</div>
        </div>
        <div class="card card-dim">
          <div class="card-label">Legacy YES (frozen)</div>
          <div class="card-value" id="sr-legacy-pnl" style="font-size:16px">—</div>
          <div class="card-sub" id="sr-legacy-settled">— settled · cannot recur</div>
        </div>
      </div>
    </div>

    <!-- Real trade cards (cumulative — includes legacy YES) -->
    <div class="card-group">
      <p class="card-group-label">Cumulative Real Trades ($10–$20/bet, includes legacy YES)</p>
      <div class="cards">
        <div class="card">
          <div class="card-label">Win Rate (NO)</div>
          <div class="card-value" id="sr-wr">—</div>
          <div class="card-sub" id="sr-settled">— settled</div>
        </div>
        <div class="card">
          <div class="card-label">Net PnL (incl. pending)</div>
          <div class="card-value" id="sr-net-pnl">—</div>
          <div class="card-sub" id="sr-net-sub">settled: — + tokens: —</div>
        </div>
        <div class="card card-dim">
          <div class="card-label">Settled Cash PnL</div>
          <div class="card-value" id="sr-pnl" style="font-size:16px">—</div>
          <div class="card-sub" id="sr-dpnl">today: —</div>
        </div>
        <div class="card card-dim">
          <div class="card-label">Pending Tokens</div>
          <div class="card-value" id="sr-pending" style="font-size:16px">—</div>
          <div class="card-sub">won, awaiting redemption</div>
        </div>
        <div class="card">
          <div class="card-label">Open Bets</div>
          <div class="card-value neu" id="sr-open">—</div>
          <div class="card-sub" id="sr-open-sub">on-chain · cur $—</div>
        </div>
        <div class="card">
          <div class="card-label">Last Updated</div>
          <div class="card-value dim" style="font-size:14px" id="s-ts">—</div>
        </div>
      </div>
    </div>

    <!-- Open positions (ON-CHAIN truth, cross-referenced with JSONL metadata) -->
    <div id="swarm-real-open-section">
      <p class="section-title" id="swarm-real-open-title">Open Positions (0)</p>
      <p style="font-size:11px;color:var(--muted);margin-top:-8px;margin-bottom:10px">
        From on-chain CTF holdings. ★ = tracked in JSONL (have metadata); untracked rows
        are positions placed via taker fallback that didn't land in the JSONL.
      </p>
      <div class="table-wrap" style="max-height:340px;overflow-y:auto;">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Market</th><th>Dir</th><th>Cur Px</th>
              <th>Face (sz)</th><th>Cur $</th><th>Cost (est)</th><th>PnL If Now</th>
              <th>Score</th><th>Entry</th>
            </tr>
          </thead>
          <tbody id="swarm-real-open-body"></tbody>
        </table>
      </div>
    </div>

    <!-- Won, pending UMA finalize -->
    <div id="swarm-pending-uma-section" style="margin-top:24px">
      <p class="section-title" id="swarm-pending-uma-title">Won, Pending UMA Finalize (0)</p>
      <p style="font-size:11px;color:var(--muted);margin-top:-8px;margin-bottom:10px">
        Resolved markets where we hold the winning side; AR will redeem once the UMA
        oracle marks them <code>redeemable=true</code>.
      </p>
      <div class="table-wrap" style="max-height:240px;overflow-y:auto;">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Market</th><th>Dir</th><th>Cur Px</th>
              <th>Face (sz)</th><th>Cost (est)</th><th>Win PnL</th><th>Score</th>
            </tr>
          </thead>
          <tbody id="swarm-pending-uma-body"></tbody>
        </table>
      </div>
    </div>

    <!-- Real settled trades (from JSONL) -->
    <div id="sr-closed-section" class="hidden">
      <p class="section-title">Settled Real Trades (JSONL only — incomplete)</p>
      <p style="font-size:11px;color:var(--muted);margin-top:-8px;margin-bottom:10px">
        Only trades captured in <code>swarm_real_trades.jsonl</code>. Some untracked
        wins/losses won't appear here; see Open Positions above for on-chain truth.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Market</th><th>Dir</th><th>Ask</th><th>Bet</th><th>Outcome</th><th>PnL</th><th>Entry Time</th>
            </tr>
          </thead>
          <tbody id="sr-closed-body"></tbody>
        </table>
      </div>
    </div>

    <!-- Paper analysis (secondary) -->
    <div class="card-group" style="margin-top:32px">
      <p class="card-group-label">Paper Analysis (@ $10/trade — all AI-gate picks)</p>
      <div class="cards">
        <div class="card">
          <div class="card-label">Win Rate</div>
          <div class="card-value" id="s-wr">—</div>
          <div class="card-sub" id="s-settled">— settled</div>
        </div>
        <div class="card">
          <div class="card-label">Total PnL</div>
          <div class="card-value" id="s-pnl">—</div>
          <div class="card-sub" id="s-dpnl">today: —</div>
        </div>
        <div class="card">
          <div class="card-label">Open Picks</div>
          <div class="card-value neu" id="s-open">—</div>
        </div>
      </div>
    </div>

    <!-- Current cycle picks -->
    <p class="section-title" style="margin-top:24px">Last Cycle Picks</p>
    <div id="swarm-picks"></div>

    <!-- Settled paper trades -->
    <div id="s-closed-section" class="hidden">
      <p class="section-title">Recent Paper Picks</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Market</th><th>Dir</th><th>Score</th>
              <th>Entry</th><th>Outcome</th><th>PnL ($10)</th>
            </tr>
          </thead>
          <tbody id="s-closed-body"></tbody>
        </table>
      </div>
    </div>
  </div><!-- /tab-swarm -->

  <!-- ── Funding Arb tab ───────────────────────────────────────────────────── -->
  <div id="tab-arb" class="tab-pane">
    <div id="arb-error" class="error-box hidden"></div>

    <div class="card-group">
      <p class="card-group-label">Binance Funding Rate Arb — Multi-Symbol Delta-Neutral (Long Spot + Short Perp)</p>
      <div class="cards">
        <div class="card">
          <div class="card-label">Status</div>
          <div class="card-value neu" id="arb-status">—</div>
          <div class="card-sub" id="arb-mode-sub">—</div>
        </div>
        <div class="card">
          <div class="card-label">Funding Collected</div>
          <div class="card-value" id="arb-funding">—</div>
          <div class="card-sub" id="arb-trades">— cycles</div>
        </div>
        <div class="card">
          <div class="card-label">Realized PnL</div>
          <div class="card-value" id="arb-pnl">—</div>
          <div class="card-sub">spot + funding</div>
        </div>
        <div class="card">
          <div class="card-label">Active Symbol</div>
          <div class="card-value neu" id="arb-active-symbol">—</div>
          <div class="card-sub" id="arb-qty">—</div>
        </div>
        <div class="card">
          <div class="card-label">Entry → Current</div>
          <div class="card-value dim" id="arb-entry">—</div>
          <div class="card-sub" id="arb-age">—</div>
        </div>
        <div class="card">
          <div class="card-label">Pending Funding</div>
          <div class="card-value dim" id="arb-pending">—</div>
          <div class="card-sub" id="arb-errors-sub">0 errors</div>
        </div>
      </div>
    </div>

    <!-- Live rate scan -->
    <p class="section-title">Live Funding Rates (last scan)</p>
    <div class="table-wrap" style="margin-bottom:20px">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Rate / 8h</th>
            <th>APY</th>
            <th>Direction</th>
            <th>vs Threshold (10.95%)</th>
          </tr>
        </thead>
        <tbody id="arb-rates-body"></tbody>
      </table>
    </div>

    <p class="section-title">Recent Log</p>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:12px;font-family:monospace;font-size:12px;max-height:260px;overflow-y:auto;">
      <div id="arb-log" style="color:var(--muted);white-space:pre-wrap;line-height:1.6"></div>
    </div>
  </div><!-- /tab-arb -->

  <!-- ── Arbitrage tab ──────────────────────────────────────────────────── -->
  <div id="tab-conviction" class="tab-pane">
    <div id="conviction-error" class="error-box hidden"></div>

    <!-- LIVE Conviction Trade Record (replaces arb area which barely fires) -->
    <div class="card-group">
      <p class="card-group-label">Conviction — LIVE Trade Record (Real CLOB Orders)</p>
      <div class="cards">
        <div class="card">
          <div class="card-label">Mode</div>
          <div class="card-value" id="conviction-mode">—</div>
          <div class="card-sub" id="conviction-mode-detail">—</div>
        </div>
        <div class="card">
          <div class="card-label">Open Positions</div>
          <div class="card-value neu" id="conviction-live-open">—</div>
          <div class="card-sub" id="conviction-live-open-detail">—</div>
        </div>
        <div class="card">
          <div class="card-label">Closed Trades</div>
          <div class="card-value" id="conviction-live-closed">—</div>
          <div class="card-sub" id="conviction-live-closed-detail">—</div>
        </div>
        <div class="card">
          <div class="card-label">Win Rate</div>
          <div class="card-value" id="conviction-live-wr">—</div>
          <div class="card-sub">of settled live trades</div>
        </div>
        <div class="card">
          <div class="card-label">Live PnL</div>
          <div class="card-value" id="conviction-live-pnl">—</div>
          <div class="card-sub">real wallet movement</div>
        </div>
      </div>
    </div>

    <p class="section-title" id="conviction-recent-title">Live Trade Record (latest 50, open + settled)</p>
    <div class="table-wrap" style="margin-bottom:20px;max-height:500px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Symbol</th>
            <th>TF</th>
            <th>Dir</th>
            <th>Skew</th>
            <th>Entry</th>
            <th>Tokens</th>
            <th>USDC In</th>
            <th>Fill</th>
            <th>PnL</th>
            <th>Order</th>
            <th>Opened</th>
          </tr>
        </thead>
        <tbody id="conviction-body"></tbody>
      </table>
    </div>

    <p class="section-title">Live Performance by Symbol/Timeframe</p>
    <div class="table-wrap" style="margin-bottom:20px;max-height:250px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>Symbol+TF</th>
            <th>Settled</th>
            <th>Wins</th>
            <th>WR</th>
            <th>PnL</th>
          </tr>
        </thead>
        <tbody id="conviction-live-perf-body"></tbody>
      </table>
    </div>

    <!-- ── PAPER SECTION (live now lives in the top trade record) ───────── -->
    <hr style="margin: 40px 0; border: 1px solid #30363d;">

    <!-- CONVICTION Paper Trading Section -->
    <div class="card-group" style="margin-top: 30px;">
      <p class="card-group-label">Conviction Signals — Paper Trading (Historical)</p>
      <div class="cards">
        <div class="card">
          <div class="card-label">Open Positions</div>
          <div class="card-value neu" id="conviction-conv-paper-open">—</div>
          <div class="card-sub" id="conviction-conv-paper-open-detail">—</div>
        </div>
        <div class="card">
          <div class="card-label">Closed Trades</div>
          <div class="card-value" id="conviction-conv-paper-closed">—</div>
          <div class="card-sub" id="conviction-conv-paper-closed-detail">—</div>
        </div>
        <div class="card">
          <div class="card-label">Win Rate</div>
          <div class="card-value" id="conviction-conv-paper-wr">—</div>
          <div class="card-sub">of closed paper signals</div>
        </div>
        <div class="card">
          <div class="card-label">Paper PnL</div>
          <div class="card-value" id="conviction-conv-paper-pnl">—</div>
          <div class="card-sub">simulated fills</div>
        </div>
      </div>
    </div>

    <p class="section-title">Paper Performance by Symbol/Timeframe</p>
    <div class="table-wrap" style="margin-bottom:20px;max-height:250px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>Symbol+TF</th>
            <th>Settled</th>
            <th>Wins</th>
            <th>WR</th>
            <th>PnL</th>
          </tr>
        </thead>
        <tbody id="conviction-paper-perf-body"></tbody>
      </table>
    </div>

    <p class="section-title" id="conviction-paper-recent-title">Paper Trade Record (newest first)</p>
    <div class="table-wrap" style="margin-bottom:20px;max-height:400px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Symbol</th>
            <th>TF</th>
            <th>Dir</th>
            <th>Skew</th>
            <th>Entry</th>
            <th>Tokens</th>
            <th>USDC In</th>
            <th>Fill</th>
            <th>PnL</th>
            <th>Order</th>
            <th>Opened</th>
          </tr>
        </thead>
        <tbody id="conviction-paper-body"></tbody>
      </table>
    </div>

    <p class="section-title" id="conviction-conv-recent-title">Recent Conviction Signals (last 20)</p>
    <div class="table-wrap" style="margin-bottom:20px;max-height:400px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>TF</th>
            <th>Direction</th>
            <th>Conviction</th>
            <th>Entry Price</th>
            <th>Bet Size</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody id="conviction-conv-body"></tbody>
      </table>
    </div>
  </div><!-- /tab-conviction -->

</div>

<script>
  // ── Chart.js instances (kept for incremental updates) ──────────────────────
  let budgetChart = null;
  let catChart    = null;
  let pnlChart    = null;
  let pnlData     = [];  // module-level mirror of d.pnl_chart — keeps tooltip closure fresh

  const CAT_COLORS = {
    politics: { bg: "rgba(88,166,255,0.25)", border: "#58a6ff" },
    crypto:   { bg: "rgba(210,153,34,0.25)", border: "#d29922" },
    sports:   { bg: "rgba(247,120,186,0.25)", border: "#f778ba" },
    other:    { bg: "rgba(163,113,247,0.25)", border: "#a371f7" },
  };
  function catColor(cat, prop) {
    return (CAT_COLORS[cat] || CAT_COLORS.other)[prop];
  }

  // ── Formatting helpers ─────────────────────────────────────────────────────
  function fmt$(v) { return "$" + parseFloat(v).toFixed(2); }
  function fmtPct(v) { return parseFloat(v).toFixed(1) + "%"; }
  function pnlClass(v) { return v > 0 ? "pos" : v < 0 ? "neg" : "dim"; }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render(d) {
    // Error state
    const errorBox = document.getElementById("error-box");
    if (d.error) {
      errorBox.textContent = "Error: " + d.error;
      errorBox.classList.remove("hidden");
      return;
    }
    errorBox.classList.add("hidden");

    // Mode badge
    const badge = document.getElementById("mode-badge");
    badge.textContent = d.mode;
    badge.className = "badge " + (d.mode === "VIRTUAL" ? "badge-virtual" : "badge-real");

    // Last updated
    const updEl = document.getElementById("updated-at");
    if (d.last_updated) {
      const dt = new Date(d.last_updated);
      updEl.textContent = "Updated " + dt.toLocaleTimeString();
    }

    // ── Logical portfolio group ──────────────────────────────────────────────
    // total = real_clob_balance + deployed — on-chain ground truth.
    // unredeemed_ctf > 0 only when wins are settled virtually but not yet redeemed on-chain.
    const portfolioEl = document.getElementById("c-budget");
    portfolioEl.textContent = fmt$(d.total);
    portfolioEl.className = "card-value " + pnlClass(d.net_pnl);

    // Sub-line breakdown: Real CLOB | Unredeemed CTF | In bets
    if (d.unredeemed_ctf > 0.01) {
      document.getElementById("c-budget-breakdown").innerHTML =
        "<span style='color:var(--blue)'>CLOB " + fmt$(d.real_clob_balance) + "</span>" +
        " &nbsp;+&nbsp; " +
        "<span style='color:var(--amber)'>Unredeemed CTF " + fmt$(d.unredeemed_ctf) + "</span>" +
        (d.deployed > 0.01 ? " &nbsp;+&nbsp; <span style='color:var(--muted)'>Bets " + fmt$(d.deployed) + "</span>" : "");
    } else {
      document.getElementById("c-budget-breakdown").innerHTML =
        "<span style='color:var(--blue)'>CLOB " + fmt$(d.real_clob_balance) + "</span>" +
        (d.deployed > 0.01 ? " &nbsp;+&nbsp; <span style='color:var(--muted)'>Bets " + fmt$(d.deployed) + "</span>" : " — fully liquid");
    }

    // True PnL (real cash) — wallet change minus external deposits
    const realPnlEl = document.getElementById("c-realpnl");
    if (d.initial_real_clob && d.initial_real_clob > 0) {
      realPnlEl.textContent = (d.real_pnl_all_time >= 0 ? "+" : "") + fmt$(d.real_pnl_all_time);
      realPnlEl.className = "card-value " + pnlClass(d.real_pnl_all_time);
      document.getElementById("c-realpnl-sub").textContent =
        "start " + fmt$(d.initial_real_clob) +
        " | deposits +" + fmt$(d.total_detected_deposits);
    } else {
      realPnlEl.textContent = "N/A";
      realPnlEl.className = "card-value dim";
    }

    // Portfolio ROI — based on wallet balance change (real_pnl_all_time / total deposited).
    // net_pnl from closed_positions is intentionally NOT shown here: it only reflects trades
    // in the current state file and is permanently negative after the 2026-04-11 state corruption.
    const cpnlEl = document.getElementById("c-cpnl");
    const realRoi = d.real_roi_pct != null ? d.real_roi_pct : 0;
    cpnlEl.textContent = (realRoi >= 0 ? "+" : "") + realRoi.toFixed(1) + "%";
    cpnlEl.className = "card-value " + pnlClass(realRoi);
    const roiEl = document.getElementById("c-roi");
    roiEl.textContent = "trade P&L " + (d.net_pnl >= 0 ? "+" : "") + fmt$(d.net_pnl) +
      " (" + d.closed_count + " trades, partial)";
    roiEl.className = "card-sub " + pnlClass(d.net_pnl);

    const dpnlEl = document.getElementById("c-dpnl");
    dpnlEl.textContent = (d.daily_net_pnl >= 0 ? "+" : "") + fmt$(d.daily_net_pnl) + " net";
    dpnlEl.className = "card-value " + pnlClass(d.daily_net_pnl);

    const hitEl = document.getElementById("c-hit");
    if (d.hit_rate !== null) {
      hitEl.textContent = fmtPct(d.hit_rate);
      hitEl.className = "card-value " + pnlClass(d.hit_rate - 50);
    } else {
      hitEl.textContent = "—";
      hitEl.className = "card-value dim";
    }
    document.getElementById("c-closed").textContent = d.closed_count + " settled";

    // ── 1H Crypto Loop stats (if available) ──────────────────────────────────
    if (d.crypto_1h && !d.crypto_1h.error) {
      const hr1h = document.querySelector(".card-group:last-of-type");
      if (hr1h && !document.getElementById("crypto-1h-stats")) {
        const statsDiv = document.createElement("div");
        statsDiv.id = "crypto-1h-stats";
        statsDiv.className = "card-group";
        statsDiv.style.borderLeft = "3px solid #d29922";
        const label = document.createElement("p");
        label.className = "card-group-label";
        label.textContent = "1H Crypto Loop (" + (d.crypto_1h.mode || "VIRTUAL") + ")";
        statsDiv.appendChild(label);
        const cardsDiv = document.createElement("div");
        cardsDiv.className = "cards";
        cardsDiv.innerHTML = `
          <div class="card">
            <div class="card-label">Closed Trades</div>
            <div class="card-value">${d.crypto_1h.closed_trades || 0}</div>
          </div>
          <div class="card">
            <div class="card-label">Hit Rate</div>
            <div class="card-value" style="color:${d.crypto_1h.hit_rate >= 65 ? 'var(--green)' : 'var(--orange)'}">${d.crypto_1h.hit_rate ? d.crypto_1h.hit_rate.toFixed(1) + "%" : "—"}</div>
          </div>
          <div class="card">
            <div class="card-label">PnL</div>
            <div class="card-value" style="color:${d.crypto_1h.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${(d.crypto_1h.pnl >= 0 ? "+" : "") + fmt$(d.crypto_1h.pnl)}</div>
          </div>
        `;
        statsDiv.appendChild(cardsDiv);
        hr1h.parentNode.insertBefore(statsDiv, hr1h.nextSibling);
      }
    }

    document.getElementById("c-pos").textContent = d.position_count;
    document.getElementById("c-pos-usdc").textContent =
      d.deployed > 0 ? fmt$(d.deployed) + " locked" : "none open";

    // ── Real wallet group ────────────────────────────────────────────────────
    // real_clob_balance = literal on-chain USDC.e (synced at startup + every 10 min AR check).
    document.getElementById("c-avail").textContent = fmt$(d.real_clob_balance);
    document.getElementById("c-avail").className = "card-value neu";
    document.getElementById("c-avail-pct").textContent =
      "synced every ~10 min";

    const depEl = document.getElementById("c-deployed");
    depEl.textContent = fmt$(d.deployed);
    depEl.className = "card-value dim";
    document.getElementById("c-deploy-pct").textContent =
      fmtPct(d.deploy_pct) + " of CLOB balance";

    // Start Capital = estimated total ever deposited = current total - net trading PnL.
    // Updates automatically when new deposits are detected (balance jumps).
    document.getElementById("c-total").textContent = fmt$(d.implied_deposited);

    // ── Budget doughnut ──────────────────────────────────────────────────────
    // Slices: logical balance (not at risk) + one slice per open-bet category
    const cats = Object.keys(d.cat_deployed).sort();
    const budgetLabels  = ["Logical Balance", ...cats.map(c => c.charAt(0).toUpperCase() + c.slice(1))];
    const budgetData    = [d.available, ...cats.map(c => d.cat_deployed[c])];
    const budgetColors  = [
      "rgba(63,185,80,0.35)",
      ...cats.map(c => catColor(c, "bg"))
    ];
    const budgetBorders = [
      "#3fb950",
      ...cats.map(c => catColor(c, "border"))
    ];

    if (budgetChart) {
      budgetChart.data.labels = budgetLabels;
      budgetChart.data.datasets[0].data    = budgetData;
      budgetChart.data.datasets[0].backgroundColor = budgetColors;
      budgetChart.data.datasets[0].borderColor      = budgetBorders;
      budgetChart.update("none");
    } else {
      budgetChart = new Chart(document.getElementById("budget-chart"), {
        type: "doughnut",
        data: {
          labels: budgetLabels,
          datasets: [{
            data: budgetData,
            backgroundColor: budgetColors,
            borderColor:     budgetBorders,
            borderWidth: 2,
            hoverOffset: 6,
          }]
        },
        options: {
          plugins: {
            legend: {
              position: "bottom",
              labels: { color: "#8b949e", font: { size: 11 }, padding: 10, boxWidth: 12 }
            },
            tooltip: {
              callbacks: {
                label: ctx => " " + ctx.label + ": $" + ctx.raw.toFixed(2)
              }
            }
          },
          cutout: "60%",
        }
      });
    }

    // ── Category bar chart ───────────────────────────────────────────────────
    if (cats.length > 0) {
      const catLabels  = cats.map(c => c.charAt(0).toUpperCase() + c.slice(1));
      const catValues  = cats.map(c => d.cat_deployed[c]);
      const catBg      = cats.map(c => catColor(c, "bg"));
      const catBorder  = cats.map(c => catColor(c, "border"));

      if (catChart) {
        catChart.data.labels = catLabels;
        catChart.data.datasets[0].data            = catValues;
        catChart.data.datasets[0].backgroundColor = catBg;
        catChart.data.datasets[0].borderColor     = catBorder;
        catChart.update("none");
      } else {
        catChart = new Chart(document.getElementById("cat-chart"), {
          type: "bar",
          data: {
            labels: catLabels,
            datasets: [{
              label: "Deployed (USDC)",
              data:            catValues,
              backgroundColor: catBg,
              borderColor:     catBorder,
              borderWidth: 2,
              borderRadius: 4,
            }]
          },
          options: {
            indexAxis: "y",
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: { label: ctx => " $" + ctx.raw.toFixed(2) }
              }
            },
            scales: {
              x: {
                grid:  { color: "rgba(255,255,255,0.06)" },
                ticks: { color: "#8b949e", callback: v => "$" + v }
              },
              y: { ticks: { color: "#e6edf3" }, grid: { display: false } }
            }
          }
        });
      }
    }

    // ── Real-time PnL chart — per trade ─────────────────────────────────────
    const pnlSection = document.getElementById("pnl-section");
    if (d.pnl_chart && d.pnl_chart.length > 0) {
      pnlData = d.pnl_chart;  // keep module-level ref fresh so tooltip closure is never stale
      pnlSection.classList.remove("hidden");
      document.getElementById("pnl-trade-count").textContent =
        d.pnl_chart.length + " trades";

      const pnlLabels = d.pnl_chart.map((_, i) => i + 1);
      const pnlVals   = d.pnl_chart.map(e => (e.cumulative != null ? e.cumulative : null));

      // Segment coloring: green when line is above zero, red when below
      const segmentColor = ctx => {
        const y0 = ctx.p0.parsed.y, y1 = ctx.p1.parsed.y;
        if (y0 >= 0 && y1 >= 0) return "#3fb950";
        if (y0 <  0 && y1 <  0) return "#f85149";
        return "#8b949e"; // crossing zero — neutral
      };
      const lastVal = pnlVals[pnlVals.length - 1];
      const fillColor = lastVal >= 0 ? "rgba(63,185,80,0.10)" : "rgba(248,81,73,0.10)";

      if (pnlChart) {
        pnlChart.data.labels = pnlLabels;
        pnlChart.data.datasets[0].data = pnlVals;
        pnlChart.data.datasets[0].backgroundColor = fillColor;
        pnlChart.update("none");
      } else {
        pnlChart = new Chart(document.getElementById("pnl-chart"), {
          type: "line",
          data: {
            labels: pnlLabels,
            datasets: [{
              label: "Cumulative PnL",
              data: pnlVals,
              segment: { borderColor: segmentColor },
              backgroundColor: fillColor,
              fill: true,
              tension: 0.15,
              pointRadius: 0,
              pointHoverRadius: 5,
              pointHoverBackgroundColor: ctx => {
                const e = pnlData[ctx.dataIndex];
                return e && e.trade_pnl > 0 ? "#3fb950" : "#f85149";
              },
              borderWidth: 2,
            }]
          },
          options: {
            animation: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  title: ctx => {
                    const e = pnlData[ctx[0].dataIndex];
                    if (!e) return "";
                    return "#" + (ctx[0].dataIndex + 1) + "  " + (e.label || "");
                  },
                  label: ctx => {
                    const e = pnlData[ctx.dataIndex];
                    if (!e) return "";
                    const sign = e.trade_pnl >= 0 ? "+" : "";
                    return [
                      " " + (e.title || "").slice(0, 40),
                      " Trade PnL: " + sign + "$" + e.trade_pnl.toFixed(2),
                      " Cumulative: $" + (e.cumulative != null ? e.cumulative.toFixed(2) : "—"),
                    ];
                  }
                }
              }
            },
            scales: {
              x: {
                ticks: {
                  color: "#8b949e",
                  maxTicksLimit: 10,
                  callback: (v, i) => {
                    // Show trade number, not date — dates crowd at this volume
                    return "#" + (i + 1);
                  }
                },
                grid: { color: "rgba(255,255,255,0.04)" }
              },
              y: {
                ticks: { color: "#8b949e", callback: v => v != null ? "$" + v.toFixed(0) : "" },
                grid: {
                  color: ctx => ctx.tick.value === 0
                    ? "rgba(255,255,255,0.25)"   // bright zero line
                    : "rgba(255,255,255,0.06)",
                },
                afterDataLimits: axis => {
                  if (axis.max < 0) axis.max = 0;
                  if (axis.min > 0) axis.min = 0;
                },
              }
            }
          }
        });
      }
    } else {
      pnlSection.classList.add("hidden");
    }

    // ── Open positions table ──────────────────────────────────────────────────
    const posTitle = document.getElementById("pos-title");
    posTitle.textContent = "Open Positions (" + d.position_count + ")";

    const tbody = document.getElementById("pos-body");
    tbody.innerHTML = "";
    d.positions.forEach((p, i) => {
      const cat = (p.category || "other").toLowerCase();
      const dir = p.direction === "YES" ? "YES" : "NO";
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + (i + 1) + "</td>" +
        "<td class='title-cell' title='" + escHtml(p.title) + "'>" + escHtml(p.title) + "</td>" +
        "<td><span class='cat cat-" + cat + "'>" + cat + "</span></td>" +
        "<td class='" + (dir === "YES" ? "dir-yes" : "dir-no") + "'>" + dir + "</td>" +
        "<td>$" + parseFloat(p.size_usdc).toFixed(2) + "</td>" +
        "<td>" + parseFloat(p.fill_price).toFixed(4) + "</td>" +
        "<td>" + (p.age_days || "—") + "</td>";
      tbody.appendChild(tr);
    });

    // ── Closed positions table ────────────────────────────────────────────────
    const closedSection = document.getElementById("closed-section");
    if (d.closed_positions && d.closed_positions.length > 0) {
      closedSection.classList.remove("hidden");
      const cb = document.getElementById("closed-body");
      cb.innerHTML = "";
      d.closed_positions.forEach((p, i) => {
        const pnl = parseFloat(p.realized_pnl || 0);
        const fill = parseFloat(p.fill_price || 0);
        const outcome = p.outcome === 1 ? "YES ✓" : p.outcome === 0 ? "NO ✓" : "—";
        const fillNote = fill > 0 ? fill.toFixed(3) : "—";
        let closedAt = "—";
        if (p.settle_time) {
          try {
            const dt = new Date(p.settle_time);
            closedAt = dt.toLocaleString([], {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
          } catch(e) {}
        }
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + (i + 1) + "</td>" +
          "<td class='title-cell' title='" + escHtml(p.title) + "'>" + escHtml(p.title) + "</td>" +
          "<td class='" + (p.direction === "YES" ? "dir-yes" : "dir-no") + "'>" + p.direction + "</td>" +
          "<td>$" + parseFloat(p.size_usdc).toFixed(2) + "</td>" +
          "<td style='color:var(--muted);font-size:12px'>" + fillNote + "</td>" +
          "<td>" + outcome + "</td>" +
          "<td class='" + pnlClass(pnl) + "'>" + (pnl >= 0 ? "+" : "") + "$" + pnl.toFixed(2) + "</td>" +
          "<td style='font-size:11px;color:var(--muted);white-space:nowrap'>" + closedAt + "</td>";
        cb.appendChild(tr);
      });
    } else {
      closedSection.classList.add("hidden");
    }
  }

  function escHtml(str) {
    return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }

  // ── Tab switching ─────────────────────────────────────────────────────────
  let _activeTab = "crypto";
  function switchTab(name) {
    _activeTab = name;
    document.querySelectorAll(".tab-btn").forEach(b =>
      b.classList.toggle("active", b.getAttribute("onclick").includes("'" + name + "'"))
    );
    document.querySelectorAll(".tab-pane").forEach(p =>
      p.classList.toggle("active", p.id === "tab-" + name)
    );
    if (name === "swarm") fetchSwarm();
    else if (name === "arb") fetchArb();
    else if (name === "conviction") fetchConviction();
    else { fetchData(); fetchWallet(); }
  }

  // ── True wallet (on-chain) rendering ──────────────────────────────────────
  function renderWallet(d) {
    if (d.error) {
      document.getElementById("w-total").textContent = "ERR";
      document.getElementById("w-total-sub").textContent = d.error;
      return;
    }
    document.getElementById("w-total").textContent = "$" + d.total_recoverable.toFixed(2);
    document.getElementById("w-total-sub").textContent =
      "cash $" + d.cash_usdc.toFixed(2) +
      " + pending $" + d.won_pending.toFixed(2) +
      " + in-play $" + d.in_play_curval.toFixed(2);

    document.getElementById("w-cash").textContent = "$" + d.cash_usdc.toFixed(2);
    document.getElementById("w-pol").textContent = d.pol_balance.toFixed(4) + " POL gas";

    document.getElementById("w-pending").textContent = "$" + d.won_pending.toFixed(2);
    document.getElementById("w-pending-cnt").textContent =
      d.counts.won_pending + " market" + (d.counts.won_pending === 1 ? "" : "s") +
      " · resolved, awaiting UMA finalize";

    document.getElementById("w-redeem").textContent = "$" + d.won_redeemable.toFixed(2);
    document.getElementById("w-redeem-cnt").textContent =
      d.counts.won_redeemable + " market" + (d.counts.won_redeemable === 1 ? "" : "s") +
      " · AR sweeps next cycle";

    document.getElementById("w-inplay-cur").textContent = "$" + d.in_play_curval.toFixed(2);
    document.getElementById("w-inplay-face").textContent =
      "face $" + d.in_play_face.toFixed(2) + " · " + d.counts.in_play + " markets";

    document.getElementById("w-lost").textContent = "$" + d.lost_face.toFixed(2);
    document.getElementById("w-lost-cnt").textContent =
      d.counts.lost + " markets · written off (cumulative)";
  }

  async function fetchWallet() {
    try {
      const res = await fetch("/api/wallet");
      const data = await res.json();
      renderWallet(data);
    } catch (e) {
      document.getElementById("w-total").textContent = "—";
      document.getElementById("w-total-sub").textContent = "fetch failed: " + e.message;
    }
  }

  // ── Strike rendering ──────────────────────────────────────────────────────
  function renderConviction(d) {
    const errEl = document.getElementById("conviction-error");
    if (d.error) {
      errEl.textContent = d.error;
      errEl.classList.remove("hidden");
      return;
    }
    errEl.classList.add("hidden");

    // Mode card
    const modeEl = document.getElementById("conviction-mode");
    const mode = d.mode || "VIRTUAL";
    modeEl.textContent = mode;
    modeEl.className = "card-value " + (mode === "LIVE" ? "pos" : "neu");
    document.getElementById("conviction-mode-detail").textContent =
      mode === "LIVE" ? "$" + (d.real_clob_balance || 0).toFixed(2) + " on CLOB" : "paper trading";

    // ── LIVE CONVICTION Summary Cards (was arb area) ─────────────────────────
    const liveOpen   = d.conv_live_open_count   || 0;
    const liveClosed = d.conv_live_closed_count || 0;
    const liveHits   = d.conv_live_hits         || 0;
    const liveWr     = d.conv_live_hit_rate;  // may be null when 0 settled
    const livePnl    = d.conv_live_pnl          || 0;
    const liveOpenUsdc = d.conv_live_open_usdc  || 0;

    document.getElementById("conviction-live-open").textContent = String(liveOpen);
    document.getElementById("conviction-live-open-detail").textContent =
      liveOpen > 0 ? ("$" + liveOpenUsdc.toFixed(2) + " committed") : "no open positions";

    document.getElementById("conviction-live-closed").textContent = String(liveClosed);
    const liveWrText = (liveWr !== null && liveWr !== undefined)
      ? liveHits + "W / " + Math.max(0, liveClosed - liveHits) + "L"
      : "0 settled";
    document.getElementById("conviction-live-closed-detail").textContent = liveWrText;

    const liveWrEl = document.getElementById("conviction-live-wr");
    liveWrEl.textContent = (liveWr !== null && liveWr !== undefined) ? liveWr.toFixed(1) + "%" : "—";
    liveWrEl.className = "card-value " + (liveWr && liveWr >= 60 ? "pos" : liveWr ? "neu" : "dim");

    const livePnlEl = document.getElementById("conviction-live-pnl");
    livePnlEl.textContent = (livePnl >= 0 ? "+" : "") + "$" + livePnl.toFixed(2);
    livePnlEl.className = "card-value " + pnlClass(livePnl);

    // ── Live Trade Record table ─────────────────────────────────────────────
    document.getElementById("conviction-recent-title").textContent =
      "Live Trade Record (" + (d.conv_live_trade_record || []).length + ", newest first)";
    const tbody = document.getElementById("conviction-body");
    tbody.innerHTML = "";
    (d.conv_live_trade_record || []).forEach(t => {
      const tr = document.createElement("tr");
      const tsRaw = (t.opened_at || 0);
      const tsShort = tsRaw > 0 ? new Date(tsRaw * 1000).toISOString().replace("T", " ").substring(5, 19) : "—";
      const statusCls = t.status === "WIN" ? "pos" : t.status === "LOSS" ? "neg" : "neu";
      const dirCls = t.direction === "UP" ? "pos" : "neg";
      const pnlCls = pnlClass(t.pnl);
      const orderShort = (t.order_id || "").substring(0, 10);
      const invertTag = t.inverted ? " <span class='neg' style='font-size:10px'>INV</span>" : "";
      const fillCell = t.fill_type ? ("<td class='dim' style='font-size:11px'>" + t.fill_type + "</td>") : "<td class='dim'>—</td>";
      tr.innerHTML =
        "<td class='" + statusCls + "'><strong>" + t.status + "</strong></td>" +
        "<td>" + t.symbol + invertTag + "</td>" +
        "<td>" + t.timeframe + "</td>" +
        "<td class='" + dirCls + "'>" + t.direction + "</td>" +
        "<td>" + ((t.conviction || 0) * 100).toFixed(1) + "%</td>" +
        "<td>$" + (t.entry_price || 0).toFixed(3) + "</td>" +
        "<td>" + (t.tokens || 0).toFixed(2) + "</td>" +
        "<td>$" + (t.filled_usdc || 0).toFixed(2) + "</td>" +
        fillCell +
        "<td class='" + pnlCls + "'>" + (t.status === "OPEN" ? "—" : ((t.pnl >= 0 ? "+" : "") + "$" + (t.pnl || 0).toFixed(2))) + "</td>" +
        "<td class='dim' style='font-size:10px;font-family:monospace'>" + orderShort + "</td>" +
        "<td class='dim' style='font-size:11px'>" + tsShort + "</td>";
      tbody.appendChild(tr);
    });
    if ((d.conv_live_trade_record || []).length === 0) {
      tbody.innerHTML = "<tr><td colspan='12' class='dim' style='text-align:center'>No live trades yet — waiting for first signal</td></tr>";
    }

    // Live performance by symbol/timeframe
    const livePerf = document.getElementById("conviction-live-perf-body");
    livePerf.innerHTML = "";
    const liveSymTf = d.conv_live_by_symbol_tf || {};
    Object.entries(liveSymTf).forEach(([stf, stats]) => {
      const tr = document.createElement("tr");
      const wins = stats.wins || 0;
      const total = stats.trades || 1;
      const wr = ((wins / total) * 100).toFixed(1);
      const wrCls = wr >= 60 ? "pos" : wr >= 50 ? "neu" : "neg";
      const pnlCls = pnlClass(stats.pnl || 0);
      tr.innerHTML =
        "<td><strong>" + stf + "</strong></td>" +
        "<td>" + total + "</td>" +
        "<td>" + wins + "</td>" +
        "<td class='" + wrCls + "'>" + wr + "%</td>" +
        "<td class='" + pnlCls + "'>" + (stats.pnl >= 0 ? "+" : "") + "$" + (stats.pnl || 0).toFixed(2) + "</td>";
      livePerf.appendChild(tr);
    });
    if (Object.keys(liveSymTf).length === 0) {
      livePerf.innerHTML = "<tr><td colspan='5' class='dim' style='text-align:center'>No settled live trades yet</td></tr>";
    }

    // ── PAPER CONVICTION rendering (bottom section) ──────────────────────────

    // CONVICTION PAPER Trading Cards
    const convPaperOpen = d.conv_paper_open_count !== undefined ? d.conv_paper_open_count : 0;
    const convPaperClosed = d.conv_paper_closed_count !== undefined ? d.conv_paper_closed_count : 0;
    const convPaperHits = d.conv_paper_hits !== undefined ? d.conv_paper_hits : 0;
    const convPaperWr = d.conv_paper_hit_rate !== undefined && d.conv_paper_hit_rate !== null ? d.conv_paper_hit_rate : null;
    const convPaperPnl = d.conv_paper_pnl !== undefined ? d.conv_paper_pnl : 0;

    document.getElementById("conviction-conv-paper-open").textContent = String(convPaperOpen);
    document.getElementById("conviction-conv-paper-open-detail").textContent = "paper positions";

    document.getElementById("conviction-conv-paper-closed").textContent = String(convPaperClosed);
    const convPaperWrText = convPaperWr !== null ? convPaperHits + "W / " + Math.max(0, convPaperClosed - convPaperHits) + "L" : "0 settled";
    document.getElementById("conviction-conv-paper-closed-detail").textContent = convPaperWrText;

    const convPaperWrEl = document.getElementById("conviction-conv-paper-wr");
    convPaperWrEl.textContent = convPaperWr !== null ? convPaperWr.toFixed(1) + "%" : "—";
    convPaperWrEl.className = "card-value " + (convPaperWr && convPaperWr >= 65 ? "pos" : convPaperWr ? "neu" : "dim");

    const convPaperPnlEl = document.getElementById("conviction-conv-paper-pnl");
    convPaperPnlEl.textContent = (convPaperPnl >= 0 ? "+" : "") + "$" + convPaperPnl.toFixed(2);
    convPaperPnlEl.className = "card-value " + pnlClass(convPaperPnl);

    // Paper performance by symbol/timeframe (mirror of live table)
    const paperPerf = document.getElementById("conviction-paper-perf-body");
    paperPerf.innerHTML = "";
    const paperSymTf = d.conv_paper_by_symbol_tf || {};
    Object.entries(paperSymTf).forEach(([stf, stats]) => {
      const tr = document.createElement("tr");
      const wins = stats.wins || 0;
      const total = stats.trades || 1;
      const wr = ((wins / total) * 100).toFixed(1);
      const wrCls = wr >= 60 ? "pos" : wr >= 50 ? "neu" : "neg";
      const pnlCls = pnlClass(stats.pnl || 0);
      tr.innerHTML =
        "<td><strong>" + stf + "</strong></td>" +
        "<td>" + total + "</td>" +
        "<td>" + wins + "</td>" +
        "<td class='" + wrCls + "'>" + wr + "%</td>" +
        "<td class='" + pnlCls + "'>" + (stats.pnl >= 0 ? "+" : "") + "$" + (stats.pnl || 0).toFixed(2) + "</td>";
      paperPerf.appendChild(tr);
    });
    if (Object.keys(paperSymTf).length === 0) {
      paperPerf.innerHTML = "<tr><td colspan='5' class='dim' style='text-align:center'>No settled paper trades yet</td></tr>";
    }

    // ── Paper Trade Record table ─────────────────────────────────────────────
    document.getElementById("conviction-paper-recent-title").textContent =
      "Paper Trade Record (" + (d.conv_paper_trade_record || []).length + ", newest first)";
    const paperTbody = document.getElementById("conviction-paper-body");
    paperTbody.innerHTML = "";
    (d.conv_paper_trade_record || []).forEach(t => {
      const tr = document.createElement("tr");
      const tsRaw = (t.opened_at || 0);
      const tsShort = tsRaw > 0 ? new Date(tsRaw * 1000).toISOString().replace("T", " ").substring(5, 19) : "—";
      const statusCls = t.status === "WIN" ? "pos" : t.status === "LOSS" ? "neg" : "neu";
      const dirCls = t.direction === "UP" ? "pos" : "neg";
      const pnlCls = pnlClass(t.pnl);
      const orderShort = (t.order_id || "").substring(0, 10);
      const invertTag = t.inverted ? " <span class='neg' style='font-size:10px'>INV</span>" : "";
      const fillCell = t.fill_type ? ("<td class='dim' style='font-size:11px'>" + t.fill_type + "</td>") : "<td class='dim'>—</td>";
      tr.innerHTML =
        "<td class='" + statusCls + "'><strong>" + t.status + "</strong></td>" +
        "<td>" + t.symbol + invertTag + "</td>" +
        "<td>" + t.timeframe + "</td>" +
        "<td class='" + dirCls + "'>" + t.direction + "</td>" +
        "<td>" + ((t.conviction || 0) * 100).toFixed(1) + "%</td>" +
        "<td>$" + (t.entry_price || 0).toFixed(3) + "</td>" +
        "<td>" + (t.tokens || 0).toFixed(2) + "</td>" +
        "<td>$" + (t.filled_usdc || 0).toFixed(2) + "</td>" +
        fillCell +
        "<td class='" + pnlCls + "'>" + (t.status === "OPEN" ? "—" : ((t.pnl >= 0 ? "+" : "") + "$" + (t.pnl || 0).toFixed(2))) + "</td>" +
        "<td class='dim' style='font-size:10px;font-family:monospace'>" + orderShort + "</td>" +
        "<td class='dim' style='font-size:11px'>" + tsShort + "</td>";
      paperTbody.appendChild(tr);
    });
    if ((d.conv_paper_trade_record || []).length === 0) {
      paperTbody.innerHTML = "<tr><td colspan='12' class='dim' style='text-align:center'>No paper trades yet</td></tr>";
    }

    // Conviction signals recent table
    document.getElementById("conviction-conv-recent-title").textContent =
      "Recent Conviction Signals (" + (d.conv_recent_signals || []).length + ")";
    const convBody = document.getElementById("conviction-conv-body");
    convBody.innerHTML = "";
    (d.conv_recent_signals || []).forEach(t => {
      const tr = document.createElement("tr");
      const dirCls = t.direction === "UP" ? "pos" : "neg";
      const tsShort = (t.timestamp || "").replace("T", " ").substring(0, 19);
      tr.innerHTML =
        "<td>" + (t.symbol || "?") + "</td>" +
        "<td>" + (t.timeframe || "?") + "</td>" +
        "<td class='" + dirCls + "'>" + (t.direction || "?") + "</td>" +
        "<td>" + ((t.conviction || 0) * 100).toFixed(0) + "%</td>" +
        "<td>$" + (t.entry_price || 0).toFixed(3) + "</td>" +
        "<td>$" + (t.bet || 0).toFixed(2) + "</td>" +
        "<td class='dim' style='font-size:11px'>" + tsShort + "</td>";
      convBody.appendChild(tr);
    });
    if ((d.conv_recent_signals || []).length === 0) {
      convBody.innerHTML = "<tr><td colspan='7' class='dim' style='text-align:center'>No conviction signals yet</td></tr>";
    }
  }

  async function fetchConviction() {
    try {
      const res = await fetch("/api/conviction?t=" + Date.now());
      const data = await res.json();
      try {
        renderConviction(data);
      } catch (e) {
        var gel = document.getElementById("global-js-error");
        if (gel) { gel.style.display = "block"; gel.textContent = "renderConviction() error: " + e.message; }
      }
    } catch (e) {
      document.getElementById("conviction-error").textContent = "Failed to fetch conviction data: " + e.message;
      document.getElementById("conviction-error").classList.remove("hidden");
    }
  }

  // ── Arb rendering ─────────────────────────────────────────────────────────
  function renderArb(d) {
    const ARB_THRESHOLD_APY = d.entry_threshold_apy || 5.48; // read from config
    const errEl = document.getElementById("arb-error");
    if (d.error) {
      errEl.textContent = d.error;
      errEl.classList.remove("hidden");
      return;
    }
    errEl.classList.add("hidden");

    // Status
    const statusEl = document.getElementById("arb-status");
    statusEl.textContent = d.status || "IDLE";
    statusEl.className = "card-value " + (d.status === "ACTIVE" ? "pos" : "neu");
    const arbModeEl = document.getElementById("arb-mode-sub");
    const arbMode = d.mode || "VIRTUAL";
    arbModeEl.textContent = arbMode + " mode";
    arbModeEl.style.color = arbMode === "LIVE" ? "var(--green)" : "var(--blue)";

    // Funding collected
    const funding = d.total_funding_collected || 0;
    const fundEl = document.getElementById("arb-funding");
    fundEl.textContent = (funding >= 0 ? "+" : "") + "$" + funding.toFixed(4);
    fundEl.className = "card-value " + pnlClass(funding);
    document.getElementById("arb-trades").textContent = (d.trade_count || 0) + " cycles";

    // Realized PnL
    const pnl = d.total_realized_pnl || 0;
    const pnlEl = document.getElementById("arb-pnl");
    pnlEl.textContent = (pnl >= 0 ? "+" : "") + "$" + pnl.toFixed(4);
    pnlEl.className = "card-value " + pnlClass(pnl);

    // Active symbol + position details
    const activeSym = d.active_symbol || "";
    const symEl = document.getElementById("arb-active-symbol");
    if (activeSym) {
      symEl.textContent = activeSym.replace("USDT", "");
      symEl.className = "card-value pos";
    } else {
      symEl.textContent = "—";
      symEl.className = "card-value dim";
    }
    const qty = d.spot_qty || 0;
    const baseAsset = activeSym ? activeSym.replace("USDT", "") : "—";
    document.getElementById("arb-qty").textContent =
      qty > 0 ? qty.toFixed(5) + " " + baseAsset + " each leg" : "no open position";

    const entry = d.entry_price || 0;
    const cur = d.current_price || 0;
    const entryEl = document.getElementById("arb-entry");
    if (entry > 0 && cur > 0) {
      entryEl.textContent = "$" + entry.toFixed(2) + " → $" + cur.toFixed(2);
      const drift = cur - entry;
      entryEl.className = "card-value " + (Math.abs(drift) < 0.10 ? "dim" : drift > 0 ? "pos" : "neg");
    } else if (entry > 0) {
      entryEl.textContent = "$" + entry.toFixed(2);
      entryEl.className = "card-value dim";
    } else {
      entryEl.textContent = "—";
      entryEl.className = "card-value dim";
    }
    if (d.entry_time && d.entry_time > 0) {
      const ageSec = Math.round((Date.now() / 1000) - d.entry_time);
      const absAge = Math.abs(ageSec);
      const ageStr = absAge > 3600
        ? (absAge / 3600).toFixed(1) + "h"
        : absAge > 60 ? Math.round(absAge / 60) + "m" : absAge + "s";
      document.getElementById("arb-age").textContent = "open " + ageStr;
    } else {
      document.getElementById("arb-age").textContent = "not in position";
    }

    // Pending funding + errors
    const pending = d.pending_funding || 0;
    const pendEl = document.getElementById("arb-pending");
    pendEl.textContent = pending > 0 ? "+$" + pending.toFixed(4) : "—";
    pendEl.className = "card-value " + (pending > 0 ? "pos" : "dim");
    const errCount = d.error_count || 0;
    document.getElementById("arb-errors-sub").textContent = errCount + " error" + (errCount === 1 ? "" : "s");

    // Live rates table
    const ratesBody = document.getElementById("arb-rates-body");
    ratesBody.innerHTML = "";
    const rates = d.current_rates || {};
    const syms = Object.keys(rates).sort();
    if (syms.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td colspan='5' style='color:var(--muted);text-align:center;padding:12px'>No rate data yet — waiting for first scan</td>";
      ratesBody.appendChild(tr);
    } else {
      syms.forEach(sym => {
        const r = rates[sym];
        const apy = r.apy;
        const absApy = Math.abs(apy);
        const isPos = apy > 0;
        const isNeg = apy < 0;
        const color = isPos ? "var(--green)" : isNeg ? "var(--red)" : "var(--muted)";
        const direction = isPos ? "LONGS PAY ▲" : isNeg ? "SHORTS PAY ▼" : "NEUTRAL";
        const dirColor = isPos ? "var(--green)" : isNeg ? "var(--red)" : "var(--muted)";
        // threshold bar: % of ±10.95% threshold
        const barPct = Math.min(100, (absApy / ARB_THRESHOLD_APY) * 100).toFixed(0);
        const barColor = absApy >= ARB_THRESHOLD_APY ? "var(--green)" : isNeg ? "var(--red)" : "var(--blue)";
        const isActive = sym === (d.active_symbol || "");
        const tr = document.createElement("tr");
        if (isActive) tr.style.background = "rgba(63,185,80,0.07)";
        tr.innerHTML =
          "<td style='font-weight:600'>" + sym.replace("USDT", "") +
            (isActive ? " <span style='font-size:10px;color:var(--green)'>● ACTIVE</span>" : "") + "</td>" +
          "<td style='color:" + color + ";font-family:monospace'>" + r.rate.toFixed(6) + "</td>" +
          "<td style='color:" + color + ";font-weight:600'>" + (isNeg ? "" : "+") + apy.toFixed(1) + "%</td>" +
          "<td style='color:" + dirColor + ";font-size:12px'>" + direction + "</td>" +
          "<td style='width:180px'>" +
            "<div style='display:flex;align-items:center;gap:8px'>" +
              "<div style='flex:1;background:var(--border);border-radius:3px;height:6px'>" +
                "<div style='width:" + barPct + "%;background:" + barColor + ";height:6px;border-radius:3px'></div>" +
              "</div>" +
              "<span style='font-size:11px;color:var(--muted);white-space:nowrap'>" + barPct + "%</span>" +
            "</div>" +
          "</td>";
        ratesBody.appendChild(tr);
      });
    }

    // Log
    const logEl = document.getElementById("arb-log");
    const lines = d.recent_log || [];
    logEl.textContent = lines.length > 0 ? lines.join("\\n") : "No log entries yet.";
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function fetchArb() {
    try {
      const res = await fetch("/api/arb");
      const data = await res.json();
      try {
        renderArb(data);
      } catch (e) {
        var gel = document.getElementById("global-js-error");
        if (gel) { gel.style.display = "block"; gel.textContent = "renderArb() error: " + e.message + " — " + (e.stack || "").split("\\n")[1]; }
      }
    } catch (e) {
      document.getElementById("arb-error").textContent = "Failed to fetch arb data: " + e.message;
      document.getElementById("arb-error").classList.remove("hidden");
    }
  }

  // ── Swarm rendering ───────────────────────────────────────────────────────
  function renderSwarm(d) {
    const errEl = document.getElementById("swarm-error");
    if (d.error) {
      errEl.textContent = d.error;
      errEl.classList.remove("hidden");
      return;
    }
    errEl.classList.add("hidden");

    // ── Era-split cards (live strategy = NO only) ──────────────────────────────
    function _setVal(elId, val, fmt, cls) {
      const el = document.getElementById(elId);
      if (val === null || val === undefined) {
        el.textContent = "—";
        el.className = "card-value dim" + (cls && cls.indexOf("font-size") >= 0 ? " " + cls : "");
        return;
      }
      el.textContent = fmt(val);
      el.className = "card-value " + (cls || "");
    }
    const live = d.live_strategy   || {n: 0, wr: null, pnl: 0};
    const r14  = d.recent_14d_no   || {n: 0, wr: null, pnl: 0};
    const lgy  = d.legacy_yes      || {n: 0, wr: null, pnl: 0};

    _setVal("sr-live-wr",  live.wr,  v => v.toFixed(1) + "%", pnlClass((live.wr || 0) - 50));
    document.getElementById("sr-live-settled").textContent = (live.n || 0) + " settled";
    _setVal("sr-live-pnl", live.pnl, v => (v >= 0 ? "+" : "") + "$" + v.toFixed(2), pnlClass(live.pnl));

    _setVal("sr-14d-wr",   r14.wr,   v => v.toFixed(1) + "%", pnlClass((r14.wr || 0) - 50));
    document.getElementById("sr-14d-settled").textContent = (r14.n || 0) + " settled";
    _setVal("sr-14d-pnl",  r14.pnl,  v => (v >= 0 ? "+" : "") + "$" + v.toFixed(2), pnlClass(r14.pnl));

    // Legacy YES is intentionally rendered with smaller font (dim card)
    const lgyEl = document.getElementById("sr-legacy-pnl");
    lgyEl.textContent = (lgy.pnl >= 0 ? "+" : "") + "$" + lgy.pnl.toFixed(2);
    lgyEl.className = "card-value " + pnlClass(lgy.pnl);
    lgyEl.style.fontSize = "16px";
    document.getElementById("sr-legacy-settled").textContent =
      (lgy.n || 0) + " settled · cannot recur";

    // ── Real trade cards ───────────────────────────────────────────────────────
    const srWrEl = document.getElementById("sr-wr");
    if (d.real_win_rate !== null && d.real_win_rate !== undefined) {
      srWrEl.textContent = d.real_win_rate.toFixed(1) + "%";
      srWrEl.className = "card-value " + pnlClass(d.real_win_rate - 50);
    } else {
      srWrEl.textContent = "—";
      srWrEl.className = "card-value dim";
    }
    document.getElementById("sr-settled").textContent = (d.real_settled_count || 0) + " settled";

    const srPnlEl = document.getElementById("sr-pnl");
    const srPnl = d.real_pnl || 0;
    srPnlEl.textContent = (srPnl >= 0 ? "+" : "") + "$" + srPnl.toFixed(2);
    srPnlEl.className = "card-value " + pnlClass(srPnl);
    const srDp = d.real_daily_pnl || 0;
    document.getElementById("sr-dpnl").textContent =
      "today: " + (srDp >= 0 ? "+" : "") + "$" + srDp.toFixed(2);

    // Net PnL = settled cash + pending CTF tokens (won but not yet redeemed)
    const srNetPnlEl = document.getElementById("sr-net-pnl");
    const srNetPnl = d.real_net_pnl != null ? d.real_net_pnl : srPnl;
    srNetPnlEl.textContent = (srNetPnl >= 0 ? "+" : "") + "$" + srNetPnl.toFixed(2);
    srNetPnlEl.className = "card-value " + pnlClass(srNetPnl);
    const pendingCtf = d.pending_ctf_usdc || 0;
    document.getElementById("sr-net-sub").textContent =
      "settled: " + (srPnl >= 0 ? "+" : "") + "$" + srPnl.toFixed(2) +
      " + tokens: $" + pendingCtf.toFixed(2);

    // Pending tokens card
    const srPendEl = document.getElementById("sr-pending");
    srPendEl.textContent = "$" + pendingCtf.toFixed(2);
    srPendEl.className = "card-value " + (pendingCtf > 0 ? "pos" : "neu");

    // Open count: prefer on-chain truth (18 in-play), fall back to JSONL count
    const openCount = (d.on_chain_in_play_count !== undefined && d.on_chain_in_play_count !== null)
                      ? d.on_chain_in_play_count
                      : (d.real_open_count || 0);
    document.getElementById("sr-open").textContent = openCount;
    const openSubEl = document.getElementById("sr-open-sub");
    if (openSubEl) {
      const curval = d.on_chain_in_play_curval || 0;
      const trackedCount = (d.real_open_count || 0);
      openSubEl.textContent = "on-chain · cur $" + curval.toFixed(2)
        + " · " + trackedCount + " tracked";
    }

    if (d.last_updated) {
      const dt = new Date(d.last_updated);
      document.getElementById("s-ts").textContent = dt.toLocaleTimeString();
    }

    // ── Open positions (on-chain truth) ────────────────────────────────────────
    const inPlay = d.on_chain_in_play || [];
    const inPlayCount = d.on_chain_in_play_count || inPlay.length;
    const inPlayCurval = d.on_chain_in_play_curval || 0;
    document.getElementById("swarm-real-open-title").textContent =
      "Open Positions (" + inPlayCount + ") — current value $" + inPlayCurval.toFixed(2);
    const realOpenBody = document.getElementById("swarm-real-open-body");
    realOpenBody.innerHTML = "";
    if (inPlay.length === 0) {
      realOpenBody.innerHTML = "<tr><td colspan='10' style='color:var(--muted);text-align:center;padding:12px'>No open on-chain positions</td></tr>";
    } else {
      inPlay.forEach((t, i) => {
        const outcomeRaw = t.outcome || "";
        const outcomeU = outcomeRaw.toUpperCase();
        const outcome = outcomeRaw.length > 16 ? outcomeRaw.substring(0, 14) + "…" : outcomeRaw;
        const dirClass = outcomeU === "YES" ? "dir-yes"
                       : outcomeU === "NO"  ? "dir-no"
                       : "dim";
        const trackedTag = t.tracked
          ? "<span style='color:var(--green);font-weight:700' title='matched in JSONL'>★ </span>"
          : "<span style='color:var(--muted)' title='no JSONL record'>· </span>";
        const cost = t.est_cost || 0;
        const cur  = t.current_value || 0;
        const pnlNow = t.est_pnl_now;
        const pnlCls = pnlNow > 0 ? "pos" : pnlNow < 0 ? "neg" : "dim";
        const score = (typeof t.score === "number") ? t.score.toFixed(2) : "—";
        let entryDt = "—";
        if (t.entry_ts) {
          try {
            entryDt = new Date(t.entry_ts).toLocaleString([], {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
          } catch(e) {}
        }
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + (i + 1) + "</td>" +
          "<td class='title-cell' title='" + escHtml(t.title) + "'>" + trackedTag + escHtml(t.title) + "</td>" +
          "<td class='" + dirClass + "'>" + outcome + "</td>" +
          "<td>" + (t.cur_price || 0).toFixed(3) + "</td>" +
          "<td>" + (t.size || 0).toFixed(2) + "</td>" +
          "<td>$" + cur.toFixed(2) + "</td>" +
          "<td class='dim'>$" + cost.toFixed(2) + "</td>" +
          "<td class='" + pnlCls + "'>" + (pnlNow >= 0 ? "+" : "") + "$" + pnlNow.toFixed(2) + "</td>" +
          "<td>" + score + "</td>" +
          "<td style='font-size:11px;color:var(--muted)'>" + entryDt + "</td>";
        realOpenBody.appendChild(tr);
      });
    }

    // ── Won, pending UMA finalize (on-chain) ──────────────────────────────────
    const pendingUma = d.on_chain_won_pending || [];
    const pendingUmaCount = d.on_chain_won_pending_count || pendingUma.length;
    const pendingUmaValue = d.on_chain_won_pending_value || 0;
    document.getElementById("swarm-pending-uma-title").textContent =
      "Won, Pending UMA Finalize (" + pendingUmaCount + ") — value $" + pendingUmaValue.toFixed(2);
    const pendingUmaBody = document.getElementById("swarm-pending-uma-body");
    pendingUmaBody.innerHTML = "";
    if (pendingUma.length === 0) {
      pendingUmaBody.innerHTML = "<tr><td colspan='8' style='color:var(--muted);text-align:center;padding:12px'>No pending UMA finalizations</td></tr>";
    } else {
      pendingUma.forEach((t, i) => {
        const outcomeRaw = t.outcome || "";
        const outcomeU = outcomeRaw.toUpperCase();
        const outcome = outcomeRaw.length > 16 ? outcomeRaw.substring(0, 14) + "…" : outcomeRaw;
        const dirClass = outcomeU === "YES" ? "dir-yes"
                       : outcomeU === "NO"  ? "dir-no"
                       : "dim";
        const trackedTag = t.tracked ? "<span style='color:var(--green);font-weight:700'>★ </span>" : "<span class='dim'>· </span>";
        const cost = t.est_cost || 0;
        const winPnl = (t.size || 0) - cost;
        const score = (typeof t.score === "number") ? t.score.toFixed(2) : "—";
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + (i + 1) + "</td>" +
          "<td class='title-cell' title='" + escHtml(t.title) + "'>" + trackedTag + escHtml(t.title) + "</td>" +
          "<td class='" + dirClass + "'>" + outcome + "</td>" +
          "<td>" + (t.cur_price || 0).toFixed(3) + "</td>" +
          "<td>" + (t.size || 0).toFixed(2) + "</td>" +
          "<td class='dim'>$" + cost.toFixed(2) + "</td>" +
          "<td class='pos'>+$" + winPnl.toFixed(2) + "</td>" +
          "<td>" + score + "</td>";
        pendingUmaBody.appendChild(tr);
      });
    }

    // ── Real settled trades ────────────────────────────────────────────────────
    const srClosedSec = document.getElementById("sr-closed-section");
    if (d.real_closed_picks && d.real_closed_picks.length > 0) {
      srClosedSec.classList.remove("hidden");
      const srCb = document.getElementById("sr-closed-body");
      srCb.innerHTML = "";
      d.real_closed_picks.forEach((t, i) => {
        const pnlV = t.pnl || 0;
        let srEntryDt = "—";
        if (t.entry_ts) {
          try {
            const dt = new Date(t.entry_ts);
            srEntryDt = dt.toLocaleString([], {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
          } catch(e) {}
        }
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + (i + 1) + "</td>" +
          "<td class='title-cell' title='" + escHtml(t.question) + "'>" + escHtml(t.question) + "</td>" +
          "<td class='" + (t.direction === "YES" ? "dir-yes" : "dir-no") + "'>" + t.direction + "</td>" +
          "<td>" + (t.ask || 0).toFixed(3) + "</td>" +
          "<td>$" + (t.bet || 0).toFixed(2) + "</td>" +
          "<td>" + (t.outcome || "—") + "</td>" +
          "<td class='" + pnlClass(pnlV) + "'>" + (pnlV >= 0 ? "+" : "") + "$" + pnlV.toFixed(4) + "</td>" +
          "<td style='font-size:11px;color:var(--muted);white-space:nowrap'>" + srEntryDt + "</td>";
        srCb.appendChild(tr);
      });
    } else {
      srClosedSec.classList.add("hidden");
    }

    // ── Paper analysis cards ───────────────────────────────────────────────────
    const wrEl = document.getElementById("s-wr");
    if (d.win_rate !== null && d.win_rate !== undefined) {
      wrEl.textContent = d.win_rate.toFixed(1) + "%";
      wrEl.className = "card-value " + pnlClass(d.win_rate - 50);
    } else {
      wrEl.textContent = "—";
      wrEl.className = "card-value dim";
    }
    document.getElementById("s-settled").textContent = (d.settled_count || 0) + " settled";

    const pnlEl = document.getElementById("s-pnl");
    const pnl = d.total_pnl || 0;
    pnlEl.textContent = (pnl >= 0 ? "+" : "") + "$" + pnl.toFixed(2);
    pnlEl.className = "card-value " + pnlClass(pnl);
    const dp = d.daily_pnl || 0;
    document.getElementById("s-dpnl").textContent =
      "today: " + (dp >= 0 ? "+" : "") + "$" + dp.toFixed(2);

    document.getElementById("s-open").textContent = d.open_count || 0;

    // Last cycle picks
    const picksEl = document.getElementById("swarm-picks");
    picksEl.innerHTML = "";
    if (!d.last_cycle_picks || d.last_cycle_picks.length === 0) {
      picksEl.innerHTML = "<p style='color:var(--muted);padding:12px 0'>No picks this cycle.</p>";
    } else {
      d.last_cycle_picks.forEach((p, i) => {
        const barW = Math.round(p.score * 120);
        const dirClass = p.direction === "YES" ? "dir-yes" : "dir-no";
        const whaleLine = p.whale && p.whale.has_signal
          ? "<span class='whale-tag'>WHALE " + p.whale.direction + " " + (p.whale.strength * 100).toFixed(0) + "%</span> "
          : "";
        const div = document.createElement("div");
        div.className = "pick-card";
        div.innerHTML =
          "<div class='pick-q'>" +
            "<span style='margin-right:8px'>#" + (i+1) + "</span>" +
            "<span class='" + dirClass + "'>" + p.direction + "</span>" +
            "  &nbsp;" +
            "<span style='display:inline-block;width:" + barW + "px' class='score-bar'></span>" +
            "<span style='font-size:12px;color:var(--muted)'>" + p.score.toFixed(3) + "</span>" +
            "  &nbsp;" + whaleLine +
            escHtml(p.question) +
          "</div>" +
          "<div class='pick-meta'>" +
            "<span>YES " + (p.yes_price * 100).toFixed(1) + "%</span>" +
            "<span>vol24h $" + (p.vol_24h || 0).toLocaleString() + "</span>" +
            "<span>" + p.yes_votes + "Y / " + p.no_votes + "N  conf " + (p.avg_conf || 0).toFixed(0) + "%</span>" +
          "</div>";
        picksEl.appendChild(div);
      });
    }

    // Paper settled trades
    const closedSec = document.getElementById("s-closed-section");
    if (d.closed_picks && d.closed_picks.length > 0) {
      closedSec.classList.remove("hidden");
      const cb = document.getElementById("s-closed-body");
      cb.innerHTML = "";
      d.closed_picks.forEach((t, i) => {
        const pnlV = t.pnl || 0;
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + (i+1) + "</td>" +
          "<td class='title-cell' title='" + escHtml(t.question) + "'>" + escHtml(t.question) + "</td>" +
          "<td class='" + (t.direction === "YES" ? "dir-yes" : "dir-no") + "'>" + t.direction + "</td>" +
          "<td>" + (t.score || 0).toFixed(3) + "</td>" +
          "<td>" + (t.entry_price || 0).toFixed(3) + "</td>" +
          "<td>" + (t.outcome || "—") + "</td>" +
          "<td class='" + pnlClass(pnlV) + "'>" + (pnlV >= 0 ? "+" : "") + "$" + pnlV.toFixed(2) + "</td>";
        cb.appendChild(tr);
      });
    } else {
      closedSec.classList.add("hidden");
    }
  }

  async function fetchSwarm() {
    try {
      const res = await fetch("/api/swarm");
      const data = await res.json();
      try {
        renderSwarm(data);
      } catch (e) {
        var gel = document.getElementById("global-js-error");
        if (gel) { gel.style.display = "block"; gel.textContent = "renderSwarm() error: " + e.message + " — " + (e.stack || "").split("\\n")[1]; }
      }
    } catch (e) {
      document.getElementById("swarm-error").textContent = "Failed to fetch swarm data: " + e.message;
      document.getElementById("swarm-error").classList.remove("hidden");
    }
  }

  // ── Auto-refresh countdown ─────────────────────────────────────────────────
  let countdown = 30;
  function tick() {
    document.getElementById("countdown").textContent = countdown;
    if (countdown <= 0) {
      countdown = 5;  // refresh every 5 seconds instead of 30
      if (_activeTab === "swarm") fetchSwarm();
      else if (_activeTab === "arb") fetchArb();
      else if (_activeTab === "conviction") fetchConviction();
      else { fetchData(); fetchWallet(); }
    } else {
      countdown--;
    }
  }

  async function fetchData() {
    try {
      const res = await fetch("/api/state");
      const data = await res.json();
      // Also fetch 1H crypto stats
      try {
        const res1h = await fetch("/api/crypto-1h");
        const data1h = await res1h.json();
        data.crypto_1h = data1h;
      } catch (e) {
        console.log("Could not fetch 1H crypto data:", e.message);
      }
      try {
        render(data);
      } catch (e) {
        var gel = document.getElementById("global-js-error");
        if (gel) { gel.style.display = "block"; gel.textContent = "render() error: " + e.message + " — " + (e.stack || "").split("\\n")[1]; }
      }
    } catch (e) {
      document.getElementById("error-box").textContent = "Failed to fetch data: " + e.message;
      document.getElementById("error-box").classList.remove("hidden");
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  fetchData();
  fetchWallet();
  setInterval(tick, 1000);
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # suppress default access log
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/api/state", "/api/refresh"):
            raw  = _load_raw_state()
            resp = _build_api_response(raw)
            self._send_json(resp)
        elif path == "/api/crypto-1h":
            self._send_json(_load_1h_crypto_state())
        elif path == "/api/swarm":
            self._send_json(_load_swarm_state())
        elif path == "/api/arb":
            self._send_json(_load_arb_state())
        elif path == "/api/conviction":
            self._send_json(_load_conviction_state())
        elif path == "/api/strike":
            self._send_json(_load_strike_state())
        elif path == "/api/wallet":
            self._send_json(_load_wallet_state())
        elif path == "/":
            self._send_html(_HTML)
        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket trading dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    args = parser.parse_args()

    mode_str = "VIRTUAL" if _VIRTUAL_MODE else "REAL"
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] Mode: {mode_str}")
    print(f"[dashboard] State: {_ROOT / _VIRTUAL_STATE_PATH}")
    print(f"[dashboard] Listening on http://{args.host}:{args.port}")
    print("[dashboard] Open the URL in your browser. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
