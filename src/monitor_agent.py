#!/usr/bin/env python3
"""
src/monitor_agent.py — Paranoid profit-focused monitoring agent.

Runs every 30 minutes via cron. Treats every dollar like it matters.
Checks health, performance, and balance — alerts FAST, acts conservatively.

What it does:
  Every run  — health check (bot alive? log stale? balance OK?)
  Every run  — WR + consecutive loss streak (alert immediately if bad)
  Every run  — hourly PnL rate ("losing $X/hr right now")
  Every run  — stuck open positions (windows are 5 min — anything >10 min is suspicious)
  Every run  — balance drop alert (>20% drop since last check)
  Daily 09Z  — calibration (session hours + CLOB band auto-tighten from real data)
  Daily 00Z  — full performance report via ntfy (with direction + session breakdown)

What it never does:
  — Increase bet size (only tightens/blocks, never opens up)
  — Make code changes (only edits crypto_params.yaml)
  — Act without logging first

Cron entry:
  */30 * * * * cd /home/ubuntu/poly-model && source .venv/bin/activate && python src/monitor_agent.py >> logs/monitor_agent.log 2>&1
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
_REAL_STATE   = _ROOT / "data"   / "real_state.json"
_MON_STATE    = _ROOT / "data"   / "monitor_state.json"
_PARAMS       = _ROOT / "config" / "crypto_params.yaml"
_CRYPTO_LOG   = _ROOT / "logs"   / "crypto.log"
_MON_LOG      = _ROOT / "logs"   / "monitor_agent.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.FileHandler(_MON_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("monitor")

# ── Env ───────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")

# ── Thresholds ────────────────────────────────────────────────────────────────
# Break-even WR at avg fill 0.74 is 74%.  We warn when WR falls below 65%
# (10% below break-even) — that means real money is bleeding.
WR_WARN_THRESHOLD    = 0.65   # last-20 WR below this triggers alert immediately
WR_CRITICAL          = 0.50   # last-20 WR below this → send critical alert
LOSS_STREAK_ALERT    = 4      # consecutive losses → immediate alert regardless of WR
MIN_TRADES_WR        = 10     # minimum settled trades before WR check is meaningful

MIN_TRADES_SESSION   = 20     # minimum trades per hour to consider session change
MIN_TRADES_BAND      = 25     # minimum trades per price bucket to recalibrate band
SESSION_BAD_GAP      = -0.05  # hour is "bad" if WR is 5%+ below break-even
SESSION_REHAB_GAP    = +0.03  # hour is "rehabilitated" if WR is 3%+ above break-even
SESSION_REHAB_MIN    = 40     # minimum trades before un-blocking a rehabilitated hour
BAND_BAD_GAP         = -0.04  # bucket is "bad" if WR is 4%+ below break-even

LOG_STALE_MINUTES    = 6      # crypto.log older than this → bot is stuck
STUCK_POSITION_MINS  = 10     # open position older than this → suspicious (windows = 5 min)
BALANCE_DROP_PCT     = 0.20   # alert if balance drops >20% since last monitor run
MIN_BALANCE_MULT     = 2.0    # alert if balance < MIN_BALANCE_MULT × max_bet_abs


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _send_alert(body: str, title: str = "PM Monitor") -> None:
    try:
        import httpx
        httpx.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            content=body.encode("utf-8"),
            headers={"Title": title},
            timeout=6.0,
        )
        log.info("ntfy sent: %s", title)
    except Exception as exc:
        log.warning("ntfy failed: %s", exc)


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("load %s failed: %s", path.name, exc)
        return default if default is not None else {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_param(name: str) -> Optional[str]:
    content = _PARAMS.read_text()
    m = re.search(rf"^{name}:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def _write_param(name: str, value) -> None:
    content = _PARAMS.read_text()
    new_content = re.sub(
        rf"^({name}:\s*)(.+)$",
        lambda m: m.group(1) + str(value),
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        log.warning("_write_param: no change for %s = %s", name, value)
        return
    _PARAMS.write_text(new_content)
    log.info("config: %s → %s", name, value)


def _write_list_param(name: str, values: list) -> None:
    content = _PARAMS.read_text()
    new_val = "[" + ", ".join(str(v) for v in sorted(values)) + "]"
    new_content = re.sub(
        rf"^({name}:\s*)(\[.*?\])$",
        lambda m: m.group(1) + new_val,
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        log.warning("_write_list_param: no change for %s = %s", name, new_val)
        return
    _PARAMS.write_text(new_content)
    log.info("config: %s → %s", name, new_val)


def _load_params() -> dict:
    try:
        import yaml
        with open(_PARAMS) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        log.error("load params failed: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Health checks
# ══════════════════════════════════════════════════════════════════════════════

def _bot_running() -> bool:
    r = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    return "crypto" in r.stdout


def _log_stale_minutes() -> float:
    if not _CRYPTO_LOG.exists():
        return 9999.0
    return (time.time() - _CRYPTO_LOG.stat().st_mtime) / 60.0


def _restart_bot() -> bool:
    subprocess.run(["screen", "-S", "crypto", "-X", "quit"], capture_output=True)
    time.sleep(3)
    subprocess.Popen(
        "screen -dmS crypto bash -c "
        "'cd /home/ubuntu/poly-model && source .venv/bin/activate "
        "&& python src/crypto_main.py >> logs/crypto.log 2>&1'",
        shell=True,
    )
    time.sleep(6)
    return _bot_running()


# ══════════════════════════════════════════════════════════════════════════════
# Performance analysis
# ══════════════════════════════════════════════════════════════════════════════

def _recent_wr(settled: list, n: int = 20) -> Optional[float]:
    if len(settled) < MIN_TRADES_WR:
        return None
    recent = sorted(settled, key=lambda p: p.get("fill_time", ""))[-n:]
    return sum(1 for p in recent if p["realized_pnl"] > 0) / len(recent)


def _consecutive_loss_streak(settled: list) -> int:
    """Count how many of the most recent trades are consecutive losses."""
    recent = sorted(settled, key=lambda p: p.get("fill_time", ""))
    streak = 0
    for p in reversed(recent):
        if p.get("realized_pnl", 0) <= 0:
            streak += 1
        else:
            break
    return streak


def _today_stats(settled: list) -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    t = [p for p in settled if (p.get("fill_time") or "")[:10] == today]
    wins = [p for p in t if p["realized_pnl"] > 0]
    pnl  = sum(p["realized_pnl"] for p in t)
    fees = sum(p.get("size_usdc", 0) * 0.02 for p in t)

    up_t   = [p for p in t if p.get("direction") == "YES"]
    down_t = [p for p in t if p.get("direction") == "NO"]
    up_w   = [p for p in up_t   if p["realized_pnl"] > 0]
    down_w = [p for p in down_t if p["realized_pnl"] > 0]

    return {
        "count": len(t),
        "wins":  len(wins),
        "wr":    len(wins) / len(t) if t else 0.0,
        "gross": round(pnl, 2),
        "net":   round(pnl - fees, 2),
        "up_count":  len(up_t),
        "up_wr":     len(up_w) / len(up_t) if up_t else 0.0,
        "down_count": len(down_t),
        "down_wr":    len(down_w) / len(down_t) if down_t else 0.0,
    }


def _pnl_rate_usd_per_hour(settled: list, window_hours: float = 2.0) -> Optional[float]:
    """
    Returns net PnL rate in $/hr over the last `window_hours` hours.
    Positive = making money. Negative = bleeding.
    Returns None if fewer than 5 trades in the window.
    """
    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - window_hours * 3600
    recent = [
        p for p in settled
        if _parse_ts(p.get("fill_time")) >= cutoff
    ]
    if len(recent) < 5:
        return None
    net = sum(p["realized_pnl"] - p.get("size_usdc", 0) * 0.02 for p in recent)
    return net / window_hours


def _parse_ts(ts_str: Optional[str]) -> float:
    if not ts_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


def _stuck_positions(open_positions: list, now_ts: float) -> list:
    """Return positions that have been open longer than STUCK_POSITION_MINS."""
    stuck = []
    for p in open_positions:
        filled_at = _parse_ts(p.get("fill_time"))
        if filled_at > 0:
            age_min = (now_ts - filled_at) / 60.0
            if age_min > STUCK_POSITION_MINS:
                stuck.append((p, age_min))
    return stuck


def _all_time_direction_stats(settled: list) -> dict:
    up_t   = [p for p in settled if p.get("direction") == "YES"]
    down_t = [p for p in settled if p.get("direction") == "NO"]
    up_w   = [p for p in up_t   if p["realized_pnl"] > 0]
    down_w = [p for p in down_t if p["realized_pnl"] > 0]
    up_pnl   = sum(p["realized_pnl"] for p in up_t)
    down_pnl = sum(p["realized_pnl"] for p in down_t)
    return {
        "up":   {"count": len(up_t),   "wr": len(up_w)/len(up_t)     if up_t   else 0, "pnl": round(up_pnl, 2)},
        "down": {"count": len(down_t), "wr": len(down_w)/len(down_t) if down_t else 0, "pnl": round(down_pnl, 2)},
    }


def _session_stats(settled: list) -> dict:
    """WR and net PnL by UTC hour."""
    by_hour: dict[int, list] = defaultdict(list)
    for p in settled:
        h = int((p.get("fill_time") or "00:00")[11:13])
        by_hour[h].append(p)

    result = {}
    for h, trades in sorted(by_hour.items()):
        wins = [p for p in trades if p["realized_pnl"] > 0]
        pnl  = sum(p["realized_pnl"] for p in trades)
        fees = sum(p.get("size_usdc", 0) * 0.02 for p in trades)
        result[h] = {
            "count": len(trades),
            "wr":    len(wins) / len(trades),
            "net":   round(pnl - fees, 2),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Daily calibration
# ══════════════════════════════════════════════════════════════════════════════

def _calibrate_session(settled: list, params: dict) -> list[str]:
    start   = params.get("trade_hour_start", 10)
    end     = params.get("trade_hour_end", 18)
    blocked = set(int(h) for h in params.get("trade_hour_block", []))

    by_hour: dict[int, list] = defaultdict(list)
    for p in settled:
        h = int((p.get("fill_time") or "00")[11:13])
        by_hour[h].append(p)

    new_blocked = set(blocked)
    changes = []

    for h in range(24):
        trades = by_hour[h]
        if len(trades) < MIN_TRADES_SESSION:
            continue
        wins    = sum(1 for p in trades if p["realized_pnl"] > 0)
        wr      = wins / len(trades)
        avg_fp  = sum(p.get("fill_price", 0.74) for p in trades) / len(trades)
        gap     = wr - avg_fp

        in_window = start <= h < end and h not in blocked

        if in_window and gap < SESSION_BAD_GAP:
            new_blocked.add(h)
            changes.append(
                f"Block {h:02d}:00 UTC — {wr*100:.0f}% WR vs {avg_fp*100:.0f}% "
                f"needed (gap {gap*100:+.0f}%, {len(trades)} trades)"
            )
        elif h in blocked and gap > SESSION_REHAB_GAP \
                and len(trades) >= SESSION_REHAB_MIN:
            new_blocked.discard(h)
            changes.append(
                f"Unblock {h:02d}:00 UTC — rehabilitated: {wr*100:.0f}% WR "
                f"(gap {gap*100:+.0f}%, {len(trades)} trades)"
            )

    if changes:
        _write_list_param("trade_hour_block", sorted(new_blocked))
    return changes


def _calibrate_band(settled: list, params: dict) -> list[str]:
    cur_min = float(params.get("min_clob_price", 0.72))
    cur_max = float(params.get("max_clob_price", 0.76))

    inband = [p for p in settled
              if cur_min <= p.get("fill_price", 0) <= cur_max]
    if len(inband) < MIN_TRADES_BAND * 3:
        return []

    by_bucket: dict[float, list] = defaultdict(list)
    for p in inband:
        b = round(p.get("fill_price", 0) * 50) / 50  # 0.02-wide buckets
        by_bucket[b].append(p)

    new_min, new_max = cur_min, cur_max
    changes = []

    for b in sorted(by_bucket.keys()):
        trades = by_bucket[b]
        if len(trades) < MIN_TRADES_BAND:
            continue
        wins = sum(1 for p in trades if p["realized_pnl"] > 0)
        wr   = wins / len(trades)
        gap  = wr - b

        if gap < BAND_BAD_GAP:
            if b <= cur_min + 0.03:
                candidate = round(b + 0.02, 2)
                if candidate > new_min:
                    new_min = candidate
                    changes.append(
                        f"Raise min_clob_price {cur_min:.2f}→{new_min:.2f}: "
                        f"bucket {b:.2f} at {wr*100:.0f}% WR (gap {gap*100:+.0f}%, "
                        f"{len(trades)} trades)"
                    )
            if b >= cur_max - 0.03:
                candidate = round(b - 0.02, 2)
                if candidate < new_max:
                    new_max = candidate
                    changes.append(
                        f"Lower max_clob_price {cur_max:.2f}→{new_max:.2f}: "
                        f"bucket {b:.2f} at {wr*100:.0f}% WR (gap {gap*100:+.0f}%, "
                        f"{len(trades)} trades)"
                    )

    if new_min >= new_max - 0.02:
        log.warning("Band calibration would collapse band — skipping")
        return []

    if changes:
        if new_min != cur_min:
            _write_param("min_clob_price", new_min)
        if new_max != cur_max:
            _write_param("max_clob_price", new_max)

    return changes


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    now    = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    log.info("=== monitor run %s ===", now.strftime("%Y-%m-%dT%H:%MZ"))

    ms      = _load_json(_MON_STATE, {})
    state   = _load_json(_REAL_STATE, {})
    settled = [p for p in state.get("closed_positions", [])
               if p.get("realized_pnl") is not None]
    open_p  = state.get("positions", [])

    params      = _load_params()
    max_bet_abs = float(params.get("max_bet_abs", 10.0))

    actions: list[str] = []
    alerts:  list[str] = []

    # ── 1. Health ──────────────────────────────────────────────────────────
    stale_min = _log_stale_minutes()
    running   = _bot_running()

    if not running:
        log.warning("Bot not running — restarting")
        ok = _restart_bot()
        if ok:
            actions.append("Bot was DOWN — restarted OK")
            _send_alert("Bot was DOWN — restarted automatically", title="PM Bot Auto-Fix")
        else:
            alerts.append("CRITICAL: Bot restart FAILED — manual fix needed")
    elif stale_min > LOG_STALE_MINUTES:
        log.warning("Log stale %.0f min — bot hung, restarting", stale_min)
        ok = _restart_bot()
        msg = f"Bot log stale ({stale_min:.0f} min) — restarted {'OK' if ok else 'FAILED'}"
        actions.append(msg)
        _send_alert(msg, title="PM Bot Auto-Fix")
    else:
        log.info("Bot healthy  log_age=%.1f min", stale_min)

    # ── 2. Balance ─────────────────────────────────────────────────────────
    clob_bal  = float(state.get("real_clob_balance", 0))
    available = float(state.get("available_usdc", 0))
    deployed  = sum(float(p.get("size_usdc", 0)) for p in open_p)

    last_clob     = float(ms.get("last_clob", clob_bal))
    balance_drop  = (last_clob - clob_bal) / last_clob if last_clob > 0 else 0.0
    min_safe_bal  = max_bet_abs * MIN_BALANCE_MULT

    log.info("CLOB=$%.2f  available=$%.2f  deployed=$%.2f  min_safe=$%.2f",
             clob_bal, available, deployed, min_safe_bal)

    if clob_bal < min_safe_bal:
        alerts.append(
            f"LOW BALANCE: CLOB=${clob_bal:.2f} < ${min_safe_bal:.0f} "
            f"(= {MIN_BALANCE_MULT:.0f}x max_bet ${max_bet_abs:.0f}) — "
            f"deposit needed or AR missed redemption"
        )
    elif balance_drop > BALANCE_DROP_PCT and last_clob > 5:
        alerts.append(
            f"BALANCE DROP: ${last_clob:.2f} → ${clob_bal:.2f} "
            f"({balance_drop*100:.0f}% drop since last check) — "
            f"investigate losses"
        )

    # ── 3. Win rate + consecutive loss streak ──────────────────────────────
    wr20   = _recent_wr(settled, 20)
    wr10   = _recent_wr(settled, 10)
    streak = _consecutive_loss_streak(settled)
    today  = _today_stats(settled)
    rate   = _pnl_rate_usd_per_hour(settled, window_hours=2.0)

    log.info(
        "WR20: %s  WR10: %s  loss_streak: %d  today: %d trades %.1f%% WR net=$%.2f",
        f"{wr20*100:.1f}%" if wr20 is not None else "n/a",
        f"{wr10*100:.1f}%" if wr10 is not None else "n/a",
        streak,
        today["count"], today["wr"] * 100, today["net"],
    )
    if rate is not None:
        log.info("PnL rate (last 2hr): %+.2f $/hr", rate)

    # Consecutive loss streak alert — most sensitive signal
    if streak >= LOSS_STREAK_ALERT:
        alerts.append(
            f"LOSS STREAK: {streak} consecutive losses\n"
            f"WR10={wr10*100:.0f}% WR20={wr20*100:.0f}%  today net=${today['net']:+.2f}\n"
            f"PnL rate: {rate:+.2f} $/hr\n"
            f"Signal may have broken down — watch closely."
        )

    # WR checks with severity levels
    low_count = int(ms.get("low_wr_count", 0))
    if wr20 is not None:
        if wr20 < WR_CRITICAL:
            # Critical: below 50%, losing money fast
            alerts.append(
                f"CRITICAL WR: last 20 trades at {wr20*100:.0f}% — "
                f"well below break-even (74%)\n"
                f"Today: {today['count']} trades, net ${today['net']:+.2f}\n"
                f"PnL rate: {f'{rate:+.2f} $/hr' if rate is not None else 'n/a'}\n"
                f"Consider pausing manually."
            )
            low_count += 1
        elif wr20 < WR_WARN_THRESHOLD:
            # Warning: below 65%, bleeding slowly
            low_count += 1
            ms["low_wr_count"] = low_count
            log.warning("Low WR: %.1f%% WR10=%.1f%% (check #%d)",
                        wr20 * 100, (wr10 or 0) * 100, low_count)
            # Alert on first check below threshold (was 2 — now immediately)
            alerts.append(
                f"LOW WIN RATE: last 20 trades at {wr20*100:.0f}% WR "
                f"(break-even ~74%)\n"
                f"WR10={wr10*100:.0f}%  streak={streak} losses\n"
                f"Today: {today['count']} trades, net ${today['net']:+.2f}\n"
                f"PnL rate: {f'{rate:+.2f} $/hr' if rate is not None else 'n/a'}"
            )
        else:
            if low_count > 0:
                log.info("WR recovered (%.1f%%) — resetting counter", wr20 * 100)
            ms["low_wr_count"] = 0
            low_count = 0

    ms["low_wr_count"] = low_count

    # ── 4. PnL rate warning (losing money fast right now) ──────────────────
    if rate is not None and rate < -3.0:
        # Losing more than $3/hr — flag even if WR check hasn't triggered
        log.warning("PnL rate very negative: %.2f $/hr", rate)
        if not any("LOW WIN RATE" in a or "CRITICAL" in a or "STREAK" in a
                   for a in alerts):
            alerts.append(
                f"BLEEDING: PnL rate = {rate:+.2f} $/hr (last 2 hours)\n"
                f"WR20={wr20*100:.0f}%  streak={streak}\n"
                f"Today net=${today['net']:+.2f}"
            )

    # ── 5. Stuck open positions ────────────────────────────────────────────
    stuck = _stuck_positions(open_p, now_ts)
    if stuck:
        names = [
            f"{p.get('market_id','?')[:12]}.. ({age:.0f} min)"
            for p, age in stuck
        ]
        log.warning("Stuck positions (%d): %s", len(stuck), names)
        alerts.append(
            f"STUCK POSITIONS ({len(stuck)}): open > {STUCK_POSITION_MINS} min "
            f"(5-min windows should settle quickly)\n"
            + "\n".join(names)
        )

    # ── 6. Daily calibration (once per day after 09:00 UTC) ───────────────
    today_str       = now.date().isoformat()
    last_calibrated = ms.get("last_calibration_date", "")

    if now.hour >= 9 and last_calibrated != today_str and len(settled) >= 50:
        log.info("Daily calibration starting (%d settled trades)", len(settled))
        ms["last_calibration_date"] = today_str
        cal_changes: list[str] = []

        try:
            session_changes = _calibrate_session(settled, params)
            cal_changes.extend(session_changes)
        except Exception as exc:
            log.error("Session calibration failed: %s", exc)

        params = _load_params()
        try:
            band_changes = _calibrate_band(settled, params)
            cal_changes.extend(band_changes)
        except Exception as exc:
            log.error("Band calibration failed: %s", exc)

        if cal_changes:
            ok = _restart_bot()
            msg = (
                "Daily calibration applied:\n"
                + "\n".join(f"  {c}" for c in cal_changes)
                + f"\nBot restarted: {'OK' if ok else 'FAILED'}"
            )
            actions.append(msg)
            _send_alert(msg, title="PM Bot Auto-Calibration")
            log.info("Calibration: %s", cal_changes)
        else:
            log.info("Calibration: no changes needed today")

    # ── 7. Daily report (00:00 UTC) — detailed breakdown ──────────────────
    last_report = ms.get("last_report_date", "")
    if now.hour == 0 and last_report != today_str:
        ms["last_report_date"] = today_str

        wins  = [p for p in settled if p["realized_pnl"] > 0]
        gross = sum(p["realized_pnl"] for p in settled)
        fees  = sum(p.get("size_usdc", 0) * 0.02 for p in settled)
        wr_all = len(wins) / len(settled) * 100 if settled else 0

        dirs  = _all_time_direction_stats(settled)
        sess  = _session_stats(settled)

        # Session table — only hours with enough trades
        sess_lines = []
        for h, s in sorted(sess.items()):
            if s["count"] >= 10:
                bar = "+" if s["net"] >= 0 else "-"
                sess_lines.append(
                    f"  {h:02d}:00Z  {s['count']:3d} trades  "
                    f"{s['wr']*100:.0f}% WR  net ${s['net']:+.2f}  {bar}"
                )

        report = (
            f"=== Daily Report {today_str} ===\n"
            f"All-time: {len(settled)} trades  {wr_all:.1f}% WR\n"
            f"Gross: ${gross:+.2f}  Fees: -${fees:.2f}  Net: ${gross-fees:+.2f}\n"
            f"CLOB balance: ${clob_bal:.2f}\n"
            f"\nToday: {today['count']} trades  {today['wr']*100:.0f}% WR  "
            f"net ${today['net']:+.2f}\n"
            f"  UP:   {today['up_count']} trades  {today['up_wr']*100:.0f}% WR\n"
            f"  DOWN: {today['down_count']} trades  {today['down_wr']*100:.0f}% WR\n"
            f"\nAll-time direction:\n"
            f"  UP:   {dirs['up']['count']} trades  {dirs['up']['wr']*100:.0f}% WR  "
            f"net ${dirs['up']['pnl']:+.2f}\n"
            f"  DOWN: {dirs['down']['count']} trades  {dirs['down']['wr']*100:.0f}% WR  "
            f"net ${dirs['down']['pnl']:+.2f}\n"
            + (f"\nSession breakdown:\n" + "\n".join(sess_lines) if sess_lines else "")
        )
        _send_alert(report, title="PM Bot Daily Report")

    # ── 8. Send alerts ─────────────────────────────────────────────────────
    for alert in alerts:
        _send_alert(alert, title="PM Bot ALERT")
        log.warning("ALERT: %s", alert[:100])

    if actions and not any(a.startswith("Bot") for a in actions):
        _send_alert(
            "Auto-fix applied:\n" + "\n".join(f"  {a}" for a in actions),
            title="PM Bot Auto-Fix",
        )

    # ── 9. Save monitor state ──────────────────────────────────────────────
    ms["last_run"]   = now.isoformat()
    ms["last_clob"]  = clob_bal
    ms["last_wr20"]  = round(wr20, 3) if wr20 is not None else None
    ms["last_streak"] = streak
    _save_json(_MON_STATE, ms)

    log.info("=== monitor run complete | balance=$%.2f wr20=%s streak=%d ===",
             clob_bal,
             f"{wr20*100:.1f}%" if wr20 is not None else "n/a",
             streak)


if __name__ == "__main__":
    main()
