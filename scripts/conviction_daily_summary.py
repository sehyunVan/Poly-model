"""
Conviction strategy daily auto-summary.

Run once per day (cron at 00:30 UTC). Outputs:
  1. Console + file log:  data/conviction_daily_summary.log
  2. ntfy push:           compact regime-health signal

Tracks the three profitable buckets identified by the 2026-05-18 analysis:
  ETH 5m  [0.54-0.55)
  BTC 15m [0.52-0.53)
  SOL 15m [0.53-0.54)

For each bucket, reports:
  - Last 7 days of daily PnL
  - Recent rolling stats (last 100 trades)
  - Lifetime vs last-7-day EV (decay signal)
  - Overall verdict
"""

import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load .env so NTFY_TOPIC is available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/poly-model/.env"))
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
PAPER_STATE = ROOT / "data" / "conviction_state.json"
LIVE_STATE  = ROOT / "data" / "conviction_real_state.json"
LOG_PATH    = ROOT / "data" / "conviction_daily_summary.log"

# The three winning buckets (from 2026-05-18 bucket analysis)
WINNING_BUCKETS = [
    ("ETH", "5m",  0.54, 0.55),
    ("BTC", "15m", 0.52, 0.53),
    ("SOL", "15m", 0.53, 0.54),
]

DAYS_BACK = 7


def _load_closed(path: Path, is_live: bool) -> list:
    """Load closed positions from a state file. Returns [] on missing/corrupt."""
    if not path.exists():
        return []
    try:
        # Snapshot copy so a mid-write file doesn't blow us up
        raw = path.read_text(encoding="utf-8")
        state = json.loads(raw)
    except Exception:
        return []
    closed = state.get("closed_positions", []) or []
    if is_live:
        return [p for p in closed if p.get("is_live", False) or path.name == "conviction_real_state.json"]
    return [p for p in closed if not p.get("is_live", False)]


def _filter_bucket(trades: list, sym: str, tf: str, lo: float, hi: float) -> list:
    out = []
    for p in trades:
        if p.get("symbol") != sym:    continue
        if p.get("timeframe") != tf:  continue
        c = float(p.get("conviction", 0))
        if not (lo <= c < hi):        continue
        out.append(p)
    return out


def _stats(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": None, "pnl": 0.0, "ev": 0.0}
    wins = sum(1 for p in trades if float(p.get("pnl", 0)) > 0)
    pnl  = sum(float(p.get("pnl", 0)) for p in trades)
    return {"n": n, "wins": wins, "wr": 100 * wins / n, "pnl": pnl, "ev": pnl / n}


def _format_stats(s: dict) -> str:
    if s["n"] == 0:
        return "      —"
    wr = f"{s['wr']:5.1f}%" if s["wr"] is not None else "  —  "
    return f"n={s['n']:>4}  WR={wr}  PnL={s['pnl']:>+7.2f}  EV={s['ev']:>+8.4f}"


def _send_ntfy(message: str, title: str = "conviction-daily", priority: str = "3") -> bool:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title":    title,
                "Priority": priority,
                "Tags":     "chart_with_upwards_trend",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


def build_summary() -> tuple[str, str]:
    """Return (full_log_text, compact_ntfy_message)."""
    paper = _load_closed(PAPER_STATE, is_live=False)
    live  = _load_closed(LIVE_STATE,  is_live=True)

    now = datetime.now(timezone.utc)
    today = now.date()
    days = [(today - timedelta(days=i)) for i in range(DAYS_BACK - 1, -1, -1)]

    lines = []
    push  = []
    lines.append("=" * 72)
    lines.append(f"CONVICTION DAILY SUMMARY — {today.isoformat()} UTC")
    lines.append("=" * 72)
    lines.append(f"Paper trades total: {len(paper)}   Live trades total: {len(live)}")
    lines.append("")

    # ── Per-bucket section ───────────────────────────────────────────────────
    decay_flags = []
    bucket_evs  = []

    for sym, tf, lo, hi in WINNING_BUCKETS:
        bkt_paper = _filter_bucket(paper, sym, tf, lo, hi)
        lifetime  = _stats(bkt_paper)

        # Last-7d slice (use closed_at)
        cutoff = now.timestamp() - DAYS_BACK * 86400
        recent_7d = [p for p in bkt_paper if float(p.get("closed_at", 0)) >= cutoff]
        recent7   = _stats(recent_7d)

        # Last 100 by chronological order
        bkt_sorted = sorted(bkt_paper, key=lambda p: p.get("closed_at", 0))
        last100 = _stats(bkt_sorted[-100:])

        # Daily PnL trajectory
        by_day = defaultdict(list)
        for p in bkt_paper:
            d = datetime.fromtimestamp(p.get("closed_at", 0), tz=timezone.utc).date()
            by_day[d].append(p)

        # Direction breakdown over last 7 days
        up_recent   = [p for p in recent_7d if p.get("direction") == "UP"]
        down_recent = [p for p in recent_7d if p.get("direction") == "DOWN"]

        label = f"{sym} {tf} [{lo:.2f}-{hi:.2f})"
        lines.append("─" * 72)
        lines.append(f"BUCKET: {label}")
        lines.append("─" * 72)
        lines.append(f"  Lifetime:    {_format_stats(lifetime)}")
        lines.append(f"  Last 7 days: {_format_stats(recent7)}")
        lines.append(f"  Last 100:    {_format_stats(last100)}")
        lines.append("")
        lines.append("  Daily PnL trajectory (last 7d):")
        lines.append(f"    {'date':<12} {'n':>4}  {'WR':>6}  {'PnL':>8}  {'EV/tr':>8}")
        for d in days:
            day_trades = by_day.get(d, [])
            st = _stats(day_trades)
            if st["n"] == 0:
                lines.append(f"    {str(d):<12} {0:>4}  {'—':>6}  {'—':>8}  {'—':>8}")
            else:
                lines.append(f"    {str(d):<12} {st['n']:>4}  {st['wr']:>5.1f}%  {st['pnl']:>+8.2f}  {st['ev']:>+8.4f}")
        lines.append("")
        lines.append("  Last-7d direction breakdown:")
        for dirn, sub in [("UP", up_recent), ("DOWN", down_recent)]:
            st = _stats(sub)
            if st["n"] == 0:
                lines.append(f"    {dirn:<5} —")
            else:
                lines.append(f"    {dirn:<5} {_format_stats(st)}")
        lines.append("")

        # Decay signal: last-7d EV vs lifetime EV
        decay = recent7["ev"] < 0 and lifetime["ev"] > 0
        decay_flags.append((label, decay, recent7["ev"], lifetime["ev"]))
        bucket_evs.append((label, recent7["ev"], recent7["pnl"], recent7["n"]))

        # Compact push line
        emoji = "🔻" if decay else ("📉" if recent7["ev"] < 0 else "📈")
        push.append(f"{emoji} {label}: 7d EV={recent7['ev']:+.4f} ({recent7['n']}tr, ${recent7['pnl']:+.2f})")

    # ── Verdict ──────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("VERDICT")
    lines.append("=" * 72)
    n_decay = sum(1 for _, d, _, _ in decay_flags if d)
    n_neg   = sum(1 for _, ev, _, _ in bucket_evs if ev < 0)
    total_7d_pnl = sum(p for _, _, p, _ in bucket_evs)

    if n_decay == 3:
        verdict = "❌ ALL THREE BUCKETS DECAYING — strategy regime has shifted, do NOT re-enable live"
        prio = "4"
    elif n_neg == 3:
        verdict = "⚠️  All three buckets net-negative last 7 days — keep live halted, watch closely"
        prio = "3"
    elif n_neg >= 2:
        verdict = "⚠️  Majority of buckets net-negative last 7 days — keep live halted"
        prio = "3"
    elif n_neg == 1:
        verdict = "🟡 1/3 buckets net-negative — could re-enable the 2 winning buckets only"
        prio = "3"
    else:
        verdict = "✅ All three buckets net-positive last 7 days — safe to re-enable live"
        prio = "2"

    lines.append(verdict)
    lines.append(f"Combined 7-day PnL across 3 buckets: ${total_7d_pnl:+.2f}")
    lines.append("")

    full_text = "\n".join(lines)

    # ── Compact ntfy message ─────────────────────────────────────────────────
    push_msg = (
        f"Combined 7d PnL: ${total_7d_pnl:+.2f}\n"
        + "\n".join(push)
        + f"\n\n{verdict}"
    )
    return full_text, push_msg, prio


def main():
    try:
        full, push, prio = build_summary()
    except Exception as e:
        err = f"conviction_daily_summary failed: {type(e).__name__}: {e}"
        print(err)
        _send_ntfy(err, title="conviction-daily-ERROR", priority="4")
        sys.exit(1)

    # Print to stdout (cron captures into log)
    print(full)

    # Append to persistent log (newest at the bottom)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(full + "\n\n")
    except Exception as e:
        print(f"warning: could not write to {LOG_PATH}: {e}")

    # Push
    _send_ntfy(push, title="conviction-daily", priority=prio)


if __name__ == "__main__":
    main()
