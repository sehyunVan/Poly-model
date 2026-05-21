"""
monitoring/alerts.py — real-time alert dispatcher.

Reads credentials from environment variables at call time so that .env
changes take effect without restarting the process.

Supported channels:
    1. ntfy.sh   — NTFY_TOPIC  (recommended — works from any server IP)
    2. Telegram  — TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    3. Discord   — DISCORD_WEBHOOK_URL
                   NOTE: Discord webhooks are blocked by Cloudflare from Oracle
                   Cloud / AWS datacenter IPs (error 1010). Use ntfy.sh instead,
                   or relay via Make.com (see docs/discord_setup.md).
    4. Log only  — always active as fallback

Usage::

    from monitoring.alerts import send_alert, send_daily_summary, send_gate_check

    send_alert("halt_new_trades triggered — daily loss limit reached", level="CRITICAL")
    send_daily_summary(virtual_portfolio)
    send_gate_check(virtual_portfolio)

Levels (mapped to emoji prefixes):
    DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

from __future__ import annotations

import json as _json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual.portfolio import VirtualPortfolio

_log = logging.getLogger("monitoring.alerts")

_LEVEL_EMOJI: dict[str, str] = {
    "DEBUG":    "🔍",
    "INFO":     "ℹ️",
    "WARNING":  "⚠️",
    "ERROR":    "❌",
    "CRITICAL": "🚨",
}

# Discord embed colors
_COLOR_GREEN  = 3066993
_COLOR_YELLOW = 16776960
_COLOR_RED    = 15158332
_COLOR_BLUE   = 3447003
_COLOR_GREY   = 9807270


# ── Core dispatcher ────────────────────────────────────────────────────────────

def send_alert(message: str, level: str = "INFO") -> None:
    """
    Dispatch *message* to all configured alert channels.

    Parameters
    ----------
    message : str
        Human-readable notification text.
    level : str
        Severity label — DEBUG | INFO | WARNING | ERROR | CRITICAL.
        Used as a prefix in the outgoing message and as the log level.

    Behaviour
    ---------
    - Never raises an exception; delivery failures are logged at WARNING.
    - If neither Telegram nor Discord is configured, the message is only
      written to the application log (graceful degradation).
    - Credentials are read from the environment on every call so that
      hotswapping .env values works without a restart.
    """
    level_upper = level.upper()
    emoji       = _LEVEL_EMOJI.get(level_upper, "📢")
    now_str     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    full_msg    = f"{emoji} [{level_upper}] {now_str}\n{message}"

    # Always log
    log_level = getattr(logging, level_upper, logging.INFO)
    _log.log(log_level, "ALERT: %s", message)

    sent = False

    # ── ntfy.sh ───────────────────────────────────────────────────────────────
    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if ntfy_topic:
        sent = _send_ntfy(ntfy_topic, message, level_upper) or sent

    # ── Telegram ──────────────────────────────────────────────────────────────
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        sent = _send_telegram(token, chat_id, full_msg) or sent

    # ── Discord ───────────────────────────────────────────────────────────────
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if webhook:
        sent = _send_discord(webhook, full_msg) or sent

    if not sent:
        _log.debug("No alert channels configured — message logged only.")


# ── ntfy.sh sender ────────────────────────────────────────────────────────────

def _send_ntfy(topic: str, message: str, level: str = "INFO") -> bool:
    """
    POST a notification to https://ntfy.sh/{topic}.

    Priority mapping:  CRITICAL→5(urgent)  ERROR→4(high)
                       WARNING→3(default)  INFO/DEBUG→2(low)
    """
    try:
        import urllib.request

        priority_map = {"CRITICAL": "5", "ERROR": "4", "WARNING": "3",
                        "INFO": "2", "DEBUG": "1"}
        priority = priority_map.get(level.upper(), "3")

        tag_map = {"CRITICAL": "rotating_light", "ERROR": "x",
                   "WARNING": "warning", "INFO": "white_check_mark", "DEBUG": "mag"}
        tag = tag_map.get(level.upper(), "bell")

        title_map = {"CRITICAL": "CRITICAL", "ERROR": "ERROR",
                     "WARNING": "WARNING", "INFO": "INFO", "DEBUG": "DEBUG"}
        title = title_map.get(level.upper(), "ALERT")

        url = f"https://ntfy.sh/{topic}"
        payload = message[:4096].encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Title":    title,          # ASCII only — HTTP header restriction
                "Priority": priority,
                "Tags":     tag,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 204):
                return True
            _log.warning("ntfy returned HTTP %s", resp.status)
            return False
    except Exception as exc:
        _log.warning("ntfy alert failed: %s", exc)
        return False


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_performance_metrics(vp: "VirtualPortfolio") -> dict:
    """
    Compute Brier score, per-category hit rate, and max drawdown from a
    VirtualPortfolio's closed positions.

    Brier score uses fill_price as a proxy for the model-predicted probability
    (the true P_R is not currently stored in VirtualPosition).  This measures
    how well-calibrated the *market price* was as a direction predictor.
    A score of 0.25 corresponds to random guessing; lower is better.

    Returns:
        {
            "brier_score":       float | None,   # None if fewer than 5 observations
            "brier_n":           int,
            "category_hit_rates": {cat: float},  # win-rate per category
            "category_counts":    {cat: int},
            "max_drawdown_usdc":  float,
        }
    """
    closed = vp.closed_positions
    if not closed:
        return {
            "brier_score": None, "brier_n": 0,
            "category_hit_rates": {}, "category_counts": {},
            "max_drawdown_usdc": 0.0,
        }

    # --- Brier score ---------------------------------------------------------
    bs_sum = 0.0
    bs_n   = 0
    for p in closed:
        if p.outcome is None or p.fill_price is None:
            continue
        # For a YES bet the fill_price approximates P(YES).
        # For a NO bet  the fill_price approximates P(NO) directly.
        p_pred = p.fill_price
        # did_win = 1 if the position profited, 0 otherwise
        did_win = 1 if (p.realized_pnl or 0.0) > 0 else 0
        bs_sum += (p_pred - did_win) ** 2
        bs_n   += 1

    brier = round(bs_sum / bs_n, 4) if bs_n >= 5 else None

    # --- Per-category hit rate -----------------------------------------------
    from collections import defaultdict
    cat_wins:  dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    for p in closed:
        cat = p.category or "other"
        cat_total[cat] += 1
        if (p.realized_pnl or 0.0) > 0:
            cat_wins[cat] += 1

    cat_hr     = {c: cat_wins[c] / cat_total[c] for c in cat_total}
    cat_counts = dict(cat_total)

    # --- Max drawdown (from realised PnL curve) ------------------------------
    sorted_closed = sorted(
        [p for p in closed if p.fill_time is not None],
        key=lambda p: p.fill_time,
    )
    running = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for p in sorted_closed:
        running += p.realized_pnl or 0.0
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    return {
        "brier_score":        brier,
        "brier_n":            bs_n,
        "category_hit_rates": cat_hr,
        "category_counts":    cat_counts,
        "max_drawdown_usdc":  round(max_dd, 2),
    }


# ── Daily summary (Discord embed / ntfy) ──────────────────────────────────────

def _load_crypto_stats() -> dict:
    """
    Read crypto_cache.jsonl and virtual_state.json to build crypto-specific stats.
    Safe — returns zeros on any error.
    """
    import json as _j
    from pathlib import Path as _P

    stats = {
        "labeled_rows": 0, "total_rows": 0,
        "crypto_pnl": 0.0, "crypto_wins": 0, "crypto_losses": 0,
        "crypto_open": 0, "crypto_settled": 0, "crypto_hit_rate": 0.0,
    }
    try:
        cache = _P(__file__).resolve().parent.parent.parent / "data" / "crypto_cache.jsonl"
        if cache.exists():
            rows = [_j.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
            stats["total_rows"]   = len(rows)
            stats["labeled_rows"] = sum(1 for r in rows if r.get("label") is not None)
    except Exception:
        pass
    try:
        state = _P(__file__).resolve().parent.parent.parent / "data" / "virtual_state.json"
        if state.exists():
            d  = _j.loads(state.read_text(encoding="utf-8"))
            cc = [p for p in d.get("closed_positions", []) if p.get("category") == "crypto"]
            co = [p for p in d.get("positions", [])        if p.get("category") == "crypto"]
            wins   = sum(1 for p in cc if (p.get("realized_pnl") or 0) > 0)
            losses = sum(1 for p in cc if (p.get("realized_pnl") or 0) <= 0)
            pnl    = sum(p.get("realized_pnl") or 0 for p in cc)
            stats.update({
                "crypto_pnl":      round(pnl, 2),
                "crypto_wins":     wins,
                "crypto_losses":   losses,
                "crypto_open":     len(co),
                "crypto_settled":  len(cc),
                "crypto_hit_rate": wins / len(cc) * 100 if cc else 0.0,
            })
    except Exception:
        pass
    return stats


def send_daily_summary(vp: "VirtualPortfolio") -> None:
    """
    Send a short daily report to Discord (embed) and/or ntfy.sh.
    Includes both general portfolio stats and crypto up/down specific section.
    Called once per midnight cycle.
    """
    webhook    = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if not webhook and not ntfy_topic:
        _log.debug("No notification channels configured — daily summary skipped.")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    closed      = vp.closed_positions
    positions   = vp.positions
    actual_cumulative = sum((p.realized_pnl or 0.0) for p in closed)
    pct         = actual_cumulative / vp.initial_budget * 100 if vp.initial_budget else 0
    wins        = sum(1 for p in closed if (p.realized_pnl or 0) > 0)
    losses      = sum(1 for p in closed if (p.realized_pnl or 0) < 0)
    hit_rate    = wins / len(closed) * 100 if closed else 0
    unique_open = len(set(p.market_id for p in positions))
    daily_pnl   = vp.pnl_history[-1]["pnl"] if vp.pnl_history else 0.0

    cs    = _load_crypto_stats()
    color = _COLOR_GREEN if actual_cumulative >= 0 else _COLOR_RED
    sign  = "+" if actual_cumulative >= 0 else ""
    dsign = "+" if daily_pnl >= 0 else ""
    csign = "+" if cs["crypto_pnl"] >= 0 else ""

    crypto_hr_str = (
        f"  HR: {cs['crypto_hit_rate']:.0f}%" if cs["crypto_settled"] else ""
    )
    crypto_field = (
        f"PnL: **{csign}${cs['crypto_pnl']:.2f}**  "
        f"({cs['crypto_wins']}W/{cs['crypto_losses']}L{crypto_hr_str})\n"
        f"Open: **{cs['crypto_open']}**  |  Settled: **{cs['crypto_settled']}**\n"
        f"Training rows: **{cs['labeled_rows']}** labeled / {cs['total_rows']} total"
    )

    embed = {
        "title": f"Daily Report — {today}",
        "color": color,
        "fields": [
            {
                "name": "Capital",
                "value": (
                    f"Budget: **${vp.initial_budget:.0f}**\n"
                    f"Available: **${vp.available_usdc:.2f}**\n"
                    f"In positions: **${sum(p.size_usdc for p in positions):.2f}**"
                ),
                "inline": True,
            },
            {
                "name": "Overall PnL",
                "value": (
                    f"Today: **{dsign}${daily_pnl:.2f}**\n"
                    f"Cumulative: **{sign}${actual_cumulative:.2f}** ({sign}{pct:.1f}%)\n"
                    f"Hit rate: **{hit_rate:.0f}%** ({wins}W/{losses}L)"
                ),
                "inline": True,
            },
            {
                "name": "Positions",
                "value": (
                    f"Open: **{len(positions)}** fills / **{unique_open}** markets\n"
                    f"Settled all-time: **{len(closed)}**"
                ),
                "inline": True,
            },
            {
                "name": "Crypto Up/Down Bot (5-min)",
                "value": crypto_field,
                "inline": False,
            },
        ],
        "footer": {"text": "Polymarket virtual bot — paper trading"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if webhook:
        _log.info("Sending daily summary to Discord.")
        _send_discord_embed(webhook, embed)

    if ntfy_topic:
        ntfy_text = (
            f"[General] Avail ${vp.available_usdc:.2f} | "
            f"Cum {sign}${actual_cumulative:.2f} ({sign}{pct:.1f}%) | "
            f"HR {hit_rate:.0f}% ({wins}W/{losses}L)\n"
            f"[Crypto 5m] PnL {csign}${cs['crypto_pnl']:.2f} | "
            f"{cs['crypto_wins']}W/{cs['crypto_losses']}L{crypto_hr_str} | "
            f"Open {cs['crypto_open']} | "
            f"Rows {cs['labeled_rows']}/{cs['total_rows']}"
        )
        _send_ntfy(ntfy_topic, ntfy_text, "INFO")


# ── Gate status (Discord embed) ────────────────────────────────────────────────

def send_gate_check(vp: "VirtualPortfolio") -> None:
    """
    Compute the TASK-15 go-live gate status and send to Discord and/or ntfy.sh.
    Called once per midnight cycle.
    """
    webhook    = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if not webhook and not ntfy_topic:
        _log.debug("No notification channels configured — gate check skipped.")
        return

    from datetime import date as _date

    closed    = vp.closed_positions
    positions = vp.positions

    # Gate 1 — data sufficiency
    start_str   = getattr(vp, "start_date", None)
    days_running = 0
    if start_str:
        try:
            start_dt    = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
            days_running = (datetime.now(timezone.utc) - start_dt).days
        except Exception:
            pass

    unique_settled = len(set(p.market_id for p in closed))
    g1_days   = days_running >= 30
    g1_mkts   = unique_settled >= 30
    g1_pass   = g1_days and g1_mkts

    # Compute richer metrics
    perf = compute_performance_metrics(vp)

    # Gate 2 — model quality
    wins      = sum(1 for p in closed if (p.realized_pnl or 0) > 0)
    hit_rate  = wins / len(closed) if closed else 0
    g2_hits   = len(closed) >= 20
    g2_hr     = hit_rate >= 0.52 if g2_hits else False
    brier     = perf["brier_score"]
    g2_brier  = (brier is not None and brier < 0.25) if brier is not None else False
    g2_pass   = g2_hits and g2_hr

    # Gate 3 — capital safety
    actual_cumulative = sum((p.realized_pnl or 0.0) for p in closed)
    pnl_pct   = actual_cumulative / vp.initial_budget if vp.initial_budget else 0
    g3_pnl    = actual_cumulative > 0
    mdd_pct   = perf["max_drawdown_usdc"] / vp.initial_budget if vp.initial_budget else 0
    g3_mdd    = mdd_pct < 0.15
    g3_pass   = g3_pnl and g3_mdd

    # Gate 4 — model characterisation (manual gate, can't auto-check)
    g4_pass   = False

    def _tick(ok: bool) -> str:
        return "✅" if ok else "❌"

    all_pass  = g1_pass and g2_pass and g3_pass and g4_pass
    color     = _COLOR_GREEN if all_pass else (_COLOR_YELLOW if (g3_pass) else _COLOR_RED)
    title     = "🚀 READY FOR LIVE TRADING" if all_pass else "🔬 Go-Live Gate Status"

    # Estimate next milestone
    today_mo = _date.today().month
    if days_running < 30 or unique_settled < 5:
        next_milestone = "March 31 — end-of-month market settlements"
    elif unique_settled < 30:
        next_milestone = "Keep running — need more settled markets"
    else:
        next_milestone = "Run Gate 4 analysis manually"

    embed = {
        "title": title,
        "color": color,
        "fields": [
            {
                "name": f"{_tick(g1_pass)} Gate 1 — Data Sufficiency",
                "value": (
                    f"Days running: **{days_running}** / 30  {_tick(g1_days)}\n"
                    f"Settled markets: **{unique_settled}** / 30  {_tick(g1_mkts)}"
                ),
                "inline": False,
            },
            {
                "name": f"{_tick(g2_pass)} Gate 2 — Model Accuracy",
                "value": (
                    f"Settled positions: **{len(closed)}** / 20  {_tick(g2_hits)}\n"
                    f"Hit rate: **{hit_rate:.0%}** (need >=52%)  {_tick(g2_hr)}\n"
                    f"Brier score: **{f'{brier:.4f}' if brier is not None else 'n/a (need 5+ obs)'}** "
                    f"(need <0.25, random=0.25)  {_tick(g2_brier)}\n"
                    f"Calibration: need real P_R storage (fill_price proxy used)"
                ),
                "inline": False,
            },
            {
                "name": f"{_tick(g3_pass)} Gate 3 — Capital Safety",
                "value": (
                    f"Cumulative PnL: **{'+' if actual_cumulative >= 0 else ''}${actual_cumulative:.2f}**"
                    f" ({pnl_pct:+.1%})  {_tick(g3_pnl)}\n"
                    f"Max Drawdown: **${perf['max_drawdown_usdc']:.2f}** ({mdd_pct:.1%}) "
                    f"(need <15%)  {_tick(g3_mdd)}"
                ),
                "inline": False,
            },
            {
                "name": f"{_tick(g4_pass)} Gate 4 — Model Characterisation",
                "value": (
                    "Category hit rates:\n"
                    + "\n".join(
                        f"  {cat}: **{perf['category_hit_rates'].get(cat, 0):.0%}** "
                        f"({perf['category_counts'].get(cat, 0)} trades)"
                        for cat in ["politics", "crypto", "sports", "other"]
                        if perf["category_counts"].get(cat, 0) > 0
                    ) or "  No settled trades yet"
                    + "\nAlpha validation: **manual check needed**"
                ),
                "inline": False,
            },
            {
                "name": "📅 Next Milestone",
                "value": next_milestone,
                "inline": False,
            },
        ],
        "footer": {"text": "All 4 gates must pass before switching to live trading"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Discord embed
    if webhook:
        _log.info("Sending gate check to Discord.")
        _send_discord_embed(webhook, embed)

    # ntfy.sh plain-text gate summary
    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if ntfy_topic:
        g_status = "READY" if all_pass else f"{sum([g1_pass,g2_pass,g3_pass,g4_pass])}/4 gates pass"
        ntfy_text = (
            f"{_tick(g1_pass)} Gate1: {days_running}d/{unique_settled}mkt  "
            f"{_tick(g2_pass)} Gate2: HR={hit_rate:.0%}({len(closed)}pos)  "
            f"{_tick(g3_pass)} Gate3: PnL={actual_cumulative:+.2f} MDD={mdd_pct:.1%}  "
            f"{_tick(g4_pass)} Gate4: manual\n"
            f"Status: {g_status} | Next: {next_milestone}"
        )
        level = "INFO" if all_pass else "WARNING" if g3_pass else "WARNING"
        _send_ntfy(ntfy_topic, ntfy_text, level)


# ── Low-level senders ──────────────────────────────────────────────────────────

def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    """
    POST a message to the Telegram Bot API.
    Returns True on HTTP 200, False otherwise.
    """
    try:
        import urllib.request
        import urllib.parse

        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = _json.dumps({
            "chat_id":    chat_id,
            "text":       text[:4096],
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
            _log.warning("Telegram returned HTTP %s", resp.status)
            return False
    except Exception as exc:
        _log.warning("Telegram alert failed: %s", exc)
        return False


def _send_discord(webhook_url: str, text: str) -> bool:
    """
    POST a plain-text message to a Discord webhook.
    Returns True on HTTP 204 (or 200), False otherwise.
    """
    try:
        import urllib.request

        payload = _json.dumps({"content": text[:2000]}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 204):
                return True
            _log.warning("Discord returned HTTP %s", resp.status)
            return False
    except Exception as exc:
        _log.warning("Discord alert failed: %s", exc)
        return False


def _send_discord_embed(webhook_url: str, embed: dict) -> bool:
    """
    POST a rich embed to a Discord webhook.
    Returns True on success, False otherwise.
    """
    try:
        import urllib.request

        payload = _json.dumps({"embeds": [embed]}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 204):
                return True
            _log.warning("Discord embed returned HTTP %s", resp.status)
            return False
    except Exception as exc:
        _log.warning("Discord embed failed: %s", exc)
        return False
