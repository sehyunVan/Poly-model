"""
Global portfolio risk evaluation and deleveraging.

evaluate_portfolio_risk() scans the current portfolio and PnL history
and returns a list of concrete actions the execution layer should carry out.

Action schema:
    {"action": str, "reason": str, ...extra fields...}

Action types:
    halt_new_trades    — stop opening any new positions
    resume_trading     — lift a previous halt (all limits back in range)
    reduce_position    — trim a specific position to target_size
    close_position     — fully close a specific position (target_size=0)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .schemas import Portfolio, RiskLimits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pnl_over_days(pnl_history: list[dict], days: int) -> float:
    """
    Sum PnL entries from the last `days` calendar days (UTC).

    Each entry in pnl_history must have:
        {"date": datetime | str,  "pnl": float}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total  = 0.0
    for entry in pnl_history:
        raw = entry.get("date")
        if isinstance(raw, str):
            try:
                raw = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        if isinstance(raw, datetime):
            ts = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                total += float(entry.get("pnl", 0.0))
    return total


def _action(action: str, reason: str, **kwargs: Any) -> dict:
    return {"action": action, "reason": reason, **kwargs}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_portfolio_risk(
    portfolio: Portfolio,
    pnl_history: list[dict],
    limits: RiskLimits,
) -> list[dict]:
    """
    Evaluate global risk and return the list of corrective actions required.

    Rules (all evaluated; multiple actions may be returned):

        Rule 1 — Daily loss limit
            daily_pnl <= -(max_daily_loss_pct * total_capital)
            → halt_new_trades + close all positions (target_size=0)

        Rule 2 — Weekly loss limit
            7-day rolling PnL <= -(max_weekly_loss_pct * total_capital)
            → halt_new_trades + close all positions

        Rule 3 — Concentration violations
            Any position whose market's total exposure exceeds
            max_single_event_pct * total_capital
            → reduce_position to the limit boundary

        (If no rules are triggered, returns a single "resume_trading" action
        to signal that trading may proceed normally.)

    Args:
        portfolio:    Current portfolio snapshot.
        pnl_history:  List of daily {"date": ..., "pnl": float} dicts.
        limits:       Active RiskLimits instance.

    Returns:
        List of action dicts.  Empty list should never occur; at minimum
        a "resume_trading" action is returned.

    Example:
        actions = evaluate_portfolio_risk(portfolio, history, limits)
        for a in actions:
            if a["action"] == "halt_new_trades":
                trading_enabled = False
            elif a["action"] == "close_position":
                execution.close(a["market_id"])
    """
    actions: list[dict] = []
    cap     = portfolio.total_capital
    halted  = False

    # ------------------------------------------------------------------
    # Rule 1 — Daily loss limit
    # ------------------------------------------------------------------
    daily_floor = -limits.max_daily_loss_pct * cap
    if portfolio.daily_pnl <= daily_floor:
        actions.append(_action(
            "halt_new_trades",
            f"daily PnL {portfolio.daily_pnl:.2f} reached floor "
            f"{daily_floor:.2f} ({limits.max_daily_loss_pct:.0%} of {cap:.2f})",
        ))
        halted = True
        for pos in portfolio.positions:
            actions.append(_action(
                "close_position",
                "daily loss limit — full deleverage",
                market_id=pos.market_id,
                token_id=pos.token_id,
                target_size=0.0,
            ))

    # ------------------------------------------------------------------
    # Rule 2 — Weekly loss limit
    # ------------------------------------------------------------------
    weekly_pnl   = _pnl_over_days(pnl_history, days=7)
    weekly_floor = -limits.max_weekly_loss_pct * cap
    if weekly_pnl <= weekly_floor:
        if not halted:
            actions.append(_action(
                "halt_new_trades",
                f"7-day rolling PnL {weekly_pnl:.2f} reached floor "
                f"{weekly_floor:.2f} ({limits.max_weekly_loss_pct:.0%} of {cap:.2f})",
            ))
            halted = True
            for pos in portfolio.positions:
                actions.append(_action(
                    "close_position",
                    "weekly loss limit — full deleverage",
                    market_id=pos.market_id,
                    token_id=pos.token_id,
                    target_size=0.0,
                ))

    # ------------------------------------------------------------------
    # Rule 3 — Single-event concentration violations
    #   (Only checked if we haven't already told everything to close)
    # ------------------------------------------------------------------
    if not halted:
        max_event = limits.max_single_event_pct * cap
        # Group positions by market_id to get per-market total exposure
        market_totals: dict[str, float] = {}
        for pos in portfolio.positions:
            market_totals[pos.market_id] = market_totals.get(pos.market_id, 0.0) + pos.size

        for market_id, total in market_totals.items():
            if total > max_event:
                actions.append(_action(
                    "reduce_position",
                    f"market {market_id} exposure {total:.2f} exceeds "
                    f"single-event limit {max_event:.2f} "
                    f"({limits.max_single_event_pct:.0%} of {cap:.2f})",
                    market_id=market_id,
                    current_size=total,
                    target_size=max_event,
                ))

    # ------------------------------------------------------------------
    # No issues found
    # ------------------------------------------------------------------
    if not actions:
        actions.append(_action("resume_trading", "all risk limits within bounds"))

    return actions
