"""
Portfolio exposure limit checker.

check_exposure_limits() is the gating function that must return True
before any order is submitted.  It runs four independent checks in order;
the first failure short-circuits the rest.
"""

from __future__ import annotations

from signal_layer.schemas import TradeSignal

from .schemas import Portfolio, RiskLimits


def check_exposure_limits(
    candidate: TradeSignal,
    raw_size: float,
    portfolio: Portfolio,
    limits: RiskLimits,
    category: str = "other",
    group_id: str | None = None,
) -> tuple[bool, str]:
    """
    Gate a candidate order against all portfolio exposure rules.

    Checks (in order — first failure returns immediately):
        1. Single-event limit  : existing market exposure + raw_size
                                 <= max_single_event_pct * total_capital
        2. Category limit      : category total + raw_size
                                 <= max_category_pct[category] * total_capital
        3. Correlated-group    : group total + raw_size
                                 <= max_correlated_group_pct * total_capital
                                 (only when group_id is not None)
        4. Daily-loss gate     : if daily_pnl already breached the daily loss
                                 limit, block all new entries

    Args:
        candidate:  TradeSignal driving the proposed order.
        raw_size:   Proposed USDC order size (before any further capping).
        portfolio:  Current portfolio snapshot.
        limits:     Active RiskLimits instance.
        category:   Market category of the candidate trade.
        group_id:   Correlated-group identifier (None = no group check).

    Returns:
        (allowed: bool, reason: str)
        reason is empty when allowed=True.
    """
    cap = portfolio.total_capital

    # ------------------------------------------------------------------
    # 1. Single-event concentration
    # ------------------------------------------------------------------
    existing_event = portfolio.exposure_for_market(candidate.market_id)
    max_event = limits.max_single_event_pct * cap
    if existing_event + raw_size > max_event:
        return False, (
            f"single-event limit: current {existing_event:.2f} + "
            f"new {raw_size:.2f} = {existing_event + raw_size:.2f} "
            f"exceeds {max_event:.2f} "
            f"({limits.max_single_event_pct:.0%} of {cap:.2f})"
        )

    # ------------------------------------------------------------------
    # 2. Category concentration
    # ------------------------------------------------------------------
    existing_cat = portfolio.exposure_for_category(category)
    max_cat = limits.category_limit(category) * cap
    if existing_cat + raw_size > max_cat:
        return False, (
            f"category '{category}' limit: current {existing_cat:.2f} + "
            f"new {raw_size:.2f} = {existing_cat + raw_size:.2f} "
            f"exceeds {max_cat:.2f} "
            f"({limits.category_limit(category):.0%} of {cap:.2f})"
        )

    # ------------------------------------------------------------------
    # 3. Correlated-group concentration
    # ------------------------------------------------------------------
    if group_id is not None:
        existing_grp = portfolio.exposure_for_group(group_id)
        max_grp = limits.max_correlated_group_pct * cap
        if existing_grp + raw_size > max_grp:
            return False, (
                f"correlated group '{group_id}' limit: current {existing_grp:.2f} + "
                f"new {raw_size:.2f} = {existing_grp + raw_size:.2f} "
                f"exceeds {max_grp:.2f} "
                f"({limits.max_correlated_group_pct:.0%} of {cap:.2f})"
            )

    # ------------------------------------------------------------------
    # 4. Daily-loss gate — no new entries once limit is breached
    # ------------------------------------------------------------------
    daily_loss_floor = -limits.max_daily_loss_pct * cap
    if portfolio.daily_pnl <= daily_loss_floor:
        return False, (
            f"daily loss gate: pnl {portfolio.daily_pnl:.2f} has reached "
            f"the floor {daily_loss_floor:.2f} "
            f"({limits.max_daily_loss_pct:.0%} of {cap:.2f}) — "
            f"new entries are blocked"
        )

    return True, ""
