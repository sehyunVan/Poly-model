"""
Kelly-based position sizing.

Full Kelly fraction:
    f* = (P_R * b - (1 - P_R)) / b

where b = (1 - P_M) / P_M is the decimal odds for a YES bet
(amount won per unit staked if YES resolves).

For a NO bet, mirror the probabilities:
    P_R_no = 1 - P_R,  P_M_no = 1 - P_M,  b_no = P_M / (1 - P_M)

Actual fraction applied:
    f = max(f*, 0) * kelly_fraction          (never bet negative Kelly)

Final size:
    raw_size  = f * capital
    capped    = min(raw_size, max_bet_pct * capital)
    final     = capped * base_size_factor    (scales with signal strength)
"""

from __future__ import annotations

from .schemas import RiskLimits


def _kelly_fraction(win_prob: float, odds_b: float, kelly_fraction: float) -> float:
    """
    Compute the fractional-Kelly bet fraction (never negative).

    Args:
        win_prob:      Estimated probability of winning (0–1).
        odds_b:        Decimal odds = amount_won / amount_staked.
        kelly_fraction: Scaling factor (0 < kelly_fraction <= 1).

    Returns:
        Non-negative fraction of capital to bet.
    """
    if odds_b <= 0:
        return 0.0
    full_kelly = (win_prob * odds_b - (1.0 - win_prob)) / odds_b
    return max(0.0, full_kelly) * kelly_fraction


def compute_position_size(
    alpha: float,
    P_R: float,
    P_M: float,
    capital: float,
    limits: RiskLimits,
    base_size_factor: float = 1.0,
    volume_24h: float = 0.0,
) -> float:
    """
    Translate a trade signal into a USDC order size.

    Processing pipeline:
        1. Determine direction from sign of alpha (YES or NO bet).
        2. Compute full-Kelly fraction f*.
        3. Scale down: f = max(f*, 0) * kelly_fraction.
        4. Raw size   = f * capital.
        5. Hard cap   = min(raw_size, max_bet_pct * capital).
        6. Final size = hard_cap * base_size_factor * efficiency_discount.

    efficiency_discount: scales from 1.0 (thin market, full edge) to 0.0
    (thick market >50k USDC, professionals dominate — bet nothing).
    The logic: our edge (LLM vs casual crowd) only exists in inefficient markets.

    Args:
        alpha:            P_R - P_M (signed alpha from signal layer).
        P_R:              Model-estimated probability of YES resolving.
        P_M:              Market-implied probability of YES (current price).
        capital:          Total available capital in USDC.
        limits:           Active RiskLimits instance.
        base_size_factor: Scaling factor from TradeSignal [0, 1].
        volume_24h:       24-hour USDC traded volume (0 = unknown, no discount).

    Returns:
        USDC order size >= 0.  Returns 0 when Kelly edge is negative or
        inputs are degenerate.

    Examples:
        # Strong YES edge: alpha=0.15, P_R=0.75, P_M=0.60
        size = compute_position_size(0.15, 0.75, 0.60, 10000, limits)

        # NO edge: alpha=-0.15 (P_R=0.45, P_M=0.60)
        size = compute_position_size(-0.15, 0.45, 0.60, 10000, limits)
    """
    import math as _math
    # Reject degenerate or non-finite inputs — never bet on NaN/Inf probabilities
    if not _math.isfinite(alpha) or not _math.isfinite(P_R) or not _math.isfinite(P_M):
        return 0.0
    if not (0.0 <= P_R <= 1.0):
        return 0.0
    if capital <= 0 or P_M <= 0 or P_M >= 1:
        return 0.0

    # Determine win probability and odds from the perspective of the bet direction
    if alpha >= 0:
        # BUY YES:  win if YES resolves
        win_prob = P_R
        odds_b   = (1.0 - P_M) / P_M
    else:
        # BUY NO:   win if NO resolves  → mirror probabilities
        win_prob = 1.0 - P_R
        P_M_no   = 1.0 - P_M
        if P_M_no <= 0 or P_M_no >= 1:
            return 0.0
        odds_b   = P_M / P_M_no          # = (1 - P_M_no) / P_M_no

    f = _kelly_fraction(win_prob, odds_b, limits.kelly_fraction)
    raw_size  = f * capital
    max_size  = limits.max_bet_pct * capital
    capped    = min(raw_size, max_size)

    # Market efficiency discount: reduce bet size in thick, efficient markets.
    # At volume=0 (unknown): no discount (efficiency_discount=1.0).
    # At volume=50k USDC: discount reaches 0 (fully efficient, no edge).
    _MAX_EFFICIENT_VOL = 50_000.0
    if volume_24h > 0:
        efficiency_discount = max(0.0, 1.0 - volume_24h / _MAX_EFFICIENT_VOL)
    else:
        efficiency_discount = 1.0

    final = capped * max(0.0, min(1.0, base_size_factor)) * efficiency_discount

    return round(final, 2)
