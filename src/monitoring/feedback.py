"""
Rule-based parameter adjustment suggestions.

suggest_parameter_adjustments() analyses a rolling window of daily metrics
and returns a list of proposed parameter changes.  Suggestions are advisory
only — they are never applied automatically.  The user must approve each one.

Rules implemented:
    1. max_drawdown > 7%   → reduce kelly_fraction by 20%
    2. max_drawdown > 10%  → additionally reduce max_bet_pct by 25%
    3. hit_rate < 45% AND sharpe < 0.5 → raise alpha_threshold by 40%
    4. brier_score > 0.20  → trigger model retraining
    5. sharpe < 0 for ≥ 3 consecutive days → suggest pausing new trades
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Rule thresholds
# ---------------------------------------------------------------------------

_MDD_KELLY_THRESHOLD     = 0.07   # Reduce kelly_fraction when MDD > 7 %
_MDD_BET_THRESHOLD       = 0.10   # Reduce max_bet_pct when MDD > 10 %
_HIT_RATE_THRESHOLD      = 0.45   # Raise alpha_threshold when hit_rate < 45 %
_BRIER_RETRAIN_THRESHOLD = 0.20   # Suggest retraining when Brier score > 0.20
_NEGATIVE_SHARPE_DAYS    = 3      # Consecutive negative-Sharpe days before pause


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggest_parameter_adjustments(
    recent_metrics: list[dict],
    current_params: dict,
) -> list[dict]:
    """
    Analyse recent daily metrics and return a list of parameter change proposals.

    Args:
        recent_metrics: Per-day metric dicts, oldest first.  Each dict should
                        contain the keys from compute_strategy_metrics() and,
                        optionally, "brier_score" from compute_prediction_metrics().
        current_params: Current parameter values.  Recognised keys:
                          alpha_threshold, kelly_fraction, max_bet_pct.

    Returns:
        List of suggestion dicts, each with keys:
            param      — name of the parameter to change
            current    — its current value
            suggested  — the proposed new value
            reason     — human-readable rationale

        Returns an empty list when no adjustments are warranted.
    """
    if not recent_metrics:
        return []

    suggestions: list[dict] = []
    n = len(recent_metrics)

    # Summary statistics over the review window
    max_mdd      = max(m.get("max_drawdown", 0.0) for m in recent_metrics)
    avg_hit_rate = sum(m.get("hit_rate", 0.0) for m in recent_metrics) / n
    avg_sharpe   = sum(m.get("sharpe_ratio", 0.0) for m in recent_metrics) / n
    latest_brier = recent_metrics[-1].get("brier_score", 0.0)

    # Count consecutive days of negative Sharpe starting from the most recent day
    consecutive_neg = 0
    for m in reversed(recent_metrics):
        if m.get("sharpe_ratio", 0.0) < 0:
            consecutive_neg += 1
        else:
            break

    # ── Rule 1: MDD > 7% → reduce kelly_fraction ────────────────────────────
    if max_mdd > _MDD_KELLY_THRESHOLD:
        current_kelly   = float(current_params.get("kelly_fraction", 0.25))
        suggested_kelly = round(current_kelly * 0.80, 4)
        suggestions.append({
            "param":     "kelly_fraction",
            "current":   current_kelly,
            "suggested": suggested_kelly,
            "reason": (
                f"Maximum drawdown over the review window is "
                f"{max_mdd * 100:.1f}% (threshold {_MDD_KELLY_THRESHOLD * 100:.0f}%). "
                f"Reducing Kelly fraction by 20% to shrink position sizes."
            ),
        })

    # ── Rule 2: MDD > 10% → also reduce max_bet_pct ─────────────────────────
    if max_mdd > _MDD_BET_THRESHOLD:
        current_bet   = float(current_params.get("max_bet_pct", 0.05))
        suggested_bet = round(current_bet * 0.75, 4)
        suggestions.append({
            "param":     "max_bet_pct",
            "current":   current_bet,
            "suggested": suggested_bet,
            "reason": (
                f"Maximum drawdown {max_mdd * 100:.1f}% exceeds "
                f"{_MDD_BET_THRESHOLD * 100:.0f}%. "
                f"Hard-capping individual bet size by 25% to limit per-trade exposure."
            ),
        })

    # ── Rule 3: Low hit rate + weak Sharpe → raise alpha_threshold ───────────
    if avg_hit_rate < _HIT_RATE_THRESHOLD and avg_sharpe < 0.5:
        current_alpha   = float(current_params.get("alpha_threshold", 0.05))
        suggested_alpha = round(min(current_alpha * 1.40, 0.20), 4)
        suggestions.append({
            "param":     "alpha_threshold",
            "current":   current_alpha,
            "suggested": suggested_alpha,
            "reason": (
                f"Average hit rate {avg_hit_rate * 100:.1f}% is below the "
                f"{_HIT_RATE_THRESHOLD * 100:.0f}% threshold. "
                f"Raising alpha_threshold by 40% to filter marginal signals "
                f"and reduce false-positive trade entries."
            ),
        })

    # ── Rule 4: Degraded Brier score → flag model retraining ─────────────────
    if latest_brier > _BRIER_RETRAIN_THRESHOLD:
        suggestions.append({
            "param":     "model_retraining",
            "current":   "not_triggered",
            "suggested": "trigger_now",
            "reason": (
                f"Latest Brier score {latest_brier:.4f} exceeds the "
                f"{_BRIER_RETRAIN_THRESHOLD:.2f} threshold. "
                f"Prediction calibration has degraded; a full retraining cycle "
                f"is recommended."
            ),
        })

    # ── Rule 5: Sustained negative Sharpe → suggest trading pause ────────────
    if consecutive_neg >= _NEGATIVE_SHARPE_DAYS:
        suggestions.append({
            "param":     "trading_enabled",
            "current":   True,
            "suggested": False,
            "reason": (
                f"Sharpe ratio has been negative for {consecutive_neg} consecutive "
                f"day(s). Consider pausing new trades until market conditions improve."
            ),
        })

    return suggestions
