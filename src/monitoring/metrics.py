"""
Strategy and prediction performance metrics.

compute_strategy_metrics()   — trading PnL, Sharpe, MDD, hit rate
compute_prediction_metrics() — Brier score, log-loss, calibration bins
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prediction.schemas import PredictionResult


# ---------------------------------------------------------------------------
# Strategy metrics
# ---------------------------------------------------------------------------


def compute_strategy_metrics(
    pnl_history: list[dict],
    initial_capital: float = 1000.0,
) -> dict:
    """
    Compute aggregate trading performance metrics from daily PnL records.

    Args:
        pnl_history:     List of {"date": datetime|str, "pnl": float} dicts,
                         one entry per trading day, oldest first.
        initial_capital: Starting equity in USDC (default 1000).  Used as the
                         equity-curve baseline so that max_drawdown is computed
                         as (peak_equity - trough_equity) / peak_equity.

    Returns:
        {
          "total_pnl":    float,  # Sum of all PnL values
          "daily_pnl":    float,  # Most recent day's PnL
          "sharpe_ratio": float,  # Annualised Sharpe ratio (rf = 0)
          "max_drawdown": float,  # Max peak-to-trough drawdown as a fraction
          "hit_rate":     float,  # Fraction of days with PnL > 0
        }
    """
    if not pnl_history:
        return {
            "total_pnl":    0.0,
            "daily_pnl":    0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "hit_rate":     0.0,
        }

    pnl_values = [float(e.get("pnl", 0.0)) for e in pnl_history]
    total_pnl  = sum(pnl_values)
    daily_pnl  = pnl_values[-1]

    # Annualised Sharpe ratio (rf = 0, 252 trading days per year)
    if len(pnl_values) >= 2:
        mean_pnl = statistics.mean(pnl_values)
        std_pnl  = statistics.stdev(pnl_values)
        sharpe   = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    # Maximum drawdown: largest peak-to-trough decline of the equity curve.
    # Equity = initial_capital + cumulative PnL, so peak is always >= initial_capital
    # at the start and division by near-zero cannot occur.
    equity    = float(initial_capital)
    peak      = equity
    max_dd    = 0.0
    for pnl in pnl_values:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    hit_rate = sum(1 for p in pnl_values if p > 0) / len(pnl_values)

    return {
        "total_pnl":    round(total_pnl, 4),
        "daily_pnl":    round(daily_pnl, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate":     round(hit_rate, 4),
    }


# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------


def compute_prediction_metrics(
    predictions: list,     # list[PredictionResult]
    actuals: list[dict],
) -> dict:
    """
    Evaluate probabilistic forecast quality against resolved market outcomes.

    Args:
        predictions: List of PredictionResult objects with .market_id and .P_R.
        actuals:     List of {"market_id": str, "outcome": int (0 or 1)} dicts.

    Returns:
        {
          "brier_score":      float,       # Mean squared probability error
          "log_loss":         float,       # Mean binary cross-entropy
          "calibration_bins": list[dict],  # 10 equal-width P_R decile bins
        }

    Each calibration bin dict contains:
        {"range": [lo, hi], "predicted_mean": float, "actual_freq": float, "count": int}
    """
    # Build lookup: market_id → resolved outcome
    outcome_map = {str(a["market_id"]): int(a["outcome"]) for a in actuals}

    paired: list[tuple[float, int]] = []
    for pred in predictions:
        outcome = outcome_map.get(str(pred.market_id))
        if outcome is not None:
            # Clip P_R away from 0/1 to avoid log(0)
            p_r = max(1e-7, min(1.0 - 1e-7, float(pred.P_R)))
            paired.append((p_r, outcome))

    if not paired:
        return {
            "brier_score":      0.0,
            "log_loss":         0.0,
            "calibration_bins": [],
        }

    p_vals = [p for p, _ in paired]
    y_vals = [y for _, y in paired]
    n      = len(paired)

    brier   = sum((p - y) ** 2 for p, y in zip(p_vals, y_vals)) / n
    logloss = -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for p, y in zip(p_vals, y_vals)
    ) / n

    # Calibration: 10 equal-width buckets over [0, 1]
    bins: list[dict] = []
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        if i < 9:
            bucket = [(p, y) for p, y in zip(p_vals, y_vals) if lo <= p < hi]
        else:
            bucket = [(p, y) for p, y in zip(p_vals, y_vals) if lo <= p <= hi]

        if bucket:
            pred_mean   = statistics.mean(p for p, _ in bucket)
            actual_freq = statistics.mean(y for _, y in bucket)
            count       = len(bucket)
        else:
            pred_mean = actual_freq = 0.0
            count     = 0

        bins.append({
            "range":          [round(lo, 1), round(hi, 1)],
            "predicted_mean": round(pred_mean, 4),
            "actual_freq":    round(actual_freq, 4),
            "count":          count,
        })

    return {
        "brier_score":      round(brier, 6),
        "log_loss":         round(logloss, 6),
        "calibration_bins": bins,
    }
