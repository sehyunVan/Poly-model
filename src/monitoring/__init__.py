"""
monitoring package — strategy metrics, daily reports, and parameter feedback.

Public interface:

    from monitoring import (
        compute_strategy_metrics,
        compute_prediction_metrics,
        generate_daily_report,
        suggest_parameter_adjustments,
    )

Responsibilities:
    - compute_strategy_metrics()   : PnL, Sharpe, MDD, hit rate from daily history.
    - compute_prediction_metrics() : Brier score, log-loss, calibration vs actuals.
    - generate_daily_report()      : Markdown report saved to logs/reports/YYYY-MM-DD.md.
    - suggest_parameter_adjustments(): Rule-based advisory suggestions (never auto-applied).
"""

from .metrics  import compute_strategy_metrics, compute_prediction_metrics
from .report   import generate_daily_report
from .feedback import suggest_parameter_adjustments
from .alerts   import send_alert, send_daily_summary, send_gate_check

__all__ = [
    "compute_strategy_metrics",
    "compute_prediction_metrics",
    "generate_daily_report",
    "suggest_parameter_adjustments",
    "send_alert",
    "send_daily_summary",
    "send_gate_check",
]
