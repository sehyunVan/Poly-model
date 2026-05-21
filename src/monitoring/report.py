"""
Daily strategy performance report generator.

generate_daily_report() builds a Markdown string summarising the day's
trading activity and optionally saves it to logs/reports/YYYY-MM-DD.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from risk.schemas import Portfolio
from signal_layer.schemas import TradeSignal

_REPORT_DIR = Path(__file__).resolve().parents[2] / "logs" / "reports"


def generate_daily_report(
    portfolio: Portfolio,
    strategy_metrics: dict,
    prediction_metrics: dict,
    signals_today: list[TradeSignal],
    date: Optional[datetime] = None,
    save: bool = True,
) -> str:
    """
    Produce a Markdown daily report and optionally persist it to disk.

    Sections:
        ## Daily Report {YYYY-MM-DD}
        ### Strategy Performance
        ### Prediction Accuracy
        ### Category Alpha Analysis
        ### Open Positions
        ### Anomalies

    Args:
        portfolio:           Current portfolio snapshot.
        strategy_metrics:    Output of compute_strategy_metrics().
        prediction_metrics:  Output of compute_prediction_metrics().
        signals_today:       All TradeSignal objects generated today.
        date:                Report date; defaults to today UTC.
        save:                When True, write to logs/reports/YYYY-MM-DD.md.

    Returns:
        Complete Markdown string.
    """
    if date is None:
        date = datetime.now(timezone.utc)
    date_str = date.strftime("%Y-%m-%d")

    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        f"## Daily Report {date_str}",
        "",
        f"Generated: {date.strftime('%Y-%m-%dT%H:%M:%SZ')}  ",
        f"Total Capital: ${portfolio.total_capital:,.2f} USDC  ",
        f"Available Capital: ${portfolio.available_capital:,.2f} USDC  ",
        f"Daily PnL: ${strategy_metrics.get('daily_pnl', portfolio.daily_pnl):+,.2f} USDC  ",
        "",
    ]

    # ── Strategy Performance ─────────────────────────────────────────────────
    sm = strategy_metrics
    lines += [
        "### Strategy Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Daily PnL | ${sm.get('daily_pnl', 0):+.2f} |",
        f"| Total PnL | ${sm.get('total_pnl', 0):+.2f} |",
        f"| Sharpe Ratio (ann.) | {sm.get('sharpe_ratio', 0):.3f} |",
        f"| Max Drawdown | {sm.get('max_drawdown', 0) * 100:.2f}% |",
        f"| Hit Rate | {sm.get('hit_rate', 0) * 100:.1f}% |",
        "",
    ]

    # ── Prediction Accuracy ───────────────────────────────────────────────────
    pm     = prediction_metrics
    brier  = pm.get("brier_score", 0.0)
    loglos = pm.get("log_loss", 0.0)
    calib  = pm.get("calibration_bins", [])

    lines += [
        "### Prediction Accuracy",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Brier Score | {brier:.4f} |",
        f"| Log Loss | {loglos:.4f} |",
        "",
    ]

    populated = [b for b in calib if b.get("count", 0) > 0]
    if populated:
        lines += [
            "**Calibration — P_R decile vs actual outcome frequency:**",
            "",
            "| P_R Range | Predicted Mean | Actual Freq | Count |",
            "|-----------|----------------|-------------|-------|",
        ]
        for b in populated:
            lines.append(
                f"| {b['range'][0]:.1f}–{b['range'][1]:.1f} "
                f"| {b['predicted_mean']:.3f} "
                f"| {b['actual_freq']:.3f} "
                f"| {b['count']} |"
            )
        lines.append("")

    # ── Category Alpha Analysis ───────────────────────────────────────────────
    lines += ["### Category Alpha Analysis", ""]

    cat_stats: dict[str, dict] = {}
    for sig in signals_today:
        # Look up category from portfolio positions
        cat = "unknown"
        for pos in portfolio.positions:
            if pos.market_id == sig.market_id:
                cat = pos.category
                break
        entry = cat_stats.setdefault(
            cat, {"buy_yes": 0, "buy_no": 0, "no_trade": 0, "alpha_sum": 0.0}
        )
        if sig.direction == "BUY_YES":
            entry["buy_yes"] += 1
        elif sig.direction == "BUY_NO":
            entry["buy_no"] += 1
        else:
            entry["no_trade"] += 1
        entry["alpha_sum"] += abs(sig.alpha)

    if cat_stats:
        lines += [
            "| Category | BUY_YES | BUY_NO | NO_TRADE | Avg \\|alpha\\| |",
            "|----------|---------|--------|----------|--------------|",
        ]
        for cat, d in sorted(cat_stats.items()):
            total = d["buy_yes"] + d["buy_no"] + d["no_trade"]
            avg_a = d["alpha_sum"] / max(total, 1)
            lines.append(
                f"| {cat} | {d['buy_yes']} | {d['buy_no']} "
                f"| {d['no_trade']} | {avg_a:.4f} |"
            )
    else:
        lines.append("_No signals generated today._")
    lines.append("")

    # ── Open Positions ────────────────────────────────────────────────────────
    if portfolio.positions:
        lines += [
            "### Open Positions",
            "",
            "| Market | Side | Size (USDC) | Entry | Current | Unrealized PnL |",
            "|--------|------|-------------|-------|---------|----------------|",
        ]
        for pos in portfolio.positions:
            mid = pos.market_id
            label = mid[:12] + "…" if len(mid) > 12 else mid
            lines.append(
                f"| {label} | {pos.side} "
                f"| ${pos.size:,.2f} "
                f"| {pos.avg_entry_price:.3f} "
                f"| {pos.current_price:.3f} "
                f"| ${pos.unrealized_pnl:+.2f} |"
            )
        lines.append("")

    # ── Anomaly Warnings ─────────────────────────────────────────────────────
    lines += ["### Anomalies", ""]
    warnings: list[str] = []

    # Daily loss proximity warning (> 3% of capital)
    loss_pct = abs(min(portfolio.daily_pnl, 0.0)) / max(portfolio.total_capital, 1.0)
    if loss_pct > 0.03:
        warnings.append(
            f"- **Daily loss approaching limit**: "
            f"${portfolio.daily_pnl:+.2f} "
            f"({loss_pct * 100:.1f}% of capital)"
        )

    # High capital deployment warning (> 80%)
    deployed    = sum(p.size for p in portfolio.positions)
    deploy_pct  = deployed / max(portfolio.total_capital, 1.0)
    if deploy_pct > 0.80:
        warnings.append(
            f"- **High capital deployment**: "
            f"${deployed:,.2f} ({deploy_pct * 100:.1f}% deployed)"
        )

    # Prediction quality degradation
    if brier > 0.20:
        warnings.append(
            f"- **Prediction quality degraded**: "
            f"Brier score {brier:.4f} > 0.20 — consider retraining models"
        )

    # Negative Sharpe
    if sm.get("sharpe_ratio", 0.0) < 0:
        warnings.append(
            f"- **Negative Sharpe ratio**: "
            f"{sm['sharpe_ratio']:.3f} — review strategy parameters"
        )

    # Liquidity warning: any position with spread-implied low depth (proxy check)
    # (A full liquidity check would require live orderbook data; this is a static guard.)
    for pos in portfolio.positions:
        if pos.size > 0.15 * portfolio.total_capital:
            warnings.append(
                f"- **Concentration risk**: position {pos.market_id[:12]}… "
                f"is ${pos.size:,.2f} ({pos.size / portfolio.total_capital * 100:.1f}% of capital)"
            )

    if warnings:
        lines += warnings
    else:
        lines.append("_No anomalies detected._")
    lines.append("")

    report_md = "\n".join(lines)

    # ── Persist to disk ───────────────────────────────────────────────────────
    if save:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = _REPORT_DIR / f"{date_str}.md"
        report_path.write_text(report_md, encoding="utf-8")

    return report_md
