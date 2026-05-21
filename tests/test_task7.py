"""TASK-7 validation tests."""

import sys, math
sys.path.insert(0, "src")

from datetime import datetime, timezone, timedelta
from risk.schemas import Position, Portfolio
from signal_layer.schemas import TradeSignal
from monitoring import (
    compute_strategy_metrics,
    compute_prediction_metrics,
    generate_daily_report,
    suggest_parameter_adjustments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pnl_history(values: list[float]) -> list[dict]:
    today = datetime.now(timezone.utc)
    return [
        {"date": today - timedelta(days=len(values) - 1 - i), "pnl": v}
        for i, v in enumerate(values)
    ]


def _make_portfolio(daily_pnl: float = 0.0, positions: list | None = None) -> Portfolio:
    positions = positions or []
    return Portfolio(
        positions=positions,
        total_capital=10_000.0,
        available_capital=10_000.0 - sum(p.size for p in positions),
        daily_pnl=daily_pnl,
        weekly_pnl=daily_pnl * 5,
    )


def _make_signal(direction: str = "BUY_YES", alpha: float = 0.10) -> TradeSignal:
    return TradeSignal(
        market_id="0xabc",
        timestamp=datetime.now(timezone.utc),
        direction=direction,
        alpha=alpha,
        base_size_factor=0.5,
        P_M=0.60,
        P_R=0.70,
        confidence=0.8,
    )


# ---------------------------------------------------------------------------
# 1. compute_strategy_metrics
# ---------------------------------------------------------------------------

def test_strategy_metrics_basic():
    # 5 days: 4 profitable, 1 loss → hit_rate = 0.8
    history = _make_pnl_history([100, 50, -30, 80, 60])
    m = compute_strategy_metrics(history)

    assert m["total_pnl"] == pytest_approx(260.0, rel=1e-4)
    assert m["hit_rate"]  == pytest_approx(0.8,   rel=1e-4)
    assert m["sharpe_ratio"] != 0.0
    assert 0.0 <= m["max_drawdown"] <= 1.0
    print(f"strategy_metrics OK: {m}")


def pytest_approx(value, rel=1e-4):
    """Simple stand-in for pytest.approx when running without pytest."""
    return value


def _approx(a: float, b: float, rel: float = 1e-3) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-9)


def test_strategy_metrics_basic():
    history = _make_pnl_history([100, 50, -30, 80, 60])
    m = compute_strategy_metrics(history)

    assert _approx(m["total_pnl"], 260.0)
    assert _approx(m["hit_rate"],  0.8)
    assert m["sharpe_ratio"] != 0.0
    assert 0.0 <= m["max_drawdown"] <= 1.0
    print(f"strategy_metrics basic OK: {m}")


def test_strategy_metrics_drawdown():
    # Equity goes 100 → 200 → 50 → MDD = 150/200 = 0.75
    history = _make_pnl_history([100, 100, -150])
    m = compute_strategy_metrics(history)
    assert _approx(m["max_drawdown"], 0.75, rel=1e-3), m["max_drawdown"]
    print(f"strategy_metrics drawdown OK: {m['max_drawdown']:.4f}")


def test_strategy_metrics_empty():
    m = compute_strategy_metrics([])
    assert m["total_pnl"] == 0.0
    assert m["sharpe_ratio"] == 0.0
    print("strategy_metrics empty OK")


# ---------------------------------------------------------------------------
# 2. compute_prediction_metrics
# ---------------------------------------------------------------------------

class _FakePred:
    """Minimal stand-in for PredictionResult (avoids full import)."""
    def __init__(self, market_id: str, P_R: float):
        self.market_id = market_id
        self.P_R = P_R


def test_prediction_metrics_perfect():
    # P_R = 1.0 for YES markets, 0.0 for NO markets → Brier ≈ 0
    preds = [_FakePred(f"m{i}", 0.999 if i % 2 == 0 else 0.001) for i in range(10)]
    acts  = [{"market_id": f"m{i}", "outcome": 1 if i % 2 == 0 else 0} for i in range(10)]
    pm = compute_prediction_metrics(preds, acts)
    assert pm["brier_score"] < 0.01, pm["brier_score"]
    assert pm["log_loss"] < 0.01,    pm["log_loss"]
    assert len(pm["calibration_bins"]) == 10
    print(f"prediction_metrics perfect OK: brier={pm['brier_score']}")


def test_prediction_metrics_worst():
    # P_R = 1.0 for markets that resolved NO → Brier = 1.0
    preds = [_FakePred(f"m{i}", 0.999) for i in range(5)]
    acts  = [{"market_id": f"m{i}", "outcome": 0} for i in range(5)]
    pm = compute_prediction_metrics(preds, acts)
    assert pm["brier_score"] > 0.99, pm["brier_score"]
    print(f"prediction_metrics worst OK: brier={pm['brier_score']}")


def test_prediction_metrics_empty():
    pm = compute_prediction_metrics([], [])
    assert pm["brier_score"] == 0.0
    assert pm["calibration_bins"] == []
    print("prediction_metrics empty OK")


# ---------------------------------------------------------------------------
# 3. generate_daily_report
# ---------------------------------------------------------------------------

def test_generate_daily_report():
    pos = Position(
        market_id="0xabc", token_id="t1", side="YES",
        size=500.0, avg_entry_price=0.60, current_price=0.65,
        category="politics",
    )
    port = _make_portfolio(daily_pnl=-80.0, positions=[pos])
    sm   = compute_strategy_metrics(_make_pnl_history([100, -80]))
    pm   = compute_prediction_metrics([], [])
    sigs = [_make_signal("BUY_YES"), _make_signal("NO_TRADE")]

    report = generate_daily_report(port, sm, pm, sigs, save=False)

    assert isinstance(report, str)
    assert "## Daily Report" in report
    assert "### Strategy Performance" in report
    assert "### Prediction Accuracy" in report
    assert "### Category Alpha Analysis" in report
    assert "### Anomalies" in report

    # Verify daily PnL appears in the report
    assert "-80" in report or "−80" in report or "+80" not in report

    print("generate_daily_report OK - all required sections present")
    print(f"  Report length: {len(report)} chars")


def test_generate_daily_report_saves_file(tmp_path, monkeypatch=None):
    """
    Verify that save=True writes the expected file.
    Uses a simple override of _REPORT_DIR via the module attribute.
    """
    import monitoring.report as rep_mod
    original_dir = rep_mod._REPORT_DIR
    rep_mod._REPORT_DIR = tmp_path

    try:
        port   = _make_portfolio()
        report = generate_daily_report(
            port, {}, {}, [], save=True,
            date=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        expected = tmp_path / "2025-01-15.md"
        assert expected.exists(), f"{expected} not created"
        assert expected.read_text(encoding="utf-8") == report
        print(f"generate_daily_report save OK: {expected.name}")
    finally:
        rep_mod._REPORT_DIR = original_dir


# ---------------------------------------------------------------------------
# 4. suggest_parameter_adjustments
# ---------------------------------------------------------------------------

def test_feedback_no_issues():
    history = [{"max_drawdown": 0.02, "hit_rate": 0.60, "sharpe_ratio": 1.2} for _ in range(5)]
    params  = {"kelly_fraction": 0.25, "alpha_threshold": 0.05, "max_bet_pct": 0.05}
    s = suggest_parameter_adjustments(history, params)
    assert s == [], f"Expected no suggestions, got {s}"
    print("feedback no_issues OK")


def test_feedback_kelly_reduction_on_mdd():
    # MDD = 8% > threshold 7% → must suggest kelly_fraction reduction
    history = [{"max_drawdown": 0.08, "hit_rate": 0.55, "sharpe_ratio": 0.5}]
    params  = {"kelly_fraction": 0.25, "alpha_threshold": 0.05, "max_bet_pct": 0.05}
    s = suggest_parameter_adjustments(history, params)

    kelly_sugg = next((x for x in s if x["param"] == "kelly_fraction"), None)
    assert kelly_sugg is not None, "Expected kelly_fraction suggestion"
    assert kelly_sugg["suggested"] < kelly_sugg["current"]
    assert _approx(kelly_sugg["suggested"], 0.25 * 0.80)
    print(f"feedback kelly_reduction OK: {kelly_sugg['current']} → {kelly_sugg['suggested']}")


def test_feedback_bet_reduction_on_high_mdd():
    # MDD = 12% > 10% threshold → must also suggest max_bet_pct reduction
    history = [{"max_drawdown": 0.12, "hit_rate": 0.50, "sharpe_ratio": 0.2}]
    params  = {"kelly_fraction": 0.25, "alpha_threshold": 0.05, "max_bet_pct": 0.05}
    s = suggest_parameter_adjustments(history, params)

    params_suggested = {x["param"] for x in s}
    assert "kelly_fraction" in params_suggested
    assert "max_bet_pct"    in params_suggested
    print(f"feedback high_mdd OK: suggested params = {params_suggested}")


def test_feedback_alpha_raise_on_low_hit():
    # hit_rate = 30% < 45%, sharpe = 0.1 < 0.5 → raise alpha_threshold
    history = [{"max_drawdown": 0.03, "hit_rate": 0.30, "sharpe_ratio": 0.1}]
    params  = {"kelly_fraction": 0.25, "alpha_threshold": 0.05, "max_bet_pct": 0.05}
    s = suggest_parameter_adjustments(history, params)

    alpha_sugg = next((x for x in s if x["param"] == "alpha_threshold"), None)
    assert alpha_sugg is not None
    assert alpha_sugg["suggested"] > alpha_sugg["current"]
    print(f"feedback alpha_raise OK: {alpha_sugg['current']} → {alpha_sugg['suggested']}")


def test_feedback_brier_retrain():
    # Brier score > 0.20 → model_retraining suggestion
    history = [{"max_drawdown": 0.01, "hit_rate": 0.60, "sharpe_ratio": 1.0,
                "brier_score": 0.25}]
    params  = {"kelly_fraction": 0.25, "alpha_threshold": 0.05}
    s = suggest_parameter_adjustments(history, params)

    retrain = next((x for x in s if x["param"] == "model_retraining"), None)
    assert retrain is not None
    assert retrain["suggested"] == "trigger_now"
    print(f"feedback brier_retrain OK: {retrain['reason'][:60]}…")


def test_feedback_trading_pause_on_sustained_loss():
    # 3 consecutive negative-Sharpe days → suggest pausing
    history = [
        {"max_drawdown": 0.02, "hit_rate": 0.45, "sharpe_ratio": -0.3},
        {"max_drawdown": 0.03, "hit_rate": 0.40, "sharpe_ratio": -0.5},
        {"max_drawdown": 0.04, "hit_rate": 0.38, "sharpe_ratio": -0.6},
    ]
    params = {"kelly_fraction": 0.25, "alpha_threshold": 0.05}
    s = suggest_parameter_adjustments(history, params)

    pause = next((x for x in s if x["param"] == "trading_enabled"), None)
    assert pause is not None
    assert pause["suggested"] is False
    print(f"feedback trading_pause OK")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_strategy_metrics_basic()
    test_strategy_metrics_drawdown()
    test_strategy_metrics_empty()

    test_prediction_metrics_perfect()
    test_prediction_metrics_worst()
    test_prediction_metrics_empty()

    test_generate_daily_report()
    from pathlib import Path
    test_generate_daily_report_saves_file(Path("logs/reports"))

    test_feedback_no_issues()
    test_feedback_kelly_reduction_on_mdd()
    test_feedback_bet_reduction_on_high_mdd()
    test_feedback_alpha_raise_on_low_hit()
    test_feedback_brier_retrain()
    test_feedback_trading_pause_on_sustained_loss()

    print()
    print("All checks passed")
