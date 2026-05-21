"""TASK-5 validation tests."""
import sys, math
sys.path.insert(0, "src")

from datetime import datetime, timezone, timedelta
from data.schemas import OrderBookSnapshot, OrderBookLevel
from signal_layer.schemas import TradeSignal
from risk import (
    Position, Portfolio, RiskLimits,
    check_exposure_limits, compute_position_size,
    estimate_slippage, check_liquidity, evaluate_portfolio_risk,
)


def make_ob(bids, asks):
    return OrderBookSnapshot(
        market_id="0xabc",
        timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=p, size=s) for p, s in bids],
        asks=[OrderBookLevel(price=p, size=s) for p, s in asks],
    )


def make_sig(market_id="0xabc"):
    return TradeSignal(
        market_id=market_id,
        timestamp=datetime.now(timezone.utc),
        direction="BUY_YES",
        alpha=0.10,
        base_size_factor=0.5,
        P_M=0.60,
        P_R=0.70,
        confidence=0.8,
    )


# ---------------------------------------------------------------------------
# 1. RiskLimits.from_config()
# ---------------------------------------------------------------------------
def test_risk_limits_config():
    lim = RiskLimits.from_config()
    assert lim.max_single_event_pct == 0.10
    assert lim.max_daily_loss_pct   == 0.05
    assert lim.max_weekly_loss_pct  == 0.10
    assert lim.kelly_fraction       == 0.25
    assert lim.max_bet_pct          == 0.05
    assert lim.max_slippage_pct     == 0.01
    assert lim.category_limit("politics") == 0.40
    assert lim.category_limit("crypto")   == 0.30
    assert lim.category_limit("unknown")  == 0.20  # fallback to 'other'
    print("RiskLimits.from_config() OK")


# ---------------------------------------------------------------------------
# 2. Position helpers
# ---------------------------------------------------------------------------
def test_position_pnl():
    pos = Position(
        market_id="0xabc", token_id="111", side="YES",
        size=500.0, avg_entry_price=0.60, current_price=0.66, category="politics",
    )
    pos = pos.recalculate_pnl()
    expected = (0.66 - 0.60) / 0.60 * 500.0   # 50.0
    assert abs(pos.unrealized_pnl - expected) < 0.01
    print(f"Position.recalculate_pnl OK: pnl={pos.unrealized_pnl:.2f}")


# ---------------------------------------------------------------------------
# 3. Portfolio helpers
# ---------------------------------------------------------------------------
def test_portfolio_helpers():
    pos1 = Position(market_id="0xabc", token_id="111", side="YES",
                    size=500.0, avg_entry_price=0.60, current_price=0.66,
                    category="politics", group_id="grp_a")
    pos2 = Position(market_id="0xabc", token_id="222", side="YES",
                    size=300.0, avg_entry_price=0.55, current_price=0.58,
                    category="politics", group_id="grp_a")
    pos3 = Position(market_id="0xdef", token_id="333", side="YES",
                    size=200.0, avg_entry_price=0.40, current_price=0.45,
                    category="crypto")
    port = Portfolio(positions=[pos1, pos2, pos3],
                     total_capital=10000.0, available_capital=9000.0,
                     daily_pnl=-50.0, weekly_pnl=200.0)
    assert port.exposure_for_market("0xabc")   == 800.0
    assert port.exposure_for_category("politics") == 800.0
    assert port.exposure_for_group("grp_a")    == 800.0
    assert port.total_deployed()               == 1000.0
    print("Portfolio helpers OK")


# ---------------------------------------------------------------------------
# 4. check_exposure_limits — PASS + 4 blocking cases
# ---------------------------------------------------------------------------
def test_check_exposure_limits():
    # Use large capital (100k) so each limit is independently reachable.
    # max_single_event = 10k, max_politics = 40k, max_group = 25k, daily_floor = -5k
    CAP = 100_000.0
    lim = RiskLimits.from_config()

    # --- Block 1: single-event ---
    # market 0xabc already has 9500; add 600 → 10100 > 10000 = 10% of 100k
    pos_big = Position(market_id="0xabc", token_id="111", side="YES",
                       size=9500.0, avg_entry_price=0.60, current_price=0.66,
                       category="politics")
    port1 = Portfolio(positions=[pos_big], total_capital=CAP,
                      available_capital=90500.0, daily_pnl=0.0)
    sig_abc = make_sig("0xabc")
    ok, r = check_exposure_limits(sig_abc, 100.0, port1, lim, category="politics")
    assert ok and r == "", r
    print("Exposure PASS OK")

    ok1, r1 = check_exposure_limits(sig_abc, 600.0, port1, lim, category="politics")
    assert not ok1 and "single-event" in r1
    print(f"Block 1 (single-event): {r1[:75]}")

    # --- Block 2: category ---
    # 39 different markets in politics, each 1000 USDC = 39k total
    # Add 2000 to a brand-new market → politics total = 41k > 40k
    # single-event for new market: 0+2000=2000 < 10k ✓
    pol_positions = [
        Position(market_id=f"0xpol{i:03d}", token_id=f"t{i}", side="YES",
                 size=1000.0, avg_entry_price=0.50, current_price=0.50,
                 category="politics")
        for i in range(39)
    ]
    port2 = Portfolio(positions=pol_positions, total_capital=CAP,
                      available_capital=61000.0, daily_pnl=0.0)
    sig_new = make_sig("0xnewmarket")
    ok2, r2 = check_exposure_limits(sig_new, 2000.0, port2, lim, category="politics")
    assert not ok2 and "category" in r2
    print(f"Block 2 (category):     {r2[:75]}")

    # --- Block 3: correlated group ---
    # group "grp_a": 24 markets × 1000 = 24k total
    # Add 2000 to a new market in the group → 26k > 25k = 25% of 100k
    # single-event: 0+2000=2000 < 10k ✓; category: 24k+2k=26k < 40k ✓
    grp_positions = [
        Position(market_id=f"0xgrp{i:03d}", token_id=f"g{i}", side="YES",
                 size=1000.0, avg_entry_price=0.50, current_price=0.50,
                 category="politics", group_id="grp_a")
        for i in range(24)
    ]
    port3 = Portfolio(positions=grp_positions, total_capital=CAP,
                      available_capital=76000.0, daily_pnl=0.0)
    sig_grp = make_sig("0xgrpnew")
    ok3, r3 = check_exposure_limits(sig_grp, 2000.0, port3, lim,
                                    category="politics", group_id="grp_a")
    assert not ok3 and "correlated group" in r3
    print(f"Block 3 (group):        {r3[:75]}")

    # --- Block 4: daily loss gate ---
    # floor = -5000; set daily_pnl = -5001
    port4 = Portfolio(positions=[], total_capital=CAP,
                      available_capital=CAP, daily_pnl=-5001.0)
    sig_any = make_sig("0xany")
    ok4, r4 = check_exposure_limits(sig_any, 100.0, port4, lim, category="politics")
    assert not ok4 and "daily loss gate" in r4
    print(f"Block 4 (daily loss):   {r4[:75]}")


# ---------------------------------------------------------------------------
# 5. compute_position_size
# ---------------------------------------------------------------------------
def test_compute_position_size():
    lim = RiskLimits.from_config()  # kelly=0.25, max_bet=5% → 500 USDC on 10k

    # Strong YES edge: P_R=0.75, P_M=0.60, alpha=+0.15
    # b=(0.40/0.60)=0.667; f*=(0.75*0.667-0.25)/0.667=0.375; f=0.375*0.25=0.09375
    # raw=937.5 → capped=500; factor=1.0 → 500
    s1 = compute_position_size(0.15, 0.75, 0.60, 10000.0, lim, 1.0)
    assert s1 == 500.0, f"got {s1}"
    print(f"YES edge size OK: {s1}")

    # NO edge: alpha=-0.15 (P_R=0.45, P_M=0.60) → win_prob=0.55, b_no=0.60/0.40=1.5
    # f*=(0.55*1.5-0.45)/1.5=0.25; f=0.25*0.25=0.0625; raw=625 → cap=500
    s2 = compute_position_size(-0.15, 0.45, 0.60, 10000.0, lim, 1.0)
    assert s2 == 500.0, f"got {s2}"
    print(f"NO edge size  OK: {s2}")

    # base_size_factor halves the result
    s3 = compute_position_size(0.15, 0.75, 0.60, 10000.0, lim, 0.5)
    assert abs(s3 - 250.0) < 1.0
    print(f"base_size_factor=0.5 OK: {s3}")

    # Negative Kelly edge → 0  (P_R < P_M → f* < 0)
    s4 = compute_position_size(0.02, 0.52, 0.60, 10000.0, lim)
    assert s4 == 0.0
    print(f"Negative Kelly OK: {s4}")

    # max_bet_pct hard cap: very high P_R still capped
    s5 = compute_position_size(0.30, 0.90, 0.60, 10000.0, lim, 1.0)
    assert s5 == lim.max_bet_pct * 10000.0
    print(f"max_bet_pct cap OK: {s5}")


# ---------------------------------------------------------------------------
# 6. estimate_slippage
# ---------------------------------------------------------------------------
def test_estimate_slippage():
    # All liquidity at best ask → 0 slippage
    ob1 = make_ob([(0.64, 1000)], [(0.66, 1000)])
    assert estimate_slippage(ob1, 100.0, "BUY") == 0.0
    print("slippage uniform: 0.0 OK")

    # Two levels: 200 USDC @ 0.66, then 100 USDC @ 0.67
    ob2 = make_ob([], [(0.66, 200), (0.67, 100)])
    slip = estimate_slippage(ob2, 300.0, "BUY")
    avg  = (200 * 0.66 + 100 * 0.67) / 300
    expected = abs(avg - 0.66) / 0.66
    assert abs(slip - expected) < 1e-5, f"{slip} vs {expected}"
    print(f"slippage two-level: {slip:.6f} OK")

    # Empty book → 1.0
    ob_empty = OrderBookSnapshot(market_id="0xabc", timestamp=datetime.now(timezone.utc))
    assert estimate_slippage(ob_empty, 100.0, "BUY") == 1.0
    print("slippage empty book: 1.0 OK")


# ---------------------------------------------------------------------------
# 7. check_liquidity
# ---------------------------------------------------------------------------
def test_check_liquidity():
    # 10 levels, 300 USDC each side → depth=6000 >> min=500; vol=2000 >> min=1000
    ob_deep = make_ob([(0.64, 300)] * 5, [(0.66, 300)] * 5)
    score = check_liquidity(ob_deep, 2000.0, 500.0, 1000.0)
    assert score == 1.0
    print(f"liquidity full: {score}")

    # Depth exactly at half (250 vs min=500) → depth_score=0.5; vol full → geomean
    ob_half = make_ob([(0.64, 12.5)] * 10, [(0.66, 12.5)] * 10)  # total bid+ask = 250
    score2 = check_liquidity(ob_half, 2000.0, 500.0, 1000.0)
    expected = math.sqrt(0.5 * 1.0)
    assert abs(score2 - round(expected, 4)) < 1e-4
    print(f"liquidity half-depth: {score2:.4f} OK")

    # Both zero → 0.0
    ob_empty = OrderBookSnapshot(market_id="0xabc", timestamp=datetime.now(timezone.utc))
    assert check_liquidity(ob_empty, 0.0) == 0.0
    print("liquidity zero: 0.0 OK")


# ---------------------------------------------------------------------------
# 8. evaluate_portfolio_risk
# ---------------------------------------------------------------------------
def test_evaluate_portfolio_risk():
    lim = RiskLimits.from_config()
    now = datetime.now(timezone.utc)

    pos_small = Position(market_id="0xabc", token_id="111", side="YES",
                         size=200.0, avg_entry_price=0.60, current_price=0.66,
                         category="politics")

    # PASS — resume_trading
    port_ok = Portfolio(positions=[], total_capital=10000.0,
                        available_capital=10000.0, daily_pnl=0.0)
    acts = evaluate_portfolio_risk(port_ok, [], lim)
    assert len(acts) == 1 and acts[0]["action"] == "resume_trading"
    print("portfolio risk clean: resume_trading OK")

    # Daily loss limit triggered
    port_daily = Portfolio(positions=[pos_small], total_capital=10000.0,
                           available_capital=9800.0, daily_pnl=-600.0)
    acts_d = evaluate_portfolio_risk(port_daily, [], lim)
    types_d = [a["action"] for a in acts_d]
    assert "halt_new_trades" in types_d
    assert types_d.count("close_position") == 1
    print(f"portfolio risk daily loss: {types_d}")

    # Weekly loss limit triggered (no daily loss)
    yesterday = now - timedelta(days=1)
    history   = [{"date": yesterday, "pnl": -1200.0}]   # -12% of 10k
    port_week = Portfolio(positions=[pos_small], total_capital=10000.0,
                          available_capital=9800.0, daily_pnl=0.0)
    acts_w = evaluate_portfolio_risk(port_week, history, lim)
    types_w = [a["action"] for a in acts_w]
    assert "halt_new_trades" in types_w and "close_position" in types_w
    print(f"portfolio risk weekly loss: {types_w}")

    # Concentration violation only (no loss halt)
    pos_big = Position(market_id="0xbig", token_id="999", side="YES",
                       size=1500.0, avg_entry_price=0.50, current_price=0.55,
                       category="crypto")
    port_conc = Portfolio(positions=[pos_big], total_capital=10000.0,
                          available_capital=8500.0, daily_pnl=0.0)
    acts_c = evaluate_portfolio_risk(port_conc, [], lim)
    reduce = next((a for a in acts_c if a["action"] == "reduce_position"), None)
    assert reduce is not None
    assert reduce["target_size"] == lim.max_single_event_pct * 10000.0   # 1000.0
    print(f"portfolio risk concentration: target_size={reduce['target_size']}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_risk_limits_config()
    test_position_pnl()
    test_portfolio_helpers()
    test_check_exposure_limits()
    test_compute_position_size()
    test_estimate_slippage()
    test_check_liquidity()
    test_evaluate_portfolio_risk()
    print()
    print("All checks passed")
