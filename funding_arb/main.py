"""
Funding Rate Arbitrage Bot — multi-position
--------------------------------------------
Strategy: Long spot + Short perp (delta-neutral) for each qualifying symbol.
Multiple simultaneous positions allowed — one per symbol, all managed independently.
Collect funding rate paid every 8h by leveraged longs.
Exit each position independently when its rate drops below threshold.

virtual_mode: true  → paper trading, no real orders, live prices
virtual_mode: false → real money on Binance (needs BINANCE_API_KEY/SECRET in .env)
"""
import logging
import os
import sys
import time

import requests
import yaml
from dotenv import load_dotenv

from src.futures_client import FuturesClient
from src.position_manager import close_position, open_position, rebalance
from src.risk import (
    funding_apy,
    funding_periods_due,
    is_margin_safe,
    should_enter,
    should_enter_reverse,
    should_exit,
)
from src.spot_client import SpotClient
from src.state import ArbState, load_state, save_state

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/arb.log")],
)
log = logging.getLogger("arb.main")


def load_config(path: str = "config/arb_params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _status_block(
    all_rates: list,
    state: ArbState,
    entry_threshold: float,
    now: float,
    position_usdt: float = 50.0,
) -> str:
    """Build a human-readable status block for the log."""
    import datetime
    threshold_apy = funding_apy(entry_threshold)
    lines = []
    ts = datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"{'─'*54}")
    lines.append(f"  ARB STATUS  {ts}")
    lines.append(f"  Entry threshold: {threshold_apy:.1%} APY  |  pos=${position_usdt:.0f}  |  trades={state.trade_count}  earned=${state.total_funding_collected:.3f}")
    lines.append("")

    # Active positions first
    active = {sym: pos for sym, pos in state.positions.items()}
    if active:
        for sym, pos in active.items():
            dur_h = (now - pos["entry_time"]) / 3600
            rate_info = next((r for r in all_rates if r["symbol"] == sym), {})
            rate = rate_info.get("rate", 0.0)
            cur_price = rate_info.get("mark_price", pos["entry_price"])
            price_chg = (cur_price - pos["entry_price"]) / pos["entry_price"] * 100
            est_pending = pos["position_usdt"] * rate * (
                (now - pos["last_funding_collected_time"]) / (8 * 3600)
            )
            lines.append(f"  ★ ACTIVE: {sym}  {dur_h:.1f}h open")
            lines.append(f"    Entry ${pos['entry_price']:.4f} → Now ${cur_price:.4f}  ({price_chg:+.2f}%)")
            lines.append(f"    Rate {funding_apy(rate):.1%} APY  |  Collected ${state.total_funding_collected:.4f}  |  Pending ≈${est_pending:.4f}")
        lines.append("")

    # Rate bar chart
    lines.append("  RATES (10 bars = threshold):")
    for r in all_rates:
        sym    = r["symbol"].replace("USDT", "")
        apy    = r["apy"]           # fractional e.g. 0.049
        is_open = r["symbol"] in active
        # fill bar: 10 blocks = threshold_apy
        filled = min(10, int(round(apy / threshold_apy * 10))) if apy > 0 else 0
        bar    = "█" * filled + "░" * (10 - filled)
        status = " [OPEN]" if is_open else (" ← ENTER!" if apy >= threshold_apy else "")
        neg    = " (negative)" if apy < 0 else ""
        lines.append(f"    {sym:<4}  {bar}  {apy*100:+5.2f}% APY{status}{neg}")

    lines.append(f"{'─'*54}")
    return "\n".join(lines)


def _ntfy(msg: str) -> None:
    topic = os.getenv("NTFY_TOPIC", "")
    if not topic:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=msg.encode(),
            headers={"Title": "Arb Bot"},
            timeout=5,
        )
    except Exception:
        pass


def collect_funding_for(state: ArbState, pos: dict, rate: float) -> float:
    """Collect N pending 8h funding periods for one position. Returns total collected."""
    periods = funding_periods_due(pos["last_funding_collected_time"])
    if periods <= 0:
        return 0.0
    collected = pos["position_usdt"] * rate * periods
    state.total_funding_collected += collected
    pos["last_funding_collected_time"] += periods * 8 * 3600
    return collected


def run() -> None:
    load_dotenv()
    cfg = load_config()
    virtual = cfg.get("virtual_mode", False)
    state = load_state()

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not virtual and (not api_key or not api_secret):
        log.critical(
            "virtual_mode=false but BINANCE_API_KEY/SECRET not set. "
            "Set credentials or set virtual_mode: true in config/arb_params.yaml"
        )
        sys.exit(1)

    spot = SpotClient(api_key, api_secret)
    futures = FuturesClient(api_key, api_secret)

    if not virtual:
        result = futures.enable_hedge_mode()
        log.info(f"Hedge mode: {result.get('msg', 'enabled')}")

    symbols = cfg.get("symbols", [cfg.get("symbol", "BTCUSDT")])
    min_hold_seconds = cfg.get("min_hold_seconds", 14400)
    enable_neg = cfg.get("enable_negative_funding", False)
    approach_alert_pct = cfg.get("approach_alert_pct", 0.70)

    mode_str = "VIRTUAL (paper trading)" if virtual else "LIVE (real money)"
    log.info("=" * 60)
    log.info(f"Funding Arb Bot (multi-position) — mode={mode_str}")
    log.info(f"symbols={symbols}")
    log.info(
        f"entry_threshold={cfg['entry_funding_rate']:.6f}/8h "
        f"({funding_apy(cfg['entry_funding_rate']):.2%} APY)  "
        f"exit_threshold={cfg['exit_funding_rate']:.6f}/8h  "
        f"min_hold={min_hold_seconds / 3600:.0f}h"
    )
    open_syms = list(state.positions.keys())
    log.info(
        f"Resuming: open_positions={open_syms}  trades={state.trade_count}  "
        f"total_funding={state.total_funding_collected:.4f} USDT  "
        f"total_pnl={state.total_realized_pnl:.4f} USDT"
    )
    log.info("=" * 60)

    _approach_alerted: dict[str, bool] = {}

    while True:
        try:
            all_rates = futures.get_all_funding_rates(symbols)

            if not all_rates:
                log.warning("Failed to fetch rates for any symbol — retrying")
                time.sleep(30)
                continue

            # Save rates to state for dashboard, then print status block
            state.current_rates = {
                r["symbol"]: {
                    "rate": r["rate"],
                    "apy": r["apy"],
                    "mark_price": r.get("mark_price", 0.0),
                }
                for r in all_rates
            }
            save_state(state)
            log.info("\n" + _status_block(all_rates, state, cfg["entry_funding_rate"], time.time(), cfg["position_usdt"]))

            # ── Per-symbol logic ──────────────────────────────────────────────
            for rate_info in all_rates:
                sym = rate_info["symbol"]
                rate = rate_info["rate"]
                price = rate_info["mark_price"]

                if sym in state.positions:
                    # ── Manage open position ──────────────────────────────────
                    pos = state.positions[sym]

                    collected = collect_funding_for(state, pos, rate)
                    if collected > 0:
                        duration_h = (time.time() - pos["entry_time"]) / 3600
                        log.info(
                            f"FUNDING {sym}: +{collected:.4f} USDT  "
                            f"total={state.total_funding_collected:.4f}  "
                            f"({duration_h:.0f}h active)"
                        )
                        _ntfy(
                            f"Arb funding {sym}: +{collected:.4f} USDT "
                            f"total={state.total_funding_collected:.4f}"
                        )
                        save_state(state)

                    margin_safe = is_margin_safe(
                        current_price=price,
                        entry_price=pos["entry_price"],
                        leverage=cfg["futures_leverage"],
                        min_distance=cfg["min_margin_distance"],
                    )
                    exit_flag, exit_reason = should_exit(
                        rate=rate,
                        exit_threshold=cfg["exit_funding_rate"],
                        margin_safe=margin_safe,
                        entry_time=pos["entry_time"],
                        min_hold_seconds=min_hold_seconds,
                    )

                    if exit_flag:
                        log.info(f"EXIT {sym} — {exit_reason}")
                        _ntfy(f"Arb EXIT {sym}: {exit_reason}")
                        close_position(state, spot, futures, cfg, sym, close_price=price)
                        save_state(state)
                    else:
                        rebalance(state, futures, cfg, sym)
                        save_state(state)

                else:
                    # ── Consider entry for idle symbol ────────────────────────
                    if should_enter(rate, cfg["entry_funding_rate"]):
                        log.info(
                            f"ENTRY {sym} — rate={rate:.6f} APY={funding_apy(rate):.2%}"
                        )
                        _ntfy(
                            f"Arb ENTRY {sym}: {funding_apy(rate):.1%} APY "
                            f"(threshold {funding_apy(cfg['entry_funding_rate']):.1%})"
                        )
                        open_position(state, spot, futures, cfg, symbol=sym)
                        _approach_alerted[sym] = False
                        save_state(state)

                    elif should_enter_reverse(rate, cfg["entry_funding_rate"], enable_neg):
                        log.info(
                            f"NEGATIVE FUNDING {sym} rate={rate:.6f} APY={funding_apy(rate):.2%} "
                            f"— reverse position not yet implemented (needs Binance Margin)"
                        )

                    else:
                        # Approach alert
                        if rate > 0:
                            approach_pct = rate / cfg["entry_funding_rate"]
                            alerted = _approach_alerted.get(sym, False)
                            if approach_pct >= approach_alert_pct and not alerted:
                                msg = (
                                    f"Arb approaching: {sym} {funding_apy(rate):.2%} APY "
                                    f"({approach_pct:.0%} of threshold)"
                                )
                                log.info(f"APPROACH ALERT — {msg}")
                                _ntfy(msg)
                                _approach_alerted[sym] = True
                            elif approach_pct < 0.50 and alerted:
                                _approach_alerted[sym] = False

        except KeyboardInterrupt:
            log.info("Interrupted — shutting down")
            break

        except Exception as e:
            state.error_count += 1
            log.error(f"Error #{state.error_count}: {e}", exc_info=True)
            save_state(state)

            if state.error_count >= 10:
                log.critical("10 consecutive errors — halting")
                _ntfy("Arb bot halted: 10 consecutive errors")
                break

            time.sleep(30)
            continue

        state.error_count = 0
        time.sleep(cfg["poll_interval"])


if __name__ == "__main__":
    run()
