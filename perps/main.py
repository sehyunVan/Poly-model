"""
perps/main.py — Phase 0 paper-trading loop for HL perps.

Long-running process. Subscribes to BTC perp WS, evaluates the signal every
loop_interval seconds, opens/closes paper positions per the semantics in
perps_params.yaml, and writes a structured JSONL log.

Run:
    python perps/main.py

Logs:
    perps/logs/perps.log        — INFO-level status
    perps/data/perps_paper.jsonl — every open/close event with full context
    perps/data/perps_state.json  — current state (open positions, cumulative PnL)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import yaml

try:
    import requests
except ImportError:  # requests is universally present on the server, no fallback needed
    requests = None  # type: ignore

# Make `src.*` imports work when run from any cwd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from hl_feed import HLFeed                          # noqa: E402
from flow import compute_signal                     # noqa: E402
from paper import (                                  # noqa: E402
    PaperState, can_open, close_position, load_state,
    open_position, save_state, should_close,
)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    # When run under screen with `>> logs/perps.log 2>&1`, stdout is the log file.
    # Use StreamHandler only to avoid duplicate writes.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("perps.main")


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_config(HERE / "config" / "perps_params.yaml")
    log = setup_logging(HERE / "logs")

    coin              = cfg["symbol"]
    loop_interval     = float(cfg["loop_interval"])
    hold_seconds      = int(cfg["hold_seconds"])
    notional_usd      = float(cfg["position_notional_usd"])
    max_concurrent    = int(cfg["max_concurrent_positions"])
    signal_threshold  = float(cfg["signal_threshold"])
    max_signal_score  = float(cfg["max_signal_score"])
    min_depth         = float(cfg["min_book_depth_usd"])
    max_spread_bps    = float(cfg["max_spread_bps"])
    cooldown_sec      = int(cfg["cooldown_seconds"])
    fee_rate          = float(cfg["fee_rate_taker"])
    slippage_ticks    = int(cfg["slippage_ticks"])
    tick_size         = float(cfg["tick_size_btc"])
    funding_per_hour  = bool(cfg["funding_per_hour"])
    paper_log_path    = str(HERE / cfg["paper_log_path"])
    state_path        = str(HERE / cfg["state_path"])

    state = load_state(state_path)
    log.info(
        "Loaded state: open=%d closed=%d wr=%.1f%% cum_pnl=%+.2f",
        len(state.open_positions), state.closed_count,
        (state.win_count / state.closed_count * 100) if state.closed_count else 0.0,
        state.cumulative_pnl,
    )

    feed = HLFeed()
    feed.subscribe_coin(coin)
    log.info("HLFeed subscribed coin=%s — warming up 90s before first signal eval", coin)
    time.sleep(90)

    # Staleness alerting state
    stale_alert_sent = False
    stale_since: float | None = None
    STALE_ALERT_AFTER = 300.0   # 5 min stale → alert
    ntfy_topic = os.environ.get("NTFY_TOPIC", "pm-bot-1207")

    def _ntfy(title: str, body: str) -> None:
        if requests is None:
            return
        try:
            requests.post(
                f"https://ntfy.sh/{ntfy_topic}",
                data=body.encode("utf-8"),
                headers={"Title": title.encode("ascii", "replace").decode()},
                timeout=5,
            )
        except Exception as exc:
            log.warning("ntfy POST failed: %s", exc)

    while True:
        try:
            now = time.time()

            # ── Staleness monitor (one-shot alert) ────────────────────────────
            if not feed.is_live(coin):
                if stale_since is None:
                    stale_since = now
                elif (now - stale_since) > STALE_ALERT_AFTER and not stale_alert_sent:
                    log.error("Feed stale for %.0fs — sending ntfy alert", now - stale_since)
                    _ntfy(
                        "perps feed STALE",
                        f"BTC feed stale {int(now - stale_since)}s on {coin}. Check WS reconnect.",
                    )
                    stale_alert_sent = True
            else:
                if stale_alert_sent:
                    log.info("Feed recovered after %.0fs", now - (stale_since or now))
                    _ntfy("perps feed recovered", f"BTC feed back live on {coin}")
                stale_since = None
                stale_alert_sent = False

            # ── 1. Close any positions that have hit their hold expiry ────────
            for pos in list(state.open_positions):
                if should_close(pos, now):
                    mid = feed.get_mid(pos.coin)
                    if mid is None:
                        log.warning("Cannot close %s — feed stale, will retry", pos.trade_id)
                        continue
                    close_position(
                        state, pos,
                        mid=mid, now=now,
                        slippage_ticks=slippage_ticks,
                        tick_size=tick_size,
                        fee_rate=fee_rate,
                        funding_per_hour=funding_per_hour,
                        log_path=paper_log_path,
                    )
                    save_state(state, state_path)

            # ── 2. Check if we can open a new position ────────────────────────
            allowed, reason = can_open(state, coin, now, cooldown_sec, max_concurrent)
            if not allowed:
                time.sleep(loop_interval)
                continue

            # ── 3. Liquidity gates (book depth + spread) ──────────────────────
            bids, asks = feed.get_book(coin)
            if not bids or not asks:
                log.debug("book stale — skip")
                time.sleep(loop_interval)
                continue
            top5_bid = sum(float(l["sz"]) * float(l["px"]) for l in bids[:5])
            top5_ask = sum(float(l["sz"]) * float(l["px"]) for l in asks[:5])
            if min(top5_bid, top5_ask) < min_depth:
                log.debug("depth too shallow bid5=%.0f ask5=%.0f — skip", top5_bid, top5_ask)
                time.sleep(loop_interval)
                continue
            spread_bps = feed.get_spread_bps(coin)
            if spread_bps is None or spread_bps > max_spread_bps:
                log.debug("spread %.2f bps > %.2f — skip", spread_bps or -1, max_spread_bps)
                time.sleep(loop_interval)
                continue

            # ── 4. Compute signal ────────────────────────────────────────────
            sig = compute_signal(
                mid_history       = feed.get_mid_history(coin, window_sec=120.0),
                bids              = bids,
                asks              = asks,
                cvd_score         = feed.get_cvd_score(coin),
                oracle_lag_score  = feed.get_oracle_lag_score(coin),
                funding_rate      = feed.get_funding_rate(coin),
                signal_threshold  = signal_threshold,
            )

            if sig.direction == "NO_TRADE":
                log.debug("NO_TRADE score=%+.3f alpha=%+.3f", sig.score, sig.alpha)
                time.sleep(loop_interval)
                continue
            if abs(sig.score) >= max_signal_score:
                log.info("SKIP score cap |%+.3f| >= %.2f", sig.score, max_signal_score)
                time.sleep(loop_interval)
                continue

            # ── 5. Open paper position ───────────────────────────────────────
            components = {
                "drift":   sig.drift_score,
                "ob":      sig.ob_score,
                "cvd":     sig.cvd_score,
                "lag":     sig.oracle_lag_score,
                "funding": sig.funding_score,
                "mom30":   sig.mom_30s_score,
                "used":    sig.components_used,
            }
            open_position(
                state,
                coin=coin,
                side=sig.direction,
                score=sig.score,
                components=components,
                mid=sig.mid,
                funding_rate=feed.get_funding_rate(coin) or 0.0,
                now=now,
                notional_usd=notional_usd,
                slippage_ticks=slippage_ticks,
                tick_size=tick_size,
                fee_rate=fee_rate,
                hold_seconds=hold_seconds,
                log_path=paper_log_path,
            )
            save_state(state, state_path)
            time.sleep(loop_interval)

        except KeyboardInterrupt:
            log.info("Interrupted — saving state and exiting")
            save_state(state, state_path)
            feed.stop()
            return 0
        except Exception as exc:
            log.exception("Loop iteration failed: %s — retrying in 5s", exc)
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
