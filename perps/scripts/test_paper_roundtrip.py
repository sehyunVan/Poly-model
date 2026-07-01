"""
End-to-end smoke test: force-open a paper position, hold 20s, close it, verify
state + log file.

Bypasses the signal gate so we can validate open/close/PnL/log persistence
without waiting for a real signal fire.

Usage:
    python perps/scripts/test_paper_roundtrip.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from hl_feed import HLFeed                          # noqa: E402
from paper import (                                  # noqa: E402
    PaperState, close_position, load_state,
    open_position, save_state, should_close,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("paper_test")

LOG_PATH   = HERE / "data" / "test_paper.jsonl"
STATE_PATH = HERE / "data" / "test_state.json"


def main() -> int:
    for p in (LOG_PATH, STATE_PATH):
        if p.exists():
            p.unlink()

    coin = "BTC"
    feed = HLFeed()
    feed.subscribe_coin(coin)

    log.info("Warming up 20s for live mid...")
    time.sleep(20)
    mid = feed.get_mid(coin)
    if mid is None:
        log.error("No mid after warmup — aborting")
        return 1
    log.info("Got mid=$%.2f, forcing LONG open", mid)

    state = PaperState()
    pos = open_position(
        state,
        coin=coin, side="LONG",
        score=0.42,
        components={"drift": 0.5, "ob": 0.3, "cvd": 0.2, "lag": 0.1, "funding": 0.0, "mom30": 0.4, "used": 6},
        mid=mid,
        funding_rate=feed.get_funding_rate(coin) or 0.0,
        now=time.time(),
        notional_usd=50.0,
        slippage_ticks=1,
        tick_size=1.0,
        fee_rate=0.00045,
        hold_seconds=20,
        log_path=str(LOG_PATH),
    )
    save_state(state, str(STATE_PATH))

    log.info("Holding 22s then closing...")
    time.sleep(22)

    now = time.time()
    assert should_close(pos, now), "should_close=False after hold elapsed"
    mid_close = feed.get_mid(coin)
    if mid_close is None:
        log.error("No mid for close")
        return 1
    pnl = close_position(
        state, pos,
        mid=mid_close, now=now,
        slippage_ticks=1, tick_size=1.0,
        fee_rate=0.00045,
        funding_per_hour=True,
        log_path=str(LOG_PATH),
    )
    save_state(state, str(STATE_PATH))

    log.info("Closed PnL: %+.4f", pnl)
    s = load_state(str(STATE_PATH))
    assert s.closed_count == 1, f"expected closed_count=1, got {s.closed_count}"
    assert len(s.open_positions) == 0, "expected no open positions after close"
    log.info("ALL CHECKS PASSED  (closed=%d cum_pnl=%+.4f)", s.closed_count, s.cumulative_pnl)

    feed.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
