"""
Quick smoke test: subscribe to BTC perp on Hyperliquid mainnet, print live
book/mid/cvd/funding for 30 seconds. Confirms WS connectivity and that the
HLFeed module correctly parses every message type we care about.

Usage:
    python perps/scripts/test_hl_feed.py

Expected output:
    [t=01s] BTC  mid=$104237.50  spread=0.5bps  cvd=None  funding=None  oracle_lag=None
    [t=02s] BTC  mid=$104238.00  spread=0.5bps  cvd=+0.12  funding=+0.000125  oracle_lag=+0.32
    ...
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running this script directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hl_feed import HLFeed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("hl_feed_test")


def fmt(x, fmt_str=".4f"):
    return "None" if x is None else format(x, fmt_str)


def main(duration_sec: int = 30, coin: str = "BTC") -> int:
    feed = HLFeed()
    feed.subscribe_coin(coin)

    log.info("Watching %s for %ds — printing once per second", coin, duration_sec)
    start = time.time()
    last = 0.0
    while time.time() - start < duration_sec:
        time.sleep(0.25)
        now = time.time()
        if now - last < 1.0:
            continue
        last = now
        t = int(now - start)
        mid = feed.get_mid(coin)
        spread = feed.get_spread_bps(coin)
        cvd = feed.get_cvd_score(coin)
        funding = feed.get_funding_rate(coin)
        oracle_lag = feed.get_oracle_lag_score(coin)
        is_live = feed.is_live(coin)
        log.info(
            "t=%02ds  %s  live=%s  mid=%s  spread_bps=%s  cvd=%s  funding=%s  oracle_lag=%s",
            t, coin, is_live,
            fmt(mid, ".2f"),
            fmt(spread, ".2f"),
            fmt(cvd, "+.3f"),
            fmt(funding, "+.6f"),
            fmt(oracle_lag, "+.3f"),
        )

    bids, asks = feed.get_book(coin)
    log.info("Final book depth: bids=%d asks=%d", len(bids), len(asks))
    if bids and asks:
        log.info("Best bid/ask: %s / %s", bids[0], asks[0])

    feed.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
