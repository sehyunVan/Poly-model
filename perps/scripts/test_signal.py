"""
Smoke test the signal pipeline: subscribe to BTC, wait 90s for drift window
to fill, then print computed PerpSignal once a second for another 30s.

Usage:
    python perps/scripts/test_signal.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hl_feed import HLFeed  # noqa: E402
from flow import compute_signal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("signal_test")

SIGNAL_THRESHOLD = 0.25
COIN = "BTC"


def main() -> int:
    feed = HLFeed()
    feed.subscribe_coin(COIN)

    log.info("Warming up — waiting 90s for drift window to fill...")
    time.sleep(90)

    log.info("Computing signal once per second for 30s")
    for i in range(30):
        bids, asks = feed.get_book(COIN)
        sig = compute_signal(
            mid_history       = feed.get_mid_history(COIN, window_sec=120.0),
            bids              = bids,
            asks              = asks,
            cvd_score         = feed.get_cvd_score(COIN),
            oracle_lag_score  = feed.get_oracle_lag_score(COIN),
            funding_rate      = feed.get_funding_rate(COIN),
            signal_threshold  = SIGNAL_THRESHOLD,
        )
        log.info(
            "t=%02ds  dir=%s  score=%+.3f  alpha=%+.3f  "
            "drift=%+.2f ob=%+.2f cvd=%+.2f lag=%+.2f fund=%+.2f mom30=%+.2f  "
            "(used=%d/6)  mid=$%.2f",
            i, sig.direction, sig.score, sig.alpha,
            sig.drift_score, sig.ob_score, sig.cvd_score,
            sig.oracle_lag_score, sig.funding_score, sig.mom_30s_score,
            sig.components_used, sig.mid,
        )
        time.sleep(1.0)

    feed.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
