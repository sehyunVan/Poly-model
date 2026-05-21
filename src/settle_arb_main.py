"""
src/settle_arb_main.py — Entry point for the settlement arb bot.

Runs as a separate screen session so it doesn't touch loop.py.
Acts only at 272-290s elapsed (after the main loop's 200-270s window closes).

Start:
  screen -S settle_arb
  source .venv/bin/activate && python src/settle_arb_main.py >> logs/settle_arb.log 2>&1

Config:
  Reads from config/crypto_params.yaml (settle_arb_* keys).
  VIRTUAL_MODE env var controls live vs paper (same as crypto loop).
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("settle_arb_main")

# Ensure src/ is on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

_CFG_PATH = _ROOT.parent / "config" / "crypto_params.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = _load_cfg()

    virtual_mode = os.getenv("VIRTUAL_MODE", "true").lower() not in ("false", "0")
    mode_str = "VIRTUAL" if virtual_mode else "LIVE"

    # settle_arb_enabled defaults to True
    if not cfg.get("settle_arb_enabled", True):
        log.info("settle_arb_enabled=false in config — exiting")
        return

    # Prefer settle_arb-specific symbol list (BTC/ETH/SOL); fall back to active_symbols
    symbols     = cfg.get("settle_arb_symbols", cfg.get("active_symbols", ["BTC"]))
    timeframes  = cfg.get("settle_arb_timeframes", ["5m"])

    log.info("=" * 60)
    log.info("Settlement Arb  mode=%s  symbols=%s  timeframes=%s",
             mode_str, symbols, timeframes)
    # Per-tf action windows + drift thresholds (display defaults; yaml may override)
    _tf_defaults = {
        "5m":  {"start": 285,  "end": 295,  "drift": 0.0005},
        "15m": {"start": 855,  "end": 885,  "drift": 0.0010},
        "1h":  {"start": 3420, "end": 3540, "drift": 0.0020},
    }
    for tf in timeframes:
        d = _tf_defaults.get(tf, {})
        if tf == "5m":
            start = cfg.get("settle_arb_start_elapsed",     d.get("start"))
            end   = cfg.get("settle_arb_end_elapsed",       d.get("end"))
            drift = cfg.get("settle_arb_min_drift",         d.get("drift"))
        else:
            start = cfg.get(f"settle_arb_{tf}_start_elapsed", d.get("start"))
            end   = cfg.get(f"settle_arb_{tf}_end_elapsed",   d.get("end"))
            drift = cfg.get(f"settle_arb_{tf}_min_drift",     d.get("drift"))
        log.info("  %s: elapsed %d–%ds  min_drift=%.3f%%", tf, start, end, drift * 100)
    log.info("  ask_band=[%.3f, %.3f]  min_payout=%.3f  bet=$%.2f",
             cfg.get("settle_arb_min_ask", 0.10),
             cfg.get("settle_arb_max_ask", 0.92),
             cfg.get("settle_arb_min_payout", 0.06),
             cfg.get("settle_arb_bet_abs", 3.0))
    log.info("=" * 60)

    # ── Chainlink oracle feed ──────────────────────────────────────────────────
    from crypto.rtds_feed import RTDSFeed
    rtds = RTDSFeed(symbols)
    rtds.start()
    log.info("Chainlink oracle feed started — warming up 20s")
    time.sleep(20)  # let oracle poll at least once before we start checking

    # ── Execution backend ──────────────────────────────────────────────────────
    from infra.backend import make_backend

    class _SettleBackend:
        """Thin wrapper: settle_arb uses a direct CLOB call, not VirtualPortfolio."""

        def __init__(self, is_live: bool):
            self._live = is_live
            if is_live:
                from crypto.execution import _get_clob_client
                self._client = _get_clob_client()

        def place_order(self, token_id: str, price: float, size_usdc: float) -> bool:
            import math
            from py_clob_client_v2.clob_types import OrderArgs  # type: ignore
            import concurrent.futures

            size = math.ceil(size_usdc / price * 10000) / 10000
            if not self._live:
                log.info("VIRTUAL fill: token=%s price=%.3f size=%.4f",
                         token_id[:12], price, size)
                return True

            def _place():
                args = OrderArgs(token_id=token_id, price=price, size=size, side="BUY")
                return self._client.create_and_post_order(args)

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    resp = ex.submit(_place).result(timeout=12)
                log.info("Order response: %s", resp)
                return True
            except Exception as e:
                log.error("Order placement failed: %s", e)
                return False

    backend = _SettleBackend(is_live=not virtual_mode)

    # ── Settlement arb engine ──────────────────────────────────────────────────
    from crypto.settle_arb import SettlementArb
    arb = SettlementArb(rtds_feed=rtds, backend=backend, cfg=cfg)

    # Poll every 10 seconds (fast enough to catch the 272-290s window reliably)
    POLL_INTERVAL = 10

    log.info("Settlement arb polling every %ds", POLL_INTERVAL)

    while True:
        try:
            arb.check()
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down")
            break
        except Exception as e:
            log.error("Unexpected error in settle_arb cycle: %s", e, exc_info=True)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
