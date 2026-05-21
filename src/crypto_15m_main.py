"""
Crypto 15-minute up/down market trading loop — entry point.

Targets Polymarket's 15-minute BTC/SOL up/down markets using the same
crowd-momentum flow signal as the 5m loop, tuned for the longer window.

Run in its own screen session:
    screen -dmS crypto15 bash -c 'source .venv/bin/activate && python src/crypto_15m_main.py >> logs/crypto_15m.log 2>&1'

Strategy notes:
  - Entry window: 600–780s into the 900s window (67–87% — same proportion as 5m)
  - drift_scale: 0.15 (vs 0.06 for 5m — price drifts further in a longer window)
  - Separate state: data/15m_virtual_state.json (virtual) / data/15m_real_state.json (live)
  - Separate cache: data/crypto_15m_cache.jsonl — accumulates 15m-specific labels
  - Start in VIRTUAL mode; go live after 100+ trades at >= 65% WR
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Must be set BEFORE importing crypto.loop ─────────────────────────────────
# loop.py reads CRYPTO_CONFIG_FILE at module level to determine which config
# file to load. Setting it here makes the 15m loop use 15m-specific parameters.
os.environ["CRYPTO_CONFIG_FILE"] = "crypto_15m_params.yaml"

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from crypto.loop import run, startup_checks, _setup_logging, VIRTUAL_MODE, _CFG
from infra.backend import make_backend


def main():
    log = _setup_logging()
    log.info("*** Crypto 15m up/down bot starting ***")
    log.info(
        "Timeframe: %s  window=%ds  entry=%d-%ds  drift_scale=%.3f",
        _CFG.get("timeframe", "?"),
        _CFG.get("window_seconds", 0),
        _CFG.get("min_window_elapsed", 0),
        _CFG.get("max_window_elapsed", 0),
        _CFG.get("drift_scale", 0.06),
    )
    try:
        state_file = _CFG.get("state_file_virtual", "15m_virtual_state.json")
        backend = make_backend(VIRTUAL_MODE, log, virtual_state_file=state_file)
        startup_checks(log, backend)
        run(log, backend)
    except RuntimeError as exc:
        log.critical("Startup validation failed — bot will not start: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.critical("15m crypto loop crashed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
