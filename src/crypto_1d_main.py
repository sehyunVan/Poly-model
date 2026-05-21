"""
Crypto 1-day up/down market trading loop — entry point.

Targets Polymarket's 1-day BTC/SOL up/down markets using the same
crowd-momentum flow signal as the 5m loop, scaled for the daily window.

Run in its own screen session:
    screen -dmS crypto1d bash -c 'source .venv/bin/activate && VIRTUAL_MODE=true python src/crypto_1d_main.py >> logs/crypto_1d.log 2>&1'

Strategy notes:
  - Entry window: 57600–79200s into the 86400s window (67–92% — same proportion as 5m)
  - drift_scale: 0.60 (vs 0.06 for 5m — significant drift accumulation over 24h)
  - Separate state: data/1d_virtual_state.json (virtual) / data/1d_real_state.json (live)
  - Separate cache: data/crypto_1d_cache.jsonl — accumulates 1d-specific labels
  - Start in VIRTUAL mode; go live after 100+ trades at >= 65% WR
  - Polling every 10 min (longer window, less frequent checks)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Must be set BEFORE importing crypto.loop ─────────────────────────────────
os.environ["CRYPTO_CONFIG_FILE"] = "crypto_1d_params.yaml"

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from crypto.loop import run, startup_checks, _setup_logging, VIRTUAL_MODE, _CFG
from infra.backend import make_backend


def main():
    log = _setup_logging()
    log.info("*** Crypto 1d up/down bot starting ***")
    log.info(
        "Timeframe: %s  window=%ds  entry=%d-%ds  drift_scale=%.3f",
        _CFG.get("timeframe", "?"),
        _CFG.get("window_seconds", 0),
        _CFG.get("min_window_elapsed", 0),
        _CFG.get("max_window_elapsed", 0),
        _CFG.get("drift_scale", 0.06),
    )
    try:
        state_file = _CFG.get("state_file_virtual", "1d_virtual_state.json")
        backend = make_backend(VIRTUAL_MODE, log, virtual_state_file=state_file)
        startup_checks(log, backend)
        run(log, backend)
    except RuntimeError as exc:
        log.critical("Startup validation failed — bot will not start: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.critical("1d crypto loop crashed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
