"""
hl_main.py — Hyperliquid perpetual trading loop entry point.

Runs the HL loop in its own screen session, completely independent of
the Polymarket crypto loop.  Both loops can run simultaneously.

Usage on server:
    screen -dmS hl bash -c 'source .venv/bin/activate && python src/hl_main.py >> logs/hl.log 2>&1'

To stop:
    screen -S hl -X quit

To watch logs:
    tail -f ~/poly-model/logs/hl.log

Environment variables (set in .env):
    HL_VIRTUAL_MODE=true     paper trading (default — safe)
    HL_VIRTUAL_MODE=false    LIVE trading on Hyperliquid
    HL_ADDRESS               wallet address (0x...)
    HL_KEY                   private key for signing

State files:
    data/hl_virt_state.json  paper trading state
    data/hl_state.json       live trading state
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from hyperliquid.loop import run, _setup_logging  # type: ignore


def main() -> None:
    log = _setup_logging()
    log.info("=" * 60)
    log.info("*** Hyperliquid perp bot starting ***")
    log.info("=" * 60)
    try:
        run(log)
    except KeyboardInterrupt:
        log.info("HL loop interrupted by user")
    except Exception as exc:
        log.critical("HL loop crashed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
