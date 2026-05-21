"""
Crypto up/down fast trading loop — entry point.

Run in its own screen session:
    screen -dmS crypto bash -c 'source .venv/bin/activate && python src/crypto_main.py >> logs/crypto.log 2>&1'
"""
from __future__ import annotations

import logging.handlers
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from crypto.loop import run, startup_checks, _setup_logging, VIRTUAL_MODE, _vstate_name
from infra.backend import make_backend


def main():
    log = _setup_logging()
    log.info("*** Crypto up/down bot starting ***")
    try:
        # make_backend() is the ONLY place that inspects VIRTUAL_MODE.
        # Everything downstream (startup_checks, run) receives the backend object
        # and never checks the env var directly.
        # Pass _vstate_name so yaml override (state_file_virtual) actually takes effect.
        backend = make_backend(VIRTUAL_MODE, log, virtual_state_file=_vstate_name)
        startup_checks(log, backend)   # hard-fails if system is in bad state
        run(log, backend)
    except RuntimeError as exc:
        # startup_checks raises RuntimeError with a human-readable message
        log.critical("Startup validation failed — bot will not start: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.critical("Crypto loop crashed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
