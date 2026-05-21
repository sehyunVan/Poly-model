"""
scripts/verify_server.py — Post-deploy health check.

Run this on the server after deploying any file changes to confirm the system
is in a good state before the bots restart.

Usage (from local machine):
    ssh -i "$SSH_KEY" ubuntu@"$HOST" \
        "cd ~/poly-model && source .venv/bin/activate && python scripts/verify_server.py"

Checks:
  1. Required Python packages are importable.
  2. .env has KEY and FUNDER set.
  3. VIRTUAL_MODE setting is correct (warns if unexpectedly true/false).
  4. Screen sessions: crypto, swarm, hl, dashboard are running.
  5. State files exist and are valid JSON (not corrupted).
  6. Bot source files are importable (catches syntax errors before restart).
  7. Dashboard port 8765 is bound (or warns if not).
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

PASS = "  [PASS]"
WARN = "  [WARN]"
FAIL = "  [FAIL]"

_errors = 0
_warnings = 0


def check(label: str, ok: bool, msg: str = "", warn_only: bool = False) -> None:
    global _errors, _warnings
    if ok:
        print(f"{PASS} {label}")
    else:
        tag = WARN if warn_only else FAIL
        print(f"{tag} {label}" + (f": {msg}" if msg else ""))
        if warn_only:
            _warnings += 1
        else:
            _errors += 1


# ── 1. Required packages ──────────────────────────────────────────────────────
print("\n── 1. Required packages ─────────────────────────────────────────────")
for pkg in ["httpx", "yaml", "numpy", "pydantic", "dotenv"]:
    try:
        importlib.import_module(pkg if pkg != "dotenv" else "dotenv")
        check(f"import {pkg}", True)
    except ImportError as exc:
        check(f"import {pkg}", False, str(exc))

try:
    import pyarrow  # noqa: F401
    check("import pyarrow", True)
except ImportError:
    check("import pyarrow", False, "pip install pyarrow", warn_only=True)

try:
    from py_clob_client.client import ClobClient  # noqa: F401
    check("import py_clob_client", True)
except ImportError as exc:
    check("import py_clob_client", False, str(exc))

try:
    from web3 import Web3  # noqa: F401
    check("import web3", True)
except ImportError:
    check("import web3", False, "pip install web3", warn_only=True)


# ── 2. .env keys ─────────────────────────────────────────────────────────────
print("\n── 2. .env configuration ────────────────────────────────────────────")
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

check("KEY set in .env",    bool(os.getenv("KEY")),    ".env missing KEY")
check("FUNDER set in .env", bool(os.getenv("FUNDER")), ".env missing FUNDER")

virtual_mode_raw = os.getenv("VIRTUAL_MODE", "true").lower()
virtual_mode = virtual_mode_raw not in ("false", "0", "no")
check(
    f"VIRTUAL_MODE={virtual_mode_raw!r}",
    True,  # always pass — just display current value
    warn_only=True,
)
if virtual_mode:
    print(f"{WARN}  VIRTUAL_MODE is currently TRUE — bot will paper trade, not live")
else:
    print(f"{PASS}  VIRTUAL_MODE=false — LIVE trading mode confirmed")

ntfy_topic = os.getenv("NTFY_TOPIC", "")
check("NTFY_TOPIC set", bool(ntfy_topic), ".env missing NTFY_TOPIC", warn_only=True)


# ── 3. Screen sessions ────────────────────────────────────────────────────────
print("\n── 3. Screen sessions ───────────────────────────────────────────────")
try:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    for session in ["crypto", "swarm", "hl", "dashboard"]:
        running = session in output
        check(f"screen -S {session}", running, "not running", warn_only=(session == "dashboard"))
except FileNotFoundError:
    check("screen command", False, "screen not found — are you on the server?")


# ── 4. State files ───────────────────────────────────────────────────────────
print("\n── 4. State files ───────────────────────────────────────────────────")
state_files = [
    _ROOT / "data" / "real_state.json",
    _ROOT / "data" / "virtual_state.json",
]
for sf in state_files:
    if not sf.exists():
        check(f"{sf.name} exists", False, "file missing — will be created on first run",
              warn_only=True)
        continue
    try:
        raw = json.loads(sf.read_text(encoding="utf-8"))
        check(f"{sf.name} valid JSON", True)
        # Check for obviously corrupted pnl_history
        pnl_sum = sum(e.get("pnl", 0) for e in raw.get("pnl_history", []))
        budget = raw.get("initial_budget", 1000.0)
        if pnl_sum < -(budget * 0.90):
            check(
                f"{sf.name} pnl_history", False,
                f"cumulative PnL ${pnl_sum:.2f} < -90% of budget ${budget:.2f} "
                f"— likely corrupt (trading-halted incident pattern)",
            )
        else:
            check(f"{sf.name} pnl_history", True)
    except Exception as exc:
        check(f"{sf.name} valid JSON", False, str(exc))


# ── 5. Source modules importable ─────────────────────────────────────────────
print("\n── 5. Source module imports ─────────────────────────────────────────")
modules_to_check = [
    ("crypto.loop",      "src/crypto/loop.py"),
    ("crypto.execution", "src/crypto/execution.py"),
    ("crypto.redeem",    "src/crypto/redeem.py"),
    ("infra.types",      "src/infra/types.py"),
    ("infra.http_client","src/infra/http_client.py"),
    ("infra.accounting", "src/infra/accounting.py"),
    ("infra.backend",    "src/infra/backend.py"),
    ("virtual.portfolio","src/virtual/portfolio.py"),
]
for mod, path in modules_to_check:
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as exc:
        check(f"import {mod}", False, f"{path}: {exc}")


# ── 6. Dashboard port ────────────────────────────────────────────────────────
print("\n── 6. Dashboard port 8765 ───────────────────────────────────────────")
try:
    result = subprocess.run(
        ["ss", "-tlnp"],
        capture_output=True, text=True,
    )
    port_bound = "8765" in result.stdout
    check("port 8765 bound", port_bound, "dashboard may not be running", warn_only=True)
except Exception as exc:
    check("port check", False, str(exc), warn_only=True)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────────────────")
if _errors == 0 and _warnings == 0:
    print("  ALL CHECKS PASSED — safe to restart bots.")
elif _errors == 0:
    print(f"  PASSED with {_warnings} warning(s) — review above before restarting.")
else:
    print(f"  {_errors} FAILURE(S), {_warnings} WARNING(S) — fix issues before restarting.")
    sys.exit(1)
