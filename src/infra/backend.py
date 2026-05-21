"""
infra/backend.py — ExecutionBackend protocol and concrete implementations.

Motivation
----------
The VIRTUAL_MODE flag is scattered across loop.py in at least six places:

  - Conditional imports at module load time (lines 142–161)
  - if VIRTUAL_MODE: ... else: ... around order placement (line 1380)
  - if not VIRTUAL_MODE: ... around balance sync (lines 915–948)
  - State file path selection (_VIRTUAL_STATE, line 140)
  - Dashboard state routing
  - logging ("mode=LIVE" vs "mode=VIRTUAL")

This creates accidental coupling between business logic and deployment mode.
A wrong VIRTUAL_MODE check is silent — it won't error, it just won't trade
(or worse, will trade with the wrong state).

This module introduces an ExecutionBackend Protocol so loop.py receives a
backend object at startup and never queries VIRTUAL_MODE again.

Protocol
--------
Any object that implements the three methods is a valid ExecutionBackend.
Python's `typing.Protocol` is structural (duck-typed) — no inheritance needed.

Implementations
---------------
  VirtualBackend  — paper trading using CLOB ask for fill price
  LiveBackend     — real CLOB order placement via crypto.execution

Migration path
--------------
This module provides the protocol and concrete classes.  loop.py still reads
VIRTUAL_MODE at startup (in crypto_main.py) to choose which backend to
instantiate — that one switch point is the only remaining mode check.

The inline `if VIRTUAL_MODE:` blocks inside `run()` should be replaced with
backend method calls.  That refactor is intentionally left as a follow-on
step so this change can be reviewed and tested independently.

Usage (future loop.py)
----------------------
    from infra.backend import VirtualBackend, LiveBackend

    backend = LiveBackend(log) if not VIRTUAL_MODE else VirtualBackend()

    # Inside the loop:
    result = backend.place_order(token_id, bet_size, market_id, clob_ask)
    balance = backend.get_balance()
    state_path = backend.state_path()
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infra.types import ClobTokenId, ConditionId  # noqa: E402


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class ExecutionBackend(Protocol):
    """
    Minimal interface for order execution and balance queries.

    Both VirtualBackend and LiveBackend implement this protocol.
    loop.py depends on this interface, not on a concrete class.

    The @runtime_checkable decorator allows isinstance() checks in tests.
    """

    def place_order(
        self,
        token_id: ClobTokenId,
        bet_size_usdc: float,
        market_id: ConditionId,
        log: logging.Logger,
        price_hint: Optional[float] = None,
        band_min: float = 0.0,
        band_max: float = 1.0,
        maker_mode: bool = True,
    ) -> dict:
        """
        Place a BUY order for `bet_size_usdc` USDC of `token_id`.

        Returns a result dict with keys:
            status      : "FILLED" | "NO_FILL"
            fill_price  : float  (0.0 if not filled)
            filled_usdc : float  (0.0 if not filled)
            order_id    : str | None
            fill_type   : "MAKER" | "TAKER" | "NONE"
            reason      : str
        """
        ...

    def get_balance(self, log: logging.Logger) -> Optional[float]:
        """
        Return current USDC balance for this loop's share of the wallet.
        Returns None on failure.
        """
        ...

    def state_path(self) -> Path:
        """Return the path to the state file for this backend."""
        ...


# ── VirtualBackend ────────────────────────────────────────────────────────────

class VirtualBackend:
    """
    Paper trading backend.

    place_order() returns an immediate synthetic fill at price_hint (CLOB ask).
    get_balance() returns None — balance is tracked in VirtualPortfolio.
    state_path() returns data/<state_filename> (default: virtual_state.json).
    """

    _ROOT = Path(__file__).resolve().parents[2]  # poly-model/

    def __init__(self, state_filename: str = "virtual_state.json") -> None:
        self._state_filename = state_filename

    def place_order(
        self,
        token_id: ClobTokenId,
        bet_size_usdc: float,
        market_id: ConditionId,
        log: logging.Logger,
        price_hint: Optional[float] = None,
        band_min: float = 0.0,
        band_max: float = 1.0,
        maker_mode: bool = True,
    ) -> dict:
        """
        Synthetic fill at price_hint (CLOB ask).

        Returns FILLED immediately — paper trades always fill.
        maker_mode is ignored in virtual mode (no real book to post into).
        """
        fill_price = price_hint if price_hint is not None else (band_min + band_max) / 2
        log.debug(
            "VirtualBackend.place_order: synthetic fill  token=%s  price=%.4f  usdc=$%.2f",
            str(token_id)[:16], fill_price, bet_size_usdc,
        )
        return {
            "status":      "FILLED",
            "fill_price":  round(fill_price, 6),
            "filled_usdc": round(bet_size_usdc, 4),
            "order_id":    None,
            "fill_type":   "VIRTUAL",
            "reason":      "virtual fill",
        }

    def get_balance(self, log: logging.Logger) -> Optional[float]:
        """Not applicable for virtual backend — balance tracked in portfolio."""
        return None

    def state_path(self) -> Path:
        return self._ROOT / "data" / self._state_filename


# ── LiveBackend ───────────────────────────────────────────────────────────────

class LiveBackend:
    """
    Real CLOB order placement backend.

    Delegates to crypto.execution.place_crypto_order() and
    crypto.execution.get_clob_balance().

    state_path() returns data/real_state.json so live state is always
    separate from paper state — prevents the "dashboard mixed virtual history
    with real state" incident.
    """

    _ROOT = Path(__file__).resolve().parents[2]  # poly-model/

    def __init__(self, log: logging.Logger) -> None:
        try:
            from crypto.execution import (          # type: ignore
                place_crypto_order as _place,
                get_clob_balance   as _balance,
            )
            self._place   = _place
            self._balance = _balance
        except ImportError as exc:
            raise RuntimeError(
                f"LiveBackend: cannot import crypto.execution — {exc}. "
                "Check that KEY/FUNDER are set and py_clob_client is installed."
            ) from exc

    def place_order(
        self,
        token_id: ClobTokenId,
        bet_size_usdc: float,
        market_id: ConditionId,
        log: logging.Logger,
        price_hint: Optional[float] = None,
        band_min: float = 0.0,
        band_max: float = 1.0,
        maker_mode: bool = True,
    ) -> dict:
        """Delegate to place_crypto_order() in crypto.execution."""
        return self._place(
            token_id, bet_size_usdc, market_id, log,
            price_hint=price_hint,
            band_min=band_min,
            band_max=band_max,
            maker_mode=maker_mode,
        )

    def get_balance(self, log: logging.Logger) -> Optional[float]:
        """Delegate to get_clob_balance() in crypto.execution."""
        return self._balance(log)

    def state_path(self) -> Path:
        return self._ROOT / "data" / "real_state.json"


# ── Factory ───────────────────────────────────────────────────────────────────

def make_backend(
    virtual_mode: bool,
    log: logging.Logger,
    virtual_state_file: str = "virtual_state.json",
) -> ExecutionBackend:
    """
    Construct the appropriate backend from the VIRTUAL_MODE flag.

    This is the ONLY place in the codebase that should inspect virtual_mode.
    All other code receives an ExecutionBackend and never checks mode directly.

    virtual_state_file: overrides the default state filename for virtual mode.
    Used by non-5m loops to maintain separate state (e.g. "15m_virtual_state.json").

    Usage in crypto_main.py:
        from infra.backend import make_backend
        backend = make_backend(VIRTUAL_MODE, log)
        run(log, backend)
    """
    if virtual_mode:
        log.info("ExecutionBackend: VirtualBackend (paper trading, state=%s)", virtual_state_file)
        return VirtualBackend(state_filename=virtual_state_file)
    else:
        log.info("ExecutionBackend: LiveBackend (LIVE trading — real CLOB orders)")
        return LiveBackend(log)
