"""
infra/http_client.py — Canonical timeout wrapper for blocking external calls.

Root cause of past incidents
-----------------------------
The 2026-03-18 90-minute hang and the similar hang in crypto/execution.py
both had the same cause: py_clob_client makes blocking TCP calls with no
timeout.  The fix was applied in three separate places with three slightly
different implementations:

  - _clob_call()    in src/data/market.py     (timeout=15s, raises TimeoutError)
  - _clob_timeout() in src/crypto/execution.py (timeout=10s, raises TimeoutError)
  - _clob_timeout() in bot/execution.py        (timeout=15s, raises TimeoutError)

This module is the single canonical implementation.  Import from here.
The three original implementations are left in place for now (to avoid a
simultaneous multi-file change on a live system), but new call sites should
use call_with_timeout() directly.

Usage
-----
    from infra.http_client import call_with_timeout

    result = call_with_timeout(client.get_balance_allowance, params, timeout=10)
    # raises concurrent.futures.TimeoutError on timeout
    # raises the original exception on API error (not swallowed)

Default timeouts (in seconds)
------------------------------
  CLOB_TIMEOUT  = 10   — py_clob_client calls (order placement, fill poll, balance)
  DATA_TIMEOUT  = 15   — data API calls (positions, markets, history)
  HTTP_TIMEOUT  = 8    — httpx REST calls (Gamma API, CLOB book fetch)

These are conservative: Polymarket CLOB calls normally complete in < 2s.
A 10s limit catches hangs without triggering false positives on slow markets.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Default timeouts — all call sites should use these unless they have a
# specific reason to deviate (and if they do, the reason should be in a comment).
CLOB_TIMEOUT = 10   # seconds — py_clob_client order/balance/fill calls
DATA_TIMEOUT = 15   # seconds — Polymarket data API / Gamma API REST calls
HTTP_TIMEOUT = 8    # seconds — httpx REST client default


def call_with_timeout(fn: Callable[..., T], *args: Any, timeout: int = CLOB_TIMEOUT) -> T:
    """
    Run fn(*args) in a worker thread with a hard deadline.

    This is the canonical replacement for:
      - _clob_call()    in src/data/market.py
      - _clob_timeout() in src/crypto/execution.py
      - _clob_timeout() in bot/execution.py

    Why a thread instead of asyncio.wait_for?
    py_clob_client is synchronous and uses the requests library internally.
    It cannot be awaited.  Running it in a thread is the correct way to apply
    a deadline to synchronous blocking I/O.

    Args:
        fn      : callable to invoke
        *args   : positional arguments forwarded to fn
        timeout : hard deadline in seconds (default: CLOB_TIMEOUT = 10s)

    Returns:
        Return value of fn(*args).

    Raises:
        concurrent.futures.TimeoutError  — if fn does not complete within timeout
        <whatever fn raises>             — propagated unchanged if fn raises before timeout
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        return future.result(timeout=timeout)
