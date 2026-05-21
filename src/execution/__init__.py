"""
execution package — order submission and live order management.

Public interface:

    from execution import (
        OrderRequest, OrderResult,
        execute_signal, close_position,
        OrderMonitor,
    )

Responsibilities:
    - execute_signal()  : translate an approved TradeSignal into a CLOB order,
                          with slippage cap enforcement and up-to-3 requotes.
    - close_position()  : liquidate an existing Position (limit or market mode).
    - OrderMonitor      : background thread that re-quotes or cancels stale GTC
                          orders when the market price drifts away.

All order activity is logged to logs/execution.log.
"""

from .schemas import OrderRequest, OrderResult
from .order   import execute_signal, close_position
from .monitor import OrderMonitor

__all__ = [
    "OrderRequest",
    "OrderResult",
    "execute_signal",
    "close_position",
    "OrderMonitor",
]
