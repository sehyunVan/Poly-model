"""
Background order monitor.

OrderMonitor tracks live GTC orders in a background thread and re-quotes
them when the market price has moved significantly away from the order price.

Behaviour:
    - Polling sweep runs every 30 seconds.
    - An order is re-quoted when the current best quote deviates > 1 % from
      the original order price.
    - Maximum 3 re-quotes per order; the order is cancelled after that.
    - Orders older than 5 minutes are automatically cancelled (timeout guard).

Usage:
    monitor = OrderMonitor()
    monitor.start()
    ...
    monitor.track(order_id, signal, price=0.63, size=100.0,
                  token_id="...", side="BUY")
    ...
    monitor.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from signal_layer.schemas import TradeSignal

from .order import _cancel_order, _get_clob_client, _get_fill_status, _place_order

_log = logging.getLogger("execution")

_MONITOR_INTERVAL_SEC = 30      # Seconds between each polling sweep
_REQUOTE_THRESHOLD    = 0.01    # Requote when market price deviates > 1 %
_MAX_REQUOTES         = 3       # Maximum requotes per order
_ORDER_TIMEOUT_SEC    = 300     # Auto-cancel orders older than 5 minutes


# ---------------------------------------------------------------------------
# Internal order record
# ---------------------------------------------------------------------------


class _OrderRecord:
    """Mutable state for a single tracked order."""

    __slots__ = (
        "order_id", "signal", "price", "size", "token_id", "side",
        "requote_count", "placed_at",
    )

    def __init__(
        self,
        order_id: str,
        signal: TradeSignal,
        price: float,
        size: float,
        token_id: str,
        side: str,
    ) -> None:
        self.order_id      = order_id
        self.signal        = signal
        self.price         = price
        self.size          = size
        self.token_id      = token_id
        self.side          = side
        self.requote_count = 0
        self.placed_at     = time.monotonic()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class OrderMonitor:
    """
    Background daemon that monitors live GTC orders and re-quotes on drift.

    Public methods:
        track(order_id, signal, price, size, token_id, side)
            Register a new live order for monitoring.
        cancel_and_report(order_id)
            Forcibly cancel an order and remove it from tracking.
        start()  /  stop()
            Manage the background thread lifecycle.
    """

    def __init__(self) -> None:
        self._orders: dict[str, _OrderRecord] = {}
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def track(
        self,
        order_id: str,
        signal: TradeSignal,
        price: float,
        size: float,
        token_id: str,
        side: str,
    ) -> None:
        """Register order_id for background monitoring."""
        record = _OrderRecord(order_id, signal, price, size, token_id, side)
        with self._lock:
            self._orders[order_id] = record
        _log.info(
            "OrderMonitor.track: order %s  market=%s  price=%.4f  size=%.2f",
            order_id, signal.market_id, price, size,
        )

    def cancel_and_report(self, order_id: str) -> None:
        """Cancel order_id via the CLOB API and remove it from the tracking table."""
        client = _get_clob_client()
        if client:
            _cancel_order(client, order_id)
        with self._lock:
            rec = self._orders.pop(order_id, None)
        if rec:
            _log.info(
                "OrderMonitor.cancel_and_report: removed order %s  "
                "market=%s  requotes=%d",
                order_id, rec.signal.market_id, rec.requote_count,
            )

    def start(self) -> None:
        """Start the background polling thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="OrderMonitor", daemon=True,
        )
        self._thread.start()
        _log.info("OrderMonitor: started  sweep_interval=%ds", _MONITOR_INTERVAL_SEC)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=_MONITOR_INTERVAL_SEC + 5)
        _log.info("OrderMonitor: stopped")

    @property
    def tracked_count(self) -> int:
        """Number of orders currently under monitoring."""
        with self._lock:
            return len(self._orders)

    # -----------------------------------------------------------------------
    # Background loop
    # -----------------------------------------------------------------------

    def _run(self) -> None:
        """Main polling loop executed in the background thread."""
        while not self._stop.is_set():
            try:
                self._sweep()
            except Exception as exc:
                _log.error("OrderMonitor sweep error: %s", exc)
            # Sleep in short increments so stop() is responsive
            self._stop.wait(timeout=_MONITOR_INTERVAL_SEC)

    def _sweep(self) -> None:
        """Evaluate every tracked order once."""
        client = _get_clob_client()
        if client is None:
            return

        with self._lock:
            order_ids = list(self._orders.keys())

        for oid in order_ids:
            with self._lock:
                rec = self._orders.get(oid)
            if rec is None:
                continue  # already removed by a concurrent operation
            self._check_order(client, rec)

    def _check_order(self, client, rec: _OrderRecord) -> None:
        """
        Evaluate a single order: remove if filled/cancelled, requote on drift,
        or cancel on timeout / max-requote exhaustion.
        """
        from data.market import get_orderbook

        oid = rec.order_id

        # 1. Check fill / external cancel
        fill_info = _get_fill_status(client, oid)
        status    = fill_info.get("status", "UNKNOWN")

        if status == "MATCHED":
            _log.info("OrderMonitor: order %s FILLED — removing", oid)
            with self._lock:
                self._orders.pop(oid, None)
            return

        if status == "CANCELLED":
            _log.info("OrderMonitor: order %s CANCELLED externally — removing", oid)
            with self._lock:
                self._orders.pop(oid, None)
            return

        # 2. Age-based timeout
        age = time.monotonic() - rec.placed_at
        if age > _ORDER_TIMEOUT_SEC:
            _log.warning(
                "OrderMonitor: order %s timed out after %.0fs — cancelling",
                oid, age,
            )
            self.cancel_and_report(oid)
            return

        # 3. Check price drift
        ob = get_orderbook(rec.signal.market_id)
        if rec.side == "BUY":
            current_best = ob.asks[0].price if ob.asks else None
        else:
            current_best = ob.bids[0].price if ob.bids else None

        if current_best is None:
            return  # Empty book — defer to next sweep

        deviation = abs(current_best - rec.price) / max(rec.price, 1e-9)
        if deviation <= _REQUOTE_THRESHOLD:
            return  # Price still acceptable

        # 4. Requote or give up
        if rec.requote_count >= _MAX_REQUOTES:
            _log.warning(
                "OrderMonitor: order %s exceeded max requotes (%d) — cancelling",
                oid, _MAX_REQUOTES,
            )
            self.cancel_and_report(oid)
            return

        # Cancel the stale order
        _cancel_order(client, oid)
        rec.requote_count += 1
        new_price = current_best

        _log.info(
            "OrderMonitor: requoting order %s  attempt=%d/%d  "
            "old_price=%.4f  new_price=%.4f",
            oid, rec.requote_count, _MAX_REQUOTES, rec.price, new_price,
        )

        # Place fresh order at the updated price
        new_oid = _place_order(client, rec.token_id, new_price, rec.size, rec.side)
        if new_oid is None:
            _log.error(
                "OrderMonitor: requote place_order failed for market %s",
                rec.signal.market_id,
            )
            with self._lock:
                self._orders.pop(oid, None)
            return

        # Swap the record: keep original placement time for the timeout clock
        with self._lock:
            new_rec = _OrderRecord(
                new_oid, rec.signal, new_price, rec.size, rec.token_id, rec.side,
            )
            new_rec.requote_count = rec.requote_count
            new_rec.placed_at     = rec.placed_at   # preserve original clock
            self._orders.pop(oid, None)
            self._orders[new_oid] = new_rec
