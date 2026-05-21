"""
Execution layer Pydantic schemas.

OrderRequest  — parameters for a single order submission.
OrderResult   — outcome of an execution attempt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    """Parameters for a single order submission."""

    market_id: str
    token_id: str            # YES or NO token ID obtained from market info
    side: str                # "BUY" or "SELL"
    price: float             # Limit price in [0.01, 0.99]
    size: float              # USDC size to trade
    order_type: str = "GTC"  # "GTC" (Good-Till-Cancelled) or "FOK" (Fill-or-Kill)


class OrderResult(BaseModel):
    """Outcome of an execution attempt (signal execution or position close)."""

    # Exchange-assigned order ID; None when the order was never submitted.
    order_id: Optional[str] = None

    # Final status of this execution attempt.
    # FILLED           — fully matched by the exchange
    # PARTIALLY_FILLED — partially matched before timeout/cancel
    # PENDING          — submitted but fill not yet confirmed
    # CANCELLED        — cancelled by the system (requote or deleverage)
    # NO_FILL          — all requote attempts exhausted without a fill
    # ERROR            — order could not be submitted (API / credential error)
    status: str

    filled_size: float = 0.0       # USDC amount actually filled
    avg_fill_price: float = 0.0    # Weighted average fill price (0 if unfilled)

    market_id: str
    token_id: str = ""
    side: str = ""
    requested_size: float = 0.0    # Original size passed to execute_signal()
    attempts: int = 0              # Total order submission attempts made
    reason: str = ""               # Human-readable explanation of the outcome

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
