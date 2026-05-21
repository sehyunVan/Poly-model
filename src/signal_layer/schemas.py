"""Signal layer schemas."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class TradeSignal(BaseModel):
    """
    Output of the signal generation step.

    direction:
        "BUY_YES"  — buy the YES token (go long on YES outcome)
        "BUY_NO"   — buy the NO token  (go long on NO outcome)
        "NO_TRADE" — no action; reason field explains why

    base_size_factor:
        Dimensionless scaling factor in [0, 1] proportional to |alpha|.
        The risk module multiplies this by the Kelly-sized capital to get
        the final USDC order size.  A value of 1.0 means full Kelly size.

    Example:
        {
            "market_id": "0xabc",
            "timestamp": "2024-01-15T12:00:00Z",
            "direction": "BUY_YES",
            "alpha": 0.12,
            "base_size_factor": 0.60,
            "reason": "alpha=0.12 above threshold; confidence=0.75"
        }
    """

    market_id: str
    timestamp: datetime
    direction: str = Field(
        ...,
        description="BUY_YES | BUY_NO | NO_TRADE",
    )
    alpha: float = Field(
        ...,
        description="P_R - P_M (signed); positive means model thinks YES is underpriced",
    )
    base_size_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Size scaling factor [0, 1] proportional to |alpha|",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the signal decision (for logs / monitoring)",
    )
    P_M: float = Field(default=0.0, ge=0.0, le=1.0, description="Market-implied probability")
    P_R: float = Field(default=0.0, ge=0.0, le=1.0, description="Model-estimated probability")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
