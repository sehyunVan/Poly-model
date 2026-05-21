"""Risk layer shared schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """
    A single open position (YES or NO token holding) in one market.

    unrealized_pnl is computed as:
        (current_price - avg_entry_price) / avg_entry_price * size

    group_id links correlated positions (e.g. multiple markets on the same
    election).  Positions with the same group_id are summed when checking the
    correlated-group exposure limit.

    Example:
        {
            "market_id": "0xabc",
            "token_id": "111222",
            "side": "YES",
            "size": 250.0,
            "avg_entry_price": 0.62,
            "current_price": 0.68,
            "unrealized_pnl": 24.19,
            "category": "politics",
            "group_id": "us_election_2024"
        }
    """

    market_id: str
    token_id: str
    side: str = Field(..., description="YES | NO")
    size: float = Field(..., ge=0.0, description="USDC-denominated cost basis")
    avg_entry_price: float = Field(..., gt=0.0, le=1.0)
    current_price: float = Field(..., ge=0.0, le=1.0)
    unrealized_pnl: float = Field(default=0.0)
    category: str = Field(default="other")
    group_id: Optional[str] = Field(
        default=None,
        description="Correlated-group identifier; None means no grouping",
    )

    def recalculate_pnl(self) -> "Position":
        """Return a copy with unrealized_pnl recomputed from current fields."""
        if self.avg_entry_price > 0:
            pnl = (self.current_price - self.avg_entry_price) / self.avg_entry_price * self.size
        else:
            pnl = 0.0
        return self.model_copy(update={"unrealized_pnl": round(pnl, 6)})


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio(BaseModel):
    """
    Snapshot of the full account state.

    total_capital    — total USDC (deployed + available).
    available_capital — USDC not currently tied up in positions.
    daily_pnl        — realised + unrealised PnL since midnight UTC.
    weekly_pnl       — same over the rolling 7-day window.

    Example:
        {
            "positions": [...],
            "total_capital": 10000.0,
            "available_capital": 7500.0,
            "daily_pnl": -120.0,
            "weekly_pnl": 380.0
        }
    """

    positions: list[Position] = Field(default_factory=list)
    total_capital: float = Field(..., gt=0.0)
    available_capital: float = Field(..., ge=0.0)
    daily_pnl: float = Field(default=0.0)
    weekly_pnl: float = Field(default=0.0)

    def exposure_for_market(self, market_id: str) -> float:
        """Net USDC exposure for a market.

        Returns max(YES-side total, NO-side total) rather than their sum.
        This correctly treats opposing positions as a hedge — holding $50 YES
        and $50 NO in the same market produces $50 net exposure, not $100.
        """
        yes_size = sum(
            p.size for p in self.positions
            if p.market_id == market_id and p.side == "YES"
        )
        no_size = sum(
            p.size for p in self.positions
            if p.market_id == market_id and p.side == "NO"
        )
        return max(yes_size, no_size)

    def exposure_for_category(self, category: str) -> float:
        """Total USDC size across all positions in a category."""
        return sum(p.size for p in self.positions if p.category == category)

    def exposure_for_group(self, group_id: str) -> float:
        """Total USDC size for a correlated group."""
        return sum(p.size for p in self.positions if p.group_id == group_id)

    def total_deployed(self) -> float:
        return sum(p.size for p in self.positions)


# ---------------------------------------------------------------------------
# RiskLimits  (loaded from config/risk_limits.yaml)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path("config/risk_limits.yaml")

_DEFAULT_CATEGORY_PCT = {
    "politics": 0.40,
    "crypto":   0.30,
    "sports":   0.20,
    "other":    0.20,
}


class RiskLimits(BaseModel):
    """
    All configurable risk thresholds.

    Load from YAML via RiskLimits.from_config(); defaults match the YAML file.
    """

    # Concentration limits
    max_single_event_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    max_category_pct: dict = Field(default_factory=lambda: dict(_DEFAULT_CATEGORY_PCT))
    max_correlated_group_pct: float = Field(default=0.25, ge=0.0, le=1.0)

    # Loss limits
    max_daily_loss_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    max_weekly_loss_pct: float = Field(default=0.10, ge=0.0, le=1.0)

    # Kelly sizing
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    max_bet_pct: float = Field(default=0.05, ge=0.0, le=1.0)

    # Slippage
    max_slippage_pct: float = Field(default=0.01, ge=0.0, le=1.0)

    # Stop-loss for virtual (paper) trading only.
    # Close a position when unrealized loss >= this fraction of its cost basis.
    # 0.0 disables the stop-loss.
    virtual_stop_loss_pct: float = Field(default=0.80, ge=0.0, le=1.0)

    @classmethod
    def from_config(cls, path: str | Path = _CONFIG_PATH) -> "RiskLimits":
        """Load limits from a YAML file, falling back to defaults on error."""
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except FileNotFoundError:
            return cls()
        except Exception as exc:
            print(f"[risk.schemas] Failed to load {path}: {exc}. Using defaults.")
            return cls()

    def category_limit(self, category: str) -> float:
        """Return the max fraction for the given category, defaulting to 'other'."""
        return self.max_category_pct.get(category, self.max_category_pct.get("other", 0.20))
