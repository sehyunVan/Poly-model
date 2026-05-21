"""
virtual — paper trading overlay for the Polymarket bot.

Enabled by setting VIRTUAL_MODE=true in .env.
All state is stored separately from real trading state.
"""

from virtual.portfolio import (   # noqa: F401
    VirtualPosition,
    VirtualPortfolio,
    load_virtual_portfolio,
    save_virtual_portfolio,
    portfolio_to_risk_portfolio,
)
from virtual.execution import simulate_fill, simulate_position_exit  # noqa: F401
from virtual.settler   import settle_resolved_positions  # noqa: F401
from virtual.auto_tune import auto_apply_suggestions  # noqa: F401
