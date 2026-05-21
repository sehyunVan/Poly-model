"""
Virtual portfolio state management.

VirtualPortfolio tracks simulated positions and balance for paper trading.
It is persisted as JSON at data/virtual_state.json (configurable).

portfolio_to_risk_portfolio() converts a VirtualPortfolio into the real
Portfolio type so that all existing risk checks work unchanged.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# ── Path setup (allow import from either src/ or project root) ───────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from risk.schemas import Portfolio, Position

import contextlib


@contextlib.contextmanager
def _portfolio_lock(path: Path):
    """Exclusive file-level advisory lock (cross-process safe on Linux).

    Uses fcntl.flock() on Linux/macOS so separate processes (main bot and
    crypto loop) serialise their reads and writes.  No-op on Windows where
    both processes do not run simultaneously.
    """
    if sys.platform == "win32":
        yield
        return
    try:
        import fcntl
        lock_path = path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as _fd:
            fcntl.flock(_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(_fd, fcntl.LOCK_UN)
    except ImportError:
        yield  # fcntl unavailable (shouldn't happen on Linux)


# ── VirtualPosition ───────────────────────────────────────────────────────────

class VirtualPosition(BaseModel):
    """
    A single simulated position.

    outcome is set to 1 (YES won) or 0 (NO won) when the market is settled
    by settler.settle_resolved_positions(). While open it is None.

    realized_pnl is computed on settlement:
        YES bet: pnl = size * (1 - fill_price) / fill_price   if YES won
                      = -size                                  if NO  won
        NO  bet: pnl = size * fill_price / (1 - fill_price)   if NO  won
                      = -size                                  if YES won
    """

    market_id: str              # ConditionId — hex condition hash (not a ClobTokenId)
    title: str
    direction: str              # "YES" | "NO"
    size_usdc: float            # USDC spent (cost basis)
    fill_price: float           # Simulated fill price [0, 1]
    fill_time: datetime
    category: str = "other"

    outcome: Optional[int] = None           # 1/0 once settled
    realized_pnl: Optional[float] = None    # set on settlement
    settle_time: Optional[datetime] = None  # UTC timestamp when position was settled
    bought_token_id: Optional[str] = None   # CLOB token purchased; used for mid-window exit


# ── VirtualPortfolio ──────────────────────────────────────────────────────────

class VirtualPortfolio(BaseModel):
    """
    Full virtual account state.

    pnl_history entries:
        {"date": "YYYY-MM-DD", "pnl": float, "cumulative_pnl": float}
    """

    initial_budget:    float = Field(default=1000.0, gt=0)
    available_usdc:    float = Field(default=1000.0, ge=0)
    # real_clob_balance: literal on-chain USDC.e, updated at C1 startup sync and
    # after each AR redemption. available_usdc grows above this as winning positions
    # settle (CTF tokens credited virtually but not yet redeemed to USDC).
    # Dashboard uses: unredeemed_ctf = available_usdc - real_clob_balance.
    real_clob_balance: float = Field(default=0.0, ge=0)
    positions:         list[VirtualPosition] = Field(default_factory=list)
    closed_positions:  list[VirtualPosition] = Field(default_factory=list)
    pnl_history:       list[dict]            = Field(default_factory=list)
    start_date:        datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Real PnL tracking (live mode only) ────────────────────────────────────
    # initial_real_clob_balance: first real CLOB balance at bot startup.
    # Set once; never overwritten so it always reflects the true starting point.
    initial_real_clob_balance: float = Field(default=0.0, ge=0)
    # total_detected_deposits: cumulative external deposits detected since start.
    # A deposit is detected when real_clob_balance jumps by more than
    # MAX_BET_ABS × 4 in one balance-sync cycle (can't be explained by a single win).
    total_detected_deposits: float = Field(default=0.0, ge=0)
    # detected_deposits: audit log of each detected deposit event.
    detected_deposits: list[dict] = Field(default_factory=list)
    # pending_ctf_usdc: value of winning CTF positions not yet redeemed on-chain.
    # Updated each AR cycle by querying the Polymarket positions API.
    # Included in real_pnl_all_time() so PnL is accurate even before redemption.
    pending_ctf_usdc: float = Field(default=0.0, ge=0)

    def mark_updated(self) -> None:
        self.last_updated = datetime.now(timezone.utc)

    def close_position_manually(self, pos: "VirtualPosition", realized_pnl: float) -> None:
        """Close an open position early (e.g. mid-window SELL) with a given PnL.
        Moves it from positions → closed_positions and returns the proceeds to available_usdc.
        """
        pos.outcome = 1 if realized_pnl >= 0 else 0
        pos.realized_pnl = round(realized_pnl, 6)
        if pos in self.positions:
            self.positions.remove(pos)
        self.closed_positions.append(pos)
        self.available_usdc = round(self.available_usdc + pos.size_usdc + realized_pnl, 6)
        self.mark_updated()

    def total_capital(self) -> float:
        """Available + deployed USDC (cost basis of open positions)."""
        deployed = sum(p.size_usdc for p in self.positions)
        return self.available_usdc + deployed

    def daily_pnl(self) -> float:
        """Realized PnL for the current UTC date."""
        today = datetime.now(timezone.utc).date().isoformat()
        for entry in reversed(self.pnl_history):
            if entry.get("date") == today:
                return entry["pnl"]
        return 0.0

    def weekly_pnl(self) -> float:
        """Realized PnL over the last 7 days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        return sum(e["pnl"] for e in self.pnl_history if e.get("date", "") >= cutoff)

    def cumulative_pnl(self) -> float:
        """Total realized PnL since start."""
        return sum(e["pnl"] for e in self.pnl_history)

    def real_pnl_all_time(self) -> float:
        """
        True cash PnL since live trading started.

        Canonical formula is documented in infra/accounting.py
        (AccountingLedger.real_pnl).  This method delegates to it so the
        formula is defined in exactly one place.

        Returns 0.0 if initial_real_clob_balance was never set (virtual mode).
        """
        if self.initial_real_clob_balance == 0.0:
            return 0.0
        try:
            from infra.accounting import AccountingLedger  # avoid circular at module load
            ledger = AccountingLedger(
                clob_balance      = self.real_clob_balance,
                pending_ctf_usdc  = self.pending_ctf_usdc,
                initial_clob      = self.initial_real_clob_balance,
                detected_deposits = self.total_detected_deposits,
                pending_credit    = 0.0,
                wallet_share      = 1.0,   # real_pnl is on full balance, not crypto share
                max_bet_abs       = 0.0,
            )
            return ledger.real_pnl()
        except ImportError:
            # Fallback if infra package not on path (e.g. isolated test)
            return (
                self.real_clob_balance
                + self.pending_ctf_usdc
                - self.initial_real_clob_balance
                - self.total_detected_deposits
            )


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_unlocked(vp: VirtualPortfolio, p: Path) -> None:
    """Write VirtualPortfolio to disk without acquiring the lock.
    Must only be called when the caller already holds _portfolio_lock(p).
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(vp.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)


def load_virtual_portfolio(
    path: str | Path,
    initial_budget: float = 1000.0,
) -> VirtualPortfolio:
    """
    Load VirtualPortfolio from JSON, or create a fresh one if the file does
    not exist.
    """
    p = Path(path)
    with _portfolio_lock(p):
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                return VirtualPortfolio.model_validate(raw)
            except Exception as exc:
                print(f"[virtual.portfolio] Failed to load {p}: {exc} — starting fresh")

        vp = VirtualPortfolio(
            initial_budget=initial_budget,
            available_usdc=initial_budget,
        )
        _save_unlocked(vp, p)
        return vp


def save_virtual_portfolio(vp: VirtualPortfolio, path: str | Path) -> None:
    """Persist VirtualPortfolio to JSON (atomic write + cross-process lock)."""
    p = Path(path)
    with _portfolio_lock(p):
        _save_unlocked(vp, p)


# ── Conversion to real Portfolio (for risk checks) ────────────────────────────

def portfolio_to_risk_portfolio(vp: VirtualPortfolio) -> Portfolio:
    """
    Convert a VirtualPortfolio to the real Portfolio type so that
    check_exposure_limits(), compute_position_size(), and
    evaluate_portfolio_risk() all work without modification.

    current_price for open positions is fetched from the live orderbook;
    fill_price is used as fallback if the API is unavailable.
    """
    # Lazy import to avoid circular import at module load time
    try:
        from data.market import get_current_price  # type: ignore
    except ImportError:
        get_current_price = None

    positions: list[Position] = []
    for vp_pos in vp.positions:
        # Attempt live price; fallback to fill price
        current_price = vp_pos.fill_price
        if get_current_price is not None:
            try:
                live = get_current_price(vp_pos.market_id)
                if live is not None and 0.0 < live < 1.0:
                    current_price = live
            except Exception:
                pass

        # Adjust current_price for NO positions (we track YES price but hold NO)
        if vp_pos.direction == "NO":
            adjusted_current = 1.0 - current_price
            adjusted_entry   = 1.0 - vp_pos.fill_price
        else:
            adjusted_current = current_price
            adjusted_entry   = vp_pos.fill_price

        # Avoid division by zero
        entry = max(adjusted_entry, 0.001)
        unrealized = (adjusted_current - entry) / entry * vp_pos.size_usdc

        positions.append(Position(
            market_id=vp_pos.market_id,
            token_id="",                      # not tracked for virtual positions
            side=vp_pos.direction,
            size=vp_pos.size_usdc,
            avg_entry_price=max(min(entry, 0.999), 0.001),
            current_price=max(min(adjusted_current, 1.0), 0.0),
            unrealized_pnl=round(unrealized, 6),
            category=vp_pos.category,
        ))

    total_cap = max(vp.total_capital(), 0.01)   # guard against zero

    return Portfolio(
        positions=positions,
        total_capital=total_cap,
        available_capital=max(vp.available_usdc, 0.0),
        daily_pnl=round(vp.daily_pnl(), 4),
        weekly_pnl=round(vp.weekly_pnl(), 4),
    )
