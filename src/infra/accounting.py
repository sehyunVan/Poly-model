"""
infra/accounting.py — Centralised financial accounting for live trading.

Motivation
----------
Four separate accounting bugs in production all had the same root cause:
financial calculations were scattered across loop.py, portfolio.py, and the
dashboard, each with a slightly different understanding of what "PnL" meant:

  1. real_pnl_all_time() excluded pending_ctf_usdc (won tokens not yet redeemed).
  2. CTF redemption proceeds were misclassified as external deposits, inflating
     total_detected_deposits and making PnL appear -$87 instead of -$23.
  3. AR auto-redemption used raw real_clob_balance instead of crypto_balance
     (wallet_share × real_clob_balance), so the threshold was never triggered.
  4. exposure_for_market() summed YES + NO sides, double-counting hedges.

This module is the single source of truth for all financial math.

Design
------
AccountingLedger is a dataclass (not Pydantic) because it holds derived state
computed from a VirtualPortfolio — it is never persisted directly.

Usage
-----
    from infra.accounting import AccountingLedger

    ledger = AccountingLedger.from_portfolio(vp, wallet_share=0.50)
    print(ledger.real_pnl())          # True cash PnL
    print(ledger.crypto_balance())    # Bot's share of CLOB wallet
    print(ledger.should_redeem())     # True if AR trigger condition is met
    if ledger.is_deposit(jump):       # Deposit or redemption proceeds?
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime; portfolio.py imports from risk.schemas
    from virtual.portfolio import VirtualPortfolio


@dataclass(frozen=True)
class AccountingLedger:
    """
    Derived financial state computed from VirtualPortfolio fields.

    All fields are read-only snapshots — mutate the underlying VirtualPortfolio,
    then call AccountingLedger.from_portfolio() again to get a fresh ledger.

    Fields
    ------
    clob_balance        : on-chain USDC.e (liquid, from C1 sync).
    pending_ctf_usdc    : value of won-but-unredeemed CTF tokens.
    initial_clob        : first-ever live balance (set once, never overwritten).
    detected_deposits   : cumulative external top-ups detected since start.
    pending_credit      : amount just redeemed (suppresses deposit false-positives).
    wallet_share        : fraction of CLOB balance reserved for this loop (0.0–1.0).
    max_bet_abs         : max single bet size (used for AR and deposit thresholds).
    """

    clob_balance:      float
    pending_ctf_usdc:  float
    initial_clob:      float
    detected_deposits: float
    pending_credit:    float
    wallet_share:      float
    max_bet_abs:       float

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_portfolio(
        cls,
        vp: "VirtualPortfolio",
        wallet_share: float,
        max_bet_abs: float,
        pending_credit: float = 0.0,
    ) -> "AccountingLedger":
        """Build a ledger from the current VirtualPortfolio state."""
        return cls(
            clob_balance      = vp.real_clob_balance,
            pending_ctf_usdc  = vp.pending_ctf_usdc,
            initial_clob      = vp.initial_real_clob_balance,
            detected_deposits = vp.total_detected_deposits,
            pending_credit    = pending_credit,
            wallet_share      = wallet_share,
            max_bet_abs       = max_bet_abs,
        )

    # ── Core financial calculations ───────────────────────────────────────────

    def real_pnl(self) -> float:
        """
        True cash PnL since live trading started.

        Formula:
            real_pnl = clob_balance + pending_ctf_usdc
                       - initial_clob
                       - detected_deposits

        Includes pending CTF token value so PnL is accurate even before
        on-chain redemption completes.

        Returns 0.0 if initial_clob was never set (virtual mode).
        """
        if self.initial_clob == 0.0:
            return 0.0
        return (
            self.clob_balance
            + self.pending_ctf_usdc
            - self.initial_clob
            - self.detected_deposits
        )

    def crypto_balance(self) -> float:
        """
        This loop's share of the CLOB wallet.

        Formula: clob_balance × wallet_share

        Bug history: AR auto-redemption originally compared raw clob_balance
        against the MAX_BET_ABS×2 threshold.  When wallet_share=0.50, the
        crypto loop's actual allocation was half the raw balance — the threshold
        was never triggered.  Always use this method, not clob_balance directly.
        """
        return self.clob_balance * self.wallet_share

    def should_redeem(self) -> bool:
        """
        Return True if the AR (auto-redeem) condition is met.

        Trigger: crypto_balance() < max_bet_abs × 2
        Meaning: we can't afford two more bets — time to redeem winning CTF tokens.
        """
        return self.crypto_balance() < self.max_bet_abs * 2

    def is_deposit(self, balance_jump: float) -> bool:
        """
        Return True if a balance jump looks like an external deposit.

        A jump is a deposit if it exceeds max_bet_abs × 4 AND is NOT
        explained by a recent redemption (pending_credit × 1.5).

        Bug history: after a big CTF redemption batch (+$49), the deposit
        detector counted it as a user top-up, inflating detected_deposits by
        $49 and making real_pnl appear -$87 instead of -$23.

        The pending_credit guard prevents this: loop.py records the redeemed
        amount before the next balance sync, and is_deposit() ignores jumps
        that are within 150% of that credit.
        """
        if balance_jump <= 0:
            return False
        if self.pending_credit > 0 and balance_jump <= self.pending_credit * 1.5:
            return False
        return balance_jump > self.max_bet_abs * 4

    # ── Exposure accounting ───────────────────────────────────────────────────

    @staticmethod
    def exposure_for_market(yes_exposure: float, no_exposure: float) -> float:
        """
        Net exposure for a single market (accounts for hedged positions).

        Returns max(yes, no) rather than yes + no.

        Bug history: the original implementation returned yes + no, which
        double-counted any hedge (YES + NO positions in the same market).
        A $5 YES + $5 NO position would show $10 exposure instead of $5.
        """
        return max(yes_exposure, no_exposure)

    # ── Display helpers ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line summary for log output."""
        return (
            f"clob=${self.clob_balance:.2f}  "
            f"ctf=${self.pending_ctf_usdc:.2f}  "
            f"crypto_bal=${self.crypto_balance():.2f}  "
            f"real_pnl=${self.real_pnl():.2f}"
        )
