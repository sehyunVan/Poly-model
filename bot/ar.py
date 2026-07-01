"""
bot/ar.py — balance-threshold auto-redemption for the swarm bot.

Background
----------
The crypto 5m loop used to host AR (auto-redeem CTF wins → cash) on a 12h timer.
That loop was halted 2026-06-11, so AR stopped firing and swarm CTF wins began
piling up unredeemed — starving the swarm's working balance (redemption is
wallet-wide: both crypto and swarm wins are swept together).

This module re-homes AR in the swarm and switches the trigger from a timer to a
balance threshold, per the 2026-06-14 decision: **when the liquid balance drops
below $50, redeem.**  `maybe_redeem()` is called once per swarm cycle; when the
balance is under the threshold and the cooldown has elapsed it:

  1. redeems all winning CTF positions on-chain (USDC.e back to the wallet), then
  2. wraps any wallet USDC.e → pUSD so the CLOB can spend it.

It is rate-limited (AR_COOLDOWN) so it never submits on-chain txs more than once
per window even if the balance stays low and nothing is redeemable.

Reuses the production redemption path in src/crypto/redeem.py (correct positionId
derivation + wrap gas fix) — NOT scripts/redeem_positions.py.
"""
from __future__ import annotations

import logging
import os
import time

# Fire AR when liquid (pUSD) balance is below this many USD.
AR_BALANCE_THRESHOLD = 50.0
# Minimum seconds between AR attempts — on-chain redeem/wrap txs cost POL gas, so
# don't re-fire every cycle while the balance sits low with nothing redeemable.
AR_COOLDOWN = 1800.0  # 30 min

_last_ar_ts: float = 0.0


def maybe_redeem(balance: float, log: logging.Logger) -> float:
    """
    If `balance` is below AR_BALANCE_THRESHOLD and the cooldown has elapsed,
    redeem winning CTF positions and wrap the proceeds to pUSD.

    Returns the USD amount redeemed (0.0 if skipped, nothing redeemable, or error).
    """
    global _last_ar_ts

    if balance >= AR_BALANCE_THRESHOLD:
        return 0.0

    now = time.time()
    if now - _last_ar_ts < AR_COOLDOWN:
        return 0.0

    funder = os.getenv("FUNDER")
    key    = os.getenv("KEY")
    if not funder or not key:
        log.warning("AR: FUNDER/KEY not set in env — cannot redeem")
        return 0.0

    try:
        from src.crypto.redeem import (
            redeem_redeemable_positions,
            wrap_usdc_e_to_pusd,
        )
    except Exception as exc:
        log.warning("AR: cannot import redeem module — %s", exc)
        return 0.0

    # Mark the attempt before doing on-chain work so a slow/failed run still
    # respects the cooldown and doesn't re-fire on the very next cycle.
    _last_ar_ts = now
    log.info(
        "AR: liquid balance $%.2f < $%.2f threshold — sweeping CTF wins...",
        balance, AR_BALANCE_THRESHOLD,
    )

    redeemed = 0.0
    try:
        # Redeem first (drops USDC.e into the wallet), then wrap ALL wallet USDC.e
        # → pUSD in one pass — this makes the freshly-redeemed funds spendable in
        # the same run (the crypto loop wrapped-then-redeemed, leaving fresh USDC.e
        # for the next cycle; threshold-AR wants the cash now).
        redeemed = redeem_redeemable_positions(funder, key, log)
        wrapped  = wrap_usdc_e_to_pusd(funder, key, log)
        if redeemed > 0 or wrapped > 0:
            log.info("AR: done — redeemed $%.2f, wrapped $%.2f → pUSD", redeemed, wrapped)
        else:
            log.info("AR: nothing redeemable / wrappable right now")
    except Exception as exc:
        log.warning("AR: error during redeem/wrap — %s", exc)

    return redeemed
