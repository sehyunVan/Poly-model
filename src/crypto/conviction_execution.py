"""
Live CLOB order execution for market conviction bot.
Minimal version: places $1 BUY orders at market ask prices.
Reuses get_clob_balance from crypto.execution (which works).
"""

import logging
from typing import Optional
import requests

log = logging.getLogger("conviction_execution")

_HTTP = requests.Session()
_HTTP.headers["User-Agent"] = "conviction-bot/1.0"


def get_clob_balance() -> float:
    """
    Get USDC.e balance from CLOB.
    Imports and reuses the working crypto.execution version.
    """
    try:
        from execution import get_clob_balance as crypto_get_balance
        bal = crypto_get_balance(log)
        return bal if bal is not None else 0.0
    except Exception as e:
        log.error(f"get_clob_balance failed: {e}")
        return 0.0


def place_conviction_order(
    market_id: str,
    token_id: str,
    bet_size_usdc: float = 1.0,
    price_hint: Optional[float] = None,
) -> Optional[dict]:
    """
    Place a $1 BUY order for a conviction signal.
    Delegates to crypto.execution.place_crypto_order (which works).

    Args:
        market_id: Polymarket market ID
        token_id: CLOB token ID (UP or DOWN token)
        bet_size_usdc: Amount to bet (default $1)
        price_hint: Suggested entry price (skips re-fetch if provided)

    Returns:
        {"token_id": str, "ask": float, "bet": float, "tx": str} on success, None on failure
    """
    try:
        from execution import place_crypto_order

        # Use the crypto loop's place_crypto_order directly
        # It expects: token_id, bet_size_usdc, market_id, log, price_hint=None, band_min=0.0, band_max=1.0, maker_mode=True
        result = place_crypto_order(
            token_id=token_id,
            bet_size_usdc=bet_size_usdc,
            market_id=market_id,
            log=log,
            price_hint=price_hint,
            band_min=0.0,
            band_max=1.0,
        )

        if result and isinstance(result, dict) and result.get("tx"):
            tx = result.get("tx", "pending")
            ask = result.get("price", price_hint)
            log.info(
                f"ORDER PLACED {token_id[:8]}... @ ${ask:.3f} "
                f"qty={bet_size_usdc/ask if ask else 0:.3f} tx={tx[:16] if tx else 'pending'}..."
            )
            return {
                "token_id": token_id,
                "ask": ask,
                "bet": bet_size_usdc,
                "tx": tx,
            }
        else:
            log.warning(f"Order placement failed for {token_id}: {result}")
            return None

    except Exception as e:
        log.error(f"place_conviction_order failed: {e}", exc_info=True)
        return None
