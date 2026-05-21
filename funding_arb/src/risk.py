"""Risk calculations for funding rate arb."""
import time


FUNDING_INTERVAL = 8 * 3600  # seconds between Binance funding settlements


def liquidation_price(entry_price: float, leverage: int) -> float:
    """
    Approximate liquidation price for a SHORT futures position.
    At Nx leverage, margin = entry/N.
    Liquidated when unrealised loss eats ~85% of margin.
      liq ≈ entry * (1 + 0.85 / leverage)
    """
    return entry_price * (1 + 0.85 / leverage)


def margin_distance(current_price: float, entry_price: float, leverage: int) -> float:
    """
    Fraction of current price between now and liquidation.
    Positive = safe (liq is above current price for a short).
    """
    liq = liquidation_price(entry_price, leverage)
    return (liq - current_price) / current_price


def is_margin_safe(
    current_price: float,
    entry_price: float,
    leverage: int,
    min_distance: float = 0.20,
) -> bool:
    """True if price is safely below the liquidation threshold."""
    dist = margin_distance(current_price, entry_price, leverage)
    return dist > min_distance


def funding_apy(rate_per_8h: float) -> float:
    """Annualised yield from funding rate."""
    return rate_per_8h * 3 * 365


def should_enter(rate: float, threshold: float) -> bool:
    """Positive funding: long spot + short perp earns the rate."""
    return rate >= threshold


def should_enter_reverse(rate: float, threshold: float, enabled: bool) -> bool:
    """
    Negative funding: short spot + long perp earns the rate paid by shorts.
    Requires Binance Margin account for spot borrowing — gated by `enabled`.
    """
    if not enabled:
        return False
    return rate <= -threshold


def funding_periods_due(last_collected_time: float) -> int:
    """
    How many 8h funding periods have elapsed since last collection.
    Handles restarts: if bot was down for 24h, returns 3 missed periods.
    Returns 0 if last_collected_time is 0 (never collected — first collection
    will be detected by next_funding_ms jump in the old approach; here we
    initialise on position open instead).
    """
    if last_collected_time <= 0:
        return 0
    elapsed = time.time() - last_collected_time
    return int(elapsed // FUNDING_INTERVAL)


def should_exit(
    rate: float,
    exit_threshold: float,
    margin_safe: bool,
    entry_time: float = 0.0,
    min_hold_seconds: float = 0.0,
) -> tuple[bool, str]:
    # Minimum hold guard — prevents thrashing if rate oscillates around threshold
    if min_hold_seconds > 0 and entry_time > 0:
        hold = time.time() - entry_time
        if hold < min_hold_seconds:
            return False, f"min hold not reached ({hold / 3600:.1f}h / {min_hold_seconds / 3600:.0f}h)"

    if rate < 0:
        return True, f"negative funding rate={rate:.6f}"
    if rate < exit_threshold:
        return True, f"rate={rate:.6f} below exit threshold={exit_threshold:.6f}"
    if not margin_safe:
        return True, "margin safety breach — price too close to liquidation"
    return False, ""
