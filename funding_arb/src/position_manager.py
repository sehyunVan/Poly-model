"""Open, close, and rebalance arb positions — one per symbol, multiple concurrent."""
import logging
import math
import time

from .futures_client import FuturesClient
from .spot_client import SpotClient
from .state import ArbState

log = logging.getLogger("arb.position")


def _floor_qty(qty: float, precision: int) -> float:
    factor = 10 ** precision
    return math.floor(qty * factor) / factor


def open_position(
    state: ArbState,
    spot: SpotClient,
    futures: FuturesClient,
    cfg: dict,
    symbol: str,
) -> None:
    """Long spot + short perp for one symbol. Stores result in state.positions[symbol]."""
    if symbol in state.positions:
        log.warning(f"open_position called but {symbol} already open — skipping")
        return

    virtual = cfg.get("virtual_mode", False)
    usdt = cfg["position_usdt"]
    leverage = cfg["futures_leverage"]
    mode_tag = "[VIRTUAL]" if virtual else "[LIVE]"

    price = futures.get_price(symbol)
    precision = futures.get_qty_precision(symbol)
    target_qty = _floor_qty(usdt / price, precision)

    if target_qty <= 0:
        raise ValueError(f"Position too small: {usdt} USDT @ {price} → qty={target_qty}")

    log.info(
        f"{mode_tag} OPEN {symbol} — price={price:.4f} qty={target_qty} "
        f"notional={usdt:.2f} USDT leverage={leverage}x"
    )

    if virtual:
        spot_qty = target_qty
        log.info(f"[VIRTUAL] Simulated spot buy: {spot_qty} {symbol} @ {price:.4f}")
        log.info(f"[VIRTUAL] Simulated futures short: {target_qty} {symbol}")
    else:
        futures.set_leverage(symbol, leverage)

        log.info(f"Buying {usdt:.2f} USDT of {symbol} spot...")
        spot_result = spot.place_market_buy_quote(symbol, usdt)
        spot_qty = float(spot_result.get("executedQty", target_qty))
        avg_price = float(spot_result.get("cummulativeQuoteQty", usdt)) / spot_qty
        log.info(f"Spot buy filled: {spot_qty} @ ~{avg_price:.4f}")

        fut_qty = _floor_qty(spot_qty, precision)
        log.info(f"Shorting {fut_qty} {symbol} on futures...")
        fut_result = futures.place_market_short(symbol, fut_qty)
        log.info(f"Futures short placed: orderId={fut_result.get('orderId')}")
        target_qty = fut_qty

    now = time.time()
    state.positions[symbol] = {
        "symbol": symbol,
        "entry_price": price,
        "spot_qty": spot_qty,
        "futures_qty": target_qty,
        "position_usdt": usdt,
        "entry_time": now,
        "last_funding_collected_time": now,
    }
    state.trade_count += 1

    liq = price * (1 + 0.85 / leverage)
    log.info(
        f"{mode_tag} Position OPEN — entry={price:.4f} liq≈{liq:.4f} "
        f"(+{(liq / price - 1) * 100:.1f}% from here)"
    )


def close_position(
    state: ArbState,
    spot: SpotClient,
    futures: FuturesClient,
    cfg: dict,
    symbol: str,
    close_price: float | None = None,
) -> None:
    """Close spot + futures legs for symbol and remove from state.positions."""
    pos = state.positions.get(symbol)
    if pos is None:
        log.warning(f"close_position called for {symbol} but no open position found")
        return

    virtual = cfg.get("virtual_mode", False)
    mode_tag = "[VIRTUAL]" if virtual else "[LIVE]"

    price = close_price or futures.get_price(symbol)
    entry_price = pos["entry_price"]
    spot_qty = pos["spot_qty"]
    fut_qty = pos["futures_qty"]

    log.info(
        f"{mode_tag} CLOSE {symbol} — "
        f"spot={spot_qty} futures={fut_qty} @ {price:.4f}"
    )

    if virtual:
        spot_pnl = (price - entry_price) * spot_qty
        fut_pnl = (entry_price - price) * fut_qty
        net_pnl = spot_pnl + fut_pnl
        duration_h = (time.time() - pos["entry_time"]) / 3600
        log.info(
            f"[VIRTUAL] Simulated close @ {price:.4f} — "
            f"spot_pnl={spot_pnl:+.4f} fut_pnl={fut_pnl:+.4f} "
            f"net={net_pnl:+.4f} over {duration_h:.1f}h"
        )
        state.total_realized_pnl += net_pnl
    else:
        try:
            futures.close_market_short(symbol, fut_qty)
            log.info("Futures short closed")
        except Exception as e:
            log.error(f"Failed to close futures: {e}")
            raise

        time.sleep(1)

        try:
            spot.place_market_sell(symbol, spot_qty)
            log.info("Spot sold")
        except Exception as e:
            log.error(f"Failed to sell spot (futures already closed!): {e}")
            raise

    duration_h = (time.time() - pos["entry_time"]) / 3600
    log.info(
        f"{mode_tag} {symbol} closed after {duration_h:.1f}h — "
        f"cumulative funding collected: {state.total_funding_collected:.4f} USDT"
    )

    del state.positions[symbol]


def rebalance(
    state: ArbState,
    futures: FuturesClient,
    cfg: dict,
    symbol: str,
) -> None:
    """Adjust futures qty if it drifted from spot qty for a specific symbol."""
    virtual = cfg.get("virtual_mode", False)
    if virtual:
        return

    pos = state.positions.get(symbol)
    if pos is None:
        return

    threshold = cfg.get("rebalance_threshold", 0.02)
    actual_fut_qty = futures.get_short_qty(symbol)
    target_qty = pos["spot_qty"]

    if target_qty == 0:
        return

    drift = abs(actual_fut_qty - target_qty) / target_qty
    if drift < threshold:
        return

    diff = target_qty - actual_fut_qty
    precision = futures.get_qty_precision(symbol)

    log.info(
        f"REBALANCE {symbol} — drift={drift:.2%} target={target_qty} "
        f"actual={actual_fut_qty} diff={diff:+}"
    )

    min_qty = 10 ** -precision
    if diff > min_qty:
        futures.place_market_short(symbol, _floor_qty(diff, precision))
    elif diff < -min_qty:
        futures.close_market_short(symbol, _floor_qty(abs(diff), precision))

    pos["futures_qty"] = futures.get_short_qty(symbol)
