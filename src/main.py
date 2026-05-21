"""
Polymarket automated trading — main loop.

Usage:
    python src/main.py                 # live trading (5-minute cycle)
    python src/main.py --dry-run       # full pipeline, but skip order submission
    python src/main.py --once          # run one cycle then exit (for testing)
    python src/main.py --log-level DEBUG

Cycle (every 5 minutes):
    1. Build portfolio snapshot from CLOB balance + open positions.
    2. Global risk evaluation — halt if daily/weekly loss limits are breached.
    3. Apply risk-mandated close/reduce actions.
    4. Fetch open markets (closing in > 1 hour).
    5. For each eligible market:
        a. Pre-trade filters (spread, volume, depth).
        b. Build feature vector (structured + text, cached 30 min).
        c. Predict P_R → compute alpha → generate TradeSignal.
        d. Exposure limit check + Kelly sizing.
        e. Slippage check.
        f. Execute order (skipped in --dry-run mode).
    6. At UTC midnight: daily report + rolling model retraining.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

# Load .env (searches from project root downward)
for _p in [_SRC.parent, _SRC.parent / "polymarket-mcp-main" / "polymarket-mcp-main"]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

# ── Project imports ───────────────────────────────────────────────────────────
from data.market         import get_markets, get_market, get_orderbook, get_current_price, get_volume_24h
from features            import build_feature_vector
from prediction.ensemble import predict_probability, reload_models
from prediction.training import RollingTrainer, record_settled_outcomes, save_feature_to_cache
from signal_layer.alpha  import generate_signals
from signal_layer.filters import passes_pre_trade_filters
from risk                import (
    Position, Portfolio, RiskLimits,
    check_exposure_limits, compute_position_size,
    estimate_slippage, check_liquidity,
    evaluate_portfolio_risk,
)
from execution           import execute_signal, close_position, OrderMonitor
from monitoring          import (
    compute_strategy_metrics,
    compute_prediction_metrics,
    generate_daily_report,
    suggest_parameter_adjustments,
    send_alert,
)
from monitoring.alerts   import send_daily_summary, send_gate_check

# ── Virtual mode configuration (read before any logic) ────────────────────────

_VIRTUAL_MODE   = os.getenv("VIRTUAL_MODE", "false").lower() == "true"
_VIRTUAL_BUDGET = float(os.getenv("VIRTUAL_BUDGET", "1000.0"))
_VIRTUAL_STATE_PATH = Path(os.getenv("VIRTUAL_STATE_PATH", "data/virtual_state.json"))

if _VIRTUAL_MODE:
    from virtual import (                           # noqa: E402
        VirtualPortfolio,
        load_virtual_portfolio,
        save_virtual_portfolio,
        portfolio_to_risk_portfolio,
        simulate_fill,
        settle_resolved_positions,
        auto_apply_suggestions,
    )

# ── Constants ─────────────────────────────────────────────────────────────────

CYCLE_INTERVAL_SEC    = 300         # 5-minute main loop
MIN_SECONDS_TO_CLOSE  = 3600        # ignore markets closing within 1 hour
MAX_SECONDS_TO_CLOSE  = 2_592_000   # ignore markets closing beyond 30 days
MIN_LIQUIDITY_SCORE   = 0.30        # skip very illiquid markets
TEXT_FEATURE_TTL_SEC  = 1800        # re-use text features for 30 minutes

_ROOT        = _SRC.parent
_STATE_FILE  = _ROOT / "data" / "trading_state.json"
_LOG_DIR     = _ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Hang protection ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def _api_timeout(seconds: int, label: str):
    """
    SIGALRM-based timeout for blocking API calls (Linux/macOS only).
    If the wrapped block takes longer than `seconds`, raises TimeoutError,
    which the cycle's outer except-block catches and logs before continuing.
    No-op on platforms without SIGALRM (e.g. Windows).
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(sig, frame):
        raise TimeoutError(f"{label} timed out after {seconds}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> logging.Logger:
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler — only when stdout is an interactive terminal.
    # When the process is started with `>> logs/main.log 2>&1`, stdout is
    # redirected to the file, so adding a StreamHandler would cause every
    # line to appear twice (once via the handler, once via the redirect).
    if sys.stdout.isatty():
        import io as _io
        _stdout = _io.TextIOWrapper(
            sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        ) if hasattr(sys.stdout, "buffer") else sys.stdout
        sh = logging.StreamHandler(_stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # Rotating file handler — 10 MB per file, keep 7 backups
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        _LOG_DIR / "main.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("main")


# ── Trading state (persisted across restarts) ─────────────────────────────────

@dataclass
class TradingState:
    pnl_history: list[dict] = field(default_factory=list)
    signals_today: list[dict] = field(default_factory=list)   # serialised dicts
    last_report_date: str = ""                                  # "YYYY-MM-DD"
    start_of_day_balance: float = 0.0
    trading_halted: bool = False

    # In-memory only (not persisted)
    text_feature_cache: dict = field(default_factory=dict)     # market_id → (ts, fv)


def _load_state() -> TradingState:
    try:
        if _STATE_FILE.exists():
            raw = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            return TradingState(
                pnl_history          = raw.get("pnl_history", []),
                signals_today        = raw.get("signals_today", []),
                last_report_date     = raw.get("last_report_date", ""),
                start_of_day_balance = float(raw.get("start_of_day_balance", 0.0)),
                trading_halted       = bool(raw.get("trading_halted", False)),
            )
    except Exception as exc:
        logging.getLogger("main").warning("Could not load state: %s — starting fresh", exc)
    return TradingState()


def _save_state(state: TradingState) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pnl_history":          state.pnl_history,
        "signals_today":        state.signals_today,
        "last_report_date":     state.last_report_date,
        "start_of_day_balance": state.start_of_day_balance,
        "trading_halted":       state.trading_halted,
    }
    _STATE_FILE.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


# ── Portfolio builder ─────────────────────────────────────────────────────────

def _get_clob_client():
    """Return an authenticated ClobClient or None."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from py_clob_client.clob_types import BalanceAllowanceParams

        key    = os.getenv("KEY")
        funder = os.getenv("FUNDER")
        host   = os.getenv("CLOB_HOST", "https://clob.polymarket.com")

        if not key or not funder:
            return None

        client = ClobClient(host, key=key, chain_id=POLYGON,
                            funder=funder, signature_type=1)
        client.set_api_creds(client.create_or_derive_api_creds())
        return client
    except Exception:
        return None


def _build_portfolio(state: TradingState, log: logging.Logger) -> Portfolio:
    """
    Fetch current USDC balance and open positions from the CLOB API and
    assemble a Portfolio snapshot.

    Falls back to an empty portfolio when credentials are absent or the
    API is unreachable (allows dry-run and CI testing without real keys).
    """
    client = _get_clob_client()

    # ── Balance ───────────────────────────────────────────────────────────────
    total_capital = state.start_of_day_balance or 1000.0   # sensible default
    available_capital = total_capital

    if client is not None:
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams
            bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=None))
            if isinstance(bal, str):
                bal = json.loads(bal)
            if isinstance(bal, dict):
                total_capital     = float(bal.get("balance", total_capital))
                available_capital = float(bal.get("allowance", total_capital))
        except Exception as exc:
            log.warning("balance fetch failed: %s", exc)

    # ── Positions ─────────────────────────────────────────────────────────────
    positions: list[Position] = []

    if client is not None:
        try:
            raw_positions = client.get_positions()
            if isinstance(raw_positions, str):
                raw_positions = json.loads(raw_positions)
            if isinstance(raw_positions, dict):
                raw_positions = raw_positions.get("data", [])
            if isinstance(raw_positions, list):
                for rp in raw_positions:
                    try:
                        market_id  = rp.get("market") or rp.get("condition_id", "")
                        token_id   = rp.get("asset_id") or rp.get("token_id", "")
                        size       = float(rp.get("size") or rp.get("amount") or 0)
                        avg_price  = float(rp.get("avg_price") or rp.get("price") or 0.5)
                        curr_price = get_current_price(market_id) or avg_price

                        # Determine side from token by matching market data
                        mkt  = get_market(market_id)
                        side = "YES"
                        cat  = "other"
                        if mkt:
                            cat  = mkt.category
                            side = "YES" if token_id == mkt.yes_token_id else "NO"

                        if size > 0 and market_id:
                            pos = Position(
                                market_id=market_id,
                                token_id=token_id,
                                side=side,
                                size=size,
                                avg_entry_price=avg_price,
                                current_price=curr_price,
                                category=cat,
                            ).recalculate_pnl()
                            positions.append(pos)
                    except Exception as pexc:
                        log.debug("skipping position parse error: %s", pexc)
        except Exception as exc:
            log.warning("positions fetch failed: %s", exc)

    deployed        = sum(p.size for p in positions)
    available_capital = max(total_capital - deployed, 0.0)
    daily_pnl       = total_capital - state.start_of_day_balance if state.start_of_day_balance else 0.0
    weekly_pnl      = sum(e.get("pnl", 0.0) for e in state.pnl_history[-7:])

    return Portfolio(
        positions=positions,
        total_capital=total_capital,
        available_capital=available_capital,
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
    )


# ── Risk action handler ───────────────────────────────────────────────────────

def _apply_risk_actions(
    actions: list[dict],
    portfolio: Portfolio,
    monitor: OrderMonitor,
    state: TradingState,
    dry_run: bool,
    log: logging.Logger,
) -> bool:
    """
    Execute risk-mandated actions (close_position / reduce_position).

    Returns True if new trading should be halted for this cycle.
    """
    halted = False

    for act in actions:
        action = act.get("action", "")

        if action == "halt_new_trades":
            reason = act.get("reason", "")
            log.warning("RISK: halting new trades — %s", reason)
            state.trading_halted = True
            halted = True
            send_alert(
                f"Trading HALTED\nReason: {reason}",
                level="CRITICAL",
            )

        elif action == "resume_trading":
            if state.trading_halted:
                log.info("RISK: lifting halt — %s", act.get("reason", ""))
            state.trading_halted = False

        elif action in ("close_position", "reduce_position"):
            mid        = act.get("market_id", "")
            target     = float(act.get("target_size", 0.0))
            pos = next((p for p in portfolio.positions if p.market_id == mid), None)

            if pos is None:
                continue

            if action == "close_position" or target == 0.0:
                log.warning("RISK: closing position %s — %s", mid, act.get("reason", ""))
                if not dry_run:
                    result = close_position(pos, mode="limit")
                    log.info("close_position result: %s", result.status)
                else:
                    log.info("[DRY-RUN] would close %s size=%.2f", mid, pos.size)
            else:
                reduce_by = pos.size - target
                if reduce_by > 0:
                    log.warning(
                        "RISK: reducing %s by %.2f → target %.2f",
                        mid, reduce_by, target,
                    )
                    if not dry_run:
                        partial_pos = pos.model_copy(update={"size": reduce_by})
                        result = close_position(partial_pos, mode="limit")
                        log.info("reduce_position result: %s", result.status)
                    else:
                        log.info("[DRY-RUN] would reduce %s by %.2f", mid, reduce_by)

        # Cancel all open orders tracked by the monitor when halting
        if halted and not dry_run:
            monitor.stop()

    return halted


# ── Per-market pipeline ───────────────────────────────────────────────────────

def _process_market(
    market,
    portfolio: Portfolio,
    limits: RiskLimits,
    state: TradingState,
    signals_out: list,
    monitor: OrderMonitor,
    dry_run: bool,
    log: logging.Logger,
    virtual_portfolio: "Optional[VirtualPortfolio]" = None,
    filter_params: "Optional[dict]" = None,
) -> None:
    """
    Run the full predict → signal → risk → execute pipeline for one market.
    Errors are caught and logged so that one bad market never kills the cycle.

    When virtual_portfolio is provided (VIRTUAL_MODE=true), order execution is
    replaced by simulate_fill() and the result is recorded in the virtual
    portfolio instead of being submitted to the CLOB.
    """
    mid = market.market_id

    # ── Pre-trade filters ─────────────────────────────────────────────────────
    ob = get_orderbook(mid)

    # Real 24h volume from the /trades endpoint.
    # Falls back to an orderbook-depth proxy when the API returns nothing
    # (e.g. very new market, API timeout, or rate-limit).
    volume_24h = get_volume_24h(mid)
    if volume_24h == 0.0:
        depth_usdc = sum(l.size for l in ob.bids[:10]) + sum(l.size for l in ob.asks[:10])
        volume_24h = depth_usdc * 2.0
        log.debug("%s using depth proxy for volume: %.2f", mid, volume_24h)
    else:
        log.debug("%s real 24h volume: %.2f USDC", mid, volume_24h)

    fp = filter_params or {}
    ok, reason = passes_pre_trade_filters(
        ob,
        volume_24h,
        min_volume_24h=fp.get("min_volume_24h", 1000.0),
        max_volume_24h=fp.get("max_volume_24h", 50000.0),
        max_spread=fp.get("max_spread", 0.03),
        min_book_depth_usdc=fp.get("min_book_depth_usdc", 100.0),
        min_market_price=fp.get("min_market_price", 0.08),
        max_market_price=fp.get("max_market_price", 0.92),
    )
    if not ok:
        log.debug("%s skipped pre-trade filter: %s", mid, reason)
        return

    liq = check_liquidity(ob, volume_24h)
    if liq < MIN_LIQUIDITY_SCORE:
        log.debug("%s liquidity score %.2f < threshold", mid, liq)
        return

    # ── Feature vector (with 30-minute cache for text features) ──────────────
    now    = datetime.now(timezone.utc)
    cached = state.text_feature_cache.get(mid)
    if cached and (now.timestamp() - cached[0]) < TEXT_FEATURE_TTL_SEC:
        fv = cached[1]
    else:
        fv = build_feature_vector(mid)
        state.text_feature_cache[mid] = (now.timestamp(), fv)
        # Persist feature vector so record_settled_outcomes() can label it later
        try:
            save_feature_to_cache(fv, outcome=None,
                                  cache_dir=_ROOT / "data" / "features_cache")
        except Exception as _e:
            log.debug("%s feature cache write skipped: %s", mid, _e)

    # ── Prediction ────────────────────────────────────────────────────────────
    pred = predict_probability(fv)
    log.debug("%s  P_M=%.3f  P_R=%.3f  conf=%.2f",
              mid, pred.P_M, pred.P_R, pred.confidence)

    # ── Signal generation ─────────────────────────────────────────────────────
    signal = generate_signals(pred, market)
    if signal.direction == "NO_TRADE":
        log.debug("%s NO_TRADE: %s", mid, signal.reason)
        return

    signals_out.append(signal)
    log.info("%s  signal=%s  alpha=%.4f  factor=%.2f",
             mid, signal.direction, signal.alpha, signal.base_size_factor)

    # ── Deduplication: skip if already holding a position in this direction ───
    _side = "YES" if signal.direction == "BUY_YES" else "NO"
    if any(p.market_id == mid and p.side == _side for p in portfolio.positions):
        log.debug("%s already has open %s position — skipping duplicate entry", mid, _side)
        return
    # In virtual mode portfolio.positions is always empty (no real CLOB trades).
    # Check the virtual portfolio directly — block ANY existing position on this
    # market (either direction) to prevent holding conflicting YES+NO positions.
    if virtual_portfolio is not None:
        if any(p.market_id == mid for p in virtual_portfolio.positions):
            log.debug("%s already has a virtual position — skipping", mid)
            return

    # ── Exposure limit check ──────────────────────────────────────────────────
    # First estimate with a preliminary size to check category/group limits.
    prelim_size = compute_position_size(
        signal.alpha, pred.P_R, pred.P_M,
        portfolio.available_capital, limits, signal.base_size_factor,
        volume_24h=volume_24h,
    )
    if prelim_size <= 0:
        log.debug("%s Kelly size = 0 — no edge", mid)
        return

    ok, reason = check_exposure_limits(
        signal, prelim_size, portfolio, limits, category=market.category,
    )
    if not ok:
        log.info("%s blocked by exposure limit: %s", mid, reason)
        return

    # ── Final sizing after exposure approval ──────────────────────────────────
    final_size = prelim_size

    # ── Slippage check ────────────────────────────────────────────────────────
    slip = estimate_slippage(ob, final_size, "BUY")
    if slip > limits.max_slippage_pct:
        log.info(
            "%s slippage %.4f > cap %.4f — skipping",
            mid, slip, limits.max_slippage_pct,
        )
        return

    # ── Order execution ───────────────────────────────────────────────────────
    if dry_run:
        log.info(
            "[DRY-RUN] would execute: %s %s  size=%.2f  slip=%.4f",
            mid, signal.direction, final_size, slip,
        )
        return

    # Virtual mode: simulate fill from live orderbook, no real order placed
    if virtual_portfolio is not None:
        result = simulate_fill(signal, final_size, ob, slippage_cap=limits.max_slippage_pct)
        log.info(
            "[VIRTUAL] %s sim-fill: status=%s  filled=%.2f  avg_price=%.4f",
            mid, result.status, result.filled_size, result.avg_fill_price,
        )
        if result.status == "FILLED" and result.filled_size > 0:
            # Alert on large fills (> 3% of total capital)
            _fill_threshold = portfolio.total_capital * 0.03
            if result.filled_size >= _fill_threshold:
                send_alert(
                    f"[VIRTUAL] Large fill: {mid}\n"
                    f"Direction: {signal.direction}  "
                    f"Size: {result.filled_size:.2f} USDC  "
                    f"Price: {result.avg_fill_price:.4f}",
                    level="INFO",
                )
            from virtual.portfolio import VirtualPosition   # avoid top-level circular import
            from risk.schemas import Position as _RiskPosition  # for intra-cycle tracking
            direction = "YES" if signal.direction == "BUY_YES" else "NO"
            vpos = VirtualPosition(
                market_id=mid,
                title=market.title,
                direction=direction,
                size_usdc=result.filled_size,
                fill_price=result.avg_fill_price,
                fill_time=datetime.now(timezone.utc),
                category=market.category,
            )
            virtual_portfolio.positions.append(vpos)
            virtual_portfolio.available_usdc -= result.filled_size
            virtual_portfolio.available_usdc  = max(virtual_portfolio.available_usdc, 0.0)
            virtual_portfolio.mark_updated()
            # Sync portfolio so subsequent markets in this cycle see the updated
            # available balance and the new position for exposure limit checks.
            portfolio.available_capital = virtual_portfolio.available_usdc
            portfolio.positions.append(_RiskPosition(
                market_id=mid,
                token_id="",
                side=direction,
                size=result.filled_size,
                avg_entry_price=result.avg_fill_price,
                current_price=result.avg_fill_price,
                unrealized_pnl=0.0,
                category=market.category,
            ))
            # Persist immediately so the next cycle reload sees this position
            save_virtual_portfolio(virtual_portfolio, _ROOT / _VIRTUAL_STATE_PATH)
        return

    # Live mode: submit real order
    result = execute_signal(signal, final_size, slippage_cap=limits.max_slippage_pct)
    log.info(
        "%s order result: status=%s  filled=%.2f  avg_price=%.4f  attempts=%d",
        mid, result.status, result.filled_size, result.avg_fill_price, result.attempts,
    )

    # Alert on large fills (> 3% of total capital)
    if result.status == "FILLED" and result.filled_size > 0:
        _fill_threshold = portfolio.total_capital * 0.03
        if result.filled_size >= _fill_threshold:
            send_alert(
                f"Large fill: {mid}\n"
                f"Direction: {signal.direction}  "
                f"Size: {result.filled_size:.2f} USDC  "
                f"Price: {result.avg_fill_price:.4f}  "
                f"Order: {result.order_id}",
                level="INFO",
            )

    # Register filled orders with the monitor for live tracking
    if result.status == "FILLED" and result.order_id:
        monitor.track(
            result.order_id, signal,
            price=result.avg_fill_price,
            size=result.filled_size,
            token_id=result.token_id,
            side=result.side,
        )


# ── Midnight tasks ────────────────────────────────────────────────────────────

def _midnight_tasks(
    portfolio: Portfolio,
    state: TradingState,
    limits: RiskLimits,
    current_balance: float,
    log: logging.Logger,
    virtual_portfolio: "Optional[VirtualPortfolio]" = None,
    signals_today: "Optional[list[dict]]" = None,
) -> None:
    """
    Tasks that run once per day at UTC midnight:
        1. Record today's PnL into history.
        2. Generate and save the daily Markdown report.
        3. Print (and in virtual mode: auto-apply) parameter adjustment suggestions.
        4. [Virtual mode] Settle resolved positions, save virtual portfolio.
        5. Record settled market outcomes into the feature cache.
        6. Trigger a rolling model retraining cycle (uses labels from step 5).
        7. Reset daily state.
    """
    today_str = date.today().isoformat()
    log.info("--- Midnight tasks for %s ---", today_str)

    # 1. Record daily PnL
    daily_pnl = current_balance - state.start_of_day_balance
    state.pnl_history.append({
        "date": today_str,
        "pnl":  round(daily_pnl, 4),
    })
    # Keep at most 365 days of history
    if len(state.pnl_history) > 365:
        state.pnl_history = state.pnl_history[-365:]
    log.info("Daily PnL recorded: %+.2f USDC", daily_pnl)

    # 2. Daily report
    try:
        sm = compute_strategy_metrics(state.pnl_history[-30:])
        pm = compute_prediction_metrics([], [])   # predictions stored separately in prod
        report = generate_daily_report(
            portfolio, sm, pm, [], save=True,
            date=datetime.now(timezone.utc),
        )
        log.info("Daily report saved (%d chars)", len(report))
    except Exception as exc:
        log.error("Daily report failed: %s", exc)

    # 3. Parameter suggestions
    try:
        sm7 = compute_strategy_metrics(state.pnl_history[-7:])
        current_params = {
            "alpha_threshold": 0.05,
            "kelly_fraction":  limits.kelly_fraction,
            "max_bet_pct":     limits.max_bet_pct,
        }
        suggestions = suggest_parameter_adjustments([sm7], current_params)
        if suggestions:
            log.warning("Parameter adjustment suggestions:")
            for s in suggestions:
                log.warning("  [%s] %s → %s | %s",
                            s["param"], s["current"], s["suggested"],
                            s["reason"][:100])
            # In virtual mode: automatically apply the suggestions to config YAML files
            if virtual_portfolio is not None:
                try:
                    applied = auto_apply_suggestions(suggestions, config_root=_ROOT)
                    if applied:
                        log.info("[VIRTUAL] Auto-applied %d param changes:", len(applied))
                        for change in applied:
                            log.info("  [VIRTUAL] %s", change)
                    # Reload limits so the next cycle uses updated values
                    limits.__dict__.update(RiskLimits.from_config().__dict__)
                except Exception as ae:
                    log.error("[VIRTUAL] Auto-apply failed: %s", ae)
        else:
            log.info("No parameter adjustments suggested.")
    except Exception as exc:
        log.error("Parameter suggestions failed: %s", exc)

    # 4. [Virtual mode] Settle resolved positions and save virtual portfolio
    if virtual_portfolio is not None:
        try:
            settle_result = settle_resolved_positions(
                virtual_portfolio,
                cache_dir=_ROOT / "data" / "features_cache",
            )
            log.info(
                "[VIRTUAL] Settled positions: %d settled, %d still open",
                settle_result["settled"], settle_result["still_open"],
            )

            # Record daily virtual PnL
            # cumulative_pnl is the true sum of all realized PnL from closed positions,
            # not pnl_history (which only reflects previously recorded days).
            today_str = date.today().isoformat()
            actual_cumulative = sum(
                (p.realized_pnl or 0.0) for p in virtual_portfolio.closed_positions
            )
            prev_cumulative = sum(e["pnl"] for e in virtual_portfolio.pnl_history)
            actual_daily = round(actual_cumulative - prev_cumulative, 4)
            virtual_portfolio.pnl_history.append({
                "date":           today_str,
                "pnl":            actual_daily,
                "cumulative_pnl": round(actual_cumulative, 4),
            })

            # Persist state
            save_virtual_portfolio(virtual_portfolio, _ROOT / _VIRTUAL_STATE_PATH)
            log.info(
                "[VIRTUAL] Portfolio saved — available=%.2f  cumulative_pnl=%+.2f",
                virtual_portfolio.available_usdc, actual_cumulative,
            )

            # Generate virtual daily report
            _vreport_dir = _LOG_DIR / "virtual_reports"
            _vreport_dir.mkdir(parents=True, exist_ok=True)
            try:
                sm_v = compute_strategy_metrics(virtual_portfolio.pnl_history[-30:])
                pm_v = compute_prediction_metrics([], [])
                # Rebuild portfolio from the freshly-updated virtual_portfolio so that
                # daily_pnl in the report header reflects today's settled PnL (not 0.0).
                fresh_portfolio = portfolio_to_risk_portfolio(virtual_portfolio)
                # Convert signal dicts to duck-typed objects for the report.
                from types import SimpleNamespace
                sig_objs = [
                    SimpleNamespace(**s)
                    for s in (signals_today or [])
                ]
                virtual_report = generate_daily_report(
                    fresh_portfolio, sm_v, pm_v, sig_objs, save=False,
                    date=datetime.now(timezone.utc),
                )
                vreport_path = _vreport_dir / f"{today_str}.md"
                vreport_path.write_text(virtual_report, encoding="utf-8")
                log.info("[VIRTUAL] Report saved to %s", vreport_path)
            except Exception as rexc:
                log.error("[VIRTUAL] Report generation failed: %s", rexc)

            # Discord: short daily summary
            try:
                send_daily_summary(virtual_portfolio)
            except Exception as _e:
                log.warning("Discord daily summary failed: %s", _e)

            # Discord: gate status (every day so you can track progress)
            try:
                send_gate_check(virtual_portfolio)
            except Exception as _e:
                log.warning("Discord gate check failed: %s", _e)

        except Exception as exc:
            log.error("[VIRTUAL] Midnight virtual tasks failed: %s", exc)

    # 5. Record settled outcomes (must run before retraining so new labels are included)
    try:
        oc = record_settled_outcomes()
        log.info(
            "Outcome recording: recorded=%d skipped=%d api_error=%s",
            oc["recorded"], oc["skipped"], oc["api_error"],
        )
        if oc["api_error"]:
            log.warning("Outcome recording encountered an API error — labels may be incomplete.")
    except Exception as exc:
        log.error("Outcome recording failed: %s", exc)

    # 6. Rolling retraining
    try:
        trainer = RollingTrainer()
        report_dict = trainer.run_training_cycle()
        log.info("Retraining cycle: %s", report_dict)
        reload_models()
        log.info("Models reloaded after retraining.")
        send_alert(
            f"Model retraining complete ({today_str})\n"
            f"Summary: {report_dict}",
            level="INFO",
        )
    except Exception as exc:
        log.error("Retraining failed: %s", exc)
        send_alert(
            f"Model retraining FAILED ({today_str})\n"
            f"Error: {exc}",
            level="ERROR",
        )

    # 7. Reset daily counters
    state.last_report_date     = today_str
    state.start_of_day_balance = current_balance
    state.signals_today        = []
    state.text_feature_cache   = {}
    state.trading_halted       = False


# ── Main loop ─────────────────────────────────────────────────────────────────

def _load_signal_params() -> dict:
    """Load pre-trade filter params from signal_params.yaml."""
    path = _ROOT / "config" / "signal_params.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("pre_trade_filters", {})
    except Exception:
        return {}


def run_loop(dry_run: bool, once: bool, log: logging.Logger) -> None:
    """
    Main 5-minute trading loop.  Runs until interrupted or `once=True`.
    """
    limits       = RiskLimits.from_config()
    state        = _load_state()
    monitor      = OrderMonitor()
    filter_params = _load_signal_params()

    # Virtual portfolio (None when VIRTUAL_MODE is off)
    _vp = None
    if _VIRTUAL_MODE:
        _vp = load_virtual_portfolio(_ROOT / _VIRTUAL_STATE_PATH, _VIRTUAL_BUDGET)
        log.info(
            "[VIRTUAL] Paper trading mode active — budget=%.2f  available=%.2f  "
            "open_positions=%d",
            _vp.initial_budget, _vp.available_usdc, len(_vp.positions),
        )

    if not dry_run:
        monitor.start()
        log.info("OrderMonitor started.")

    # Graceful shutdown on SIGINT / SIGTERM
    _shutdown = {"requested": False}
    def _handle_signal(sig, frame):
        log.info("Shutdown signal received — finishing current cycle.")
        _shutdown["requested"] = True
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(
        "Trading loop started  dry_run=%s  cycle=%ds  "
        "term_window=[%ds, %ds]  (1h – 30d)",
        dry_run, CYCLE_INTERVAL_SEC, MIN_SECONDS_TO_CLOSE, MAX_SECONDS_TO_CLOSE,
    )

    # Initialise start-of-day balance on first run
    if state.start_of_day_balance == 0.0:
        if _vp is not None:
            state.start_of_day_balance = _vp.total_capital()
        else:
            portfolio = _build_portfolio(state, log)
            state.start_of_day_balance = portfolio.total_capital
        _save_state(state)
        log.info("Initialised start-of-day balance: %.2f USDC",
                 state.start_of_day_balance)

    while not _shutdown["requested"]:
        cycle_start = time.monotonic()
        now         = datetime.now(timezone.utc)
        log.info("=== Cycle start %s ===", now.strftime("%Y-%m-%dT%H:%M:%SZ"))

        try:
            # ── 1. Build portfolio ────────────────────────────────────────────
            if _vp is not None:
                # Reload from disk so in-memory changes are reflected
                _vp = load_virtual_portfolio(_ROOT / _VIRTUAL_STATE_PATH, _VIRTUAL_BUDGET)
                portfolio = portfolio_to_risk_portfolio(_vp)
            else:
                portfolio = _build_portfolio(state, log)
            log.info(
                "Portfolio: total=%.2f  available=%.2f  daily_pnl=%+.2f  "
                "positions=%d%s",
                portfolio.total_capital, portfolio.available_capital,
                portfolio.daily_pnl, len(portfolio.positions),
                "  [VIRTUAL]" if _vp is not None else "",
            )

            # ── 2. Global risk evaluation ─────────────────────────────────────
            risk_actions = evaluate_portfolio_risk(
                portfolio, state.pnl_history, limits,
            )
            halted = _apply_risk_actions(
                risk_actions, portfolio, monitor, state, dry_run, log,
            )

            if halted:
                log.warning("New trades halted by risk module — skipping signal cycle.")
                _save_state(state)
            else:
                # ── 3. Fetch eligible markets ─────────────────────────────────
                with _api_timeout(120, "get_markets"):
                    markets = get_markets(status="open", limit=100)
                eligible = [
                    m for m in markets
                    if MIN_SECONDS_TO_CLOSE
                    < (m.close_time - now).total_seconds()
                    <= MAX_SECONDS_TO_CLOSE
                    and m.category != "crypto"  # crypto handled by dedicated crypto loop
                ]
                log.info(
                    "Markets: %d open, %d eligible (1h–5d to close, non-crypto)",
                    len(markets), len(eligible),
                )

                # ── 4. Process each market ────────────────────────────────────
                signals_this_cycle: list = []
                for market in eligible:
                    if _shutdown["requested"]:
                        break
                    try:
                        _process_market(
                            market, portfolio, limits, state,
                            signals_this_cycle, monitor, dry_run, log,
                            virtual_portfolio=_vp,
                            filter_params=filter_params,
                        )
                    except Exception as exc:
                        log.error("market %s failed: %s", market.market_id, exc,
                                  exc_info=True)

                log.info("Cycle complete -- %d signals generated, %d tradeable",
                         len(signals_this_cycle),
                         sum(1 for s in signals_this_cycle
                             if s.direction != "NO_TRADE"))

                # Serialise signals for the daily report
                for s in signals_this_cycle:
                    state.signals_today.append({
                        "market_id": s.market_id,
                        "direction": s.direction,
                        "alpha":     s.alpha,
                        "ts":        now.isoformat(),
                    })

                _save_state(state)

            # ── 5. Midnight tasks (once per day) ──────────────────────────────
            today_str = date.today().isoformat()
            if state.last_report_date != today_str and now.hour == 0:
                with _api_timeout(600, "_midnight_tasks"):
                    _midnight_tasks(
                        portfolio, state, limits,
                        current_balance=portfolio.total_capital,
                        log=log,
                        virtual_portfolio=_vp,
                        signals_today=state.signals_today,
                    )
                _save_state(state)

        except Exception as cycle_exc:
            log.error("Unhandled cycle error: %s", cycle_exc, exc_info=True)
            send_alert(
                f"Main loop unhandled exception\n"
                f"Cycle: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
                f"Error: {cycle_exc}",
                level="ERROR",
            )

        if once or _shutdown["requested"]:
            break

        # ── Sleep until next cycle ────────────────────────────────────────────
        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, CYCLE_INTERVAL_SEC - elapsed)
        log.info("Cycle took %.1fs — sleeping %.1fs", elapsed, sleep_for)
        time.sleep(sleep_for)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not dry_run:
        monitor.stop()
    _save_state(state)
    log.info("Trading loop stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket automated trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py --dry-run        # test without real orders\n"
            "  python src/main.py --once           # single cycle and exit\n"
            "  python src/main.py --log-level DEBUG\n"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full pipeline but skip actual order submission.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Execute one cycle and exit (useful for smoke tests).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    log = _setup_logging(args.log_level)

    if args.dry_run:
        log.info("*** DRY-RUN MODE -- no orders will be placed ***")
    if _VIRTUAL_MODE:
        log.info(
            "*** VIRTUAL MODE -- paper trading active  budget=%.2f USDC ***",
            _VIRTUAL_BUDGET,
        )

    run_loop(dry_run=args.dry_run, once=args.once, log=log)


if __name__ == "__main__":
    main()
