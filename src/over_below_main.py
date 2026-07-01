"""
over_below_main.py — Entry point for the over/below LLM price-target scanner.

Scans Polymarket for markets asking whether a crypto asset (BTC/ETH/SOL) will
reach a specific price level by a given date. Uses two LLMs in parallel to
estimate the true probability and bets when they agree AND diverge from market
price by at least min_edge.

Run (paper mode, default):
    source .venv/bin/activate && python src/over_below_main.py >> logs/over_below.log 2>&1

To run live, set OVER_BELOW_VIRTUAL_MODE=false in .env (requires CLOB credentials).

Screen session:
    screen -dmS over_below bash -c 'source .venv/bin/activate && python src/over_below_main.py >> logs/over_below.log 2>&1'
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
for _p in [_ROOT, _ROOT / "polymarket-mcp-main" / "polymarket-mcp-main"]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

# ── Logging ────────────────────────────────────────────────────────────────────
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "over_below.log", maxBytes=10 * 1024 * 1024, backupCount=3,
)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler,
                                                   logging.StreamHandler(sys.stdout)])
log = logging.getLogger("over_below_main")

# ── Config ─────────────────────────────────────────────────────────────────────
_CONFIG_PATH = _ROOT / "config" / "over_below_params.yaml"

def _load_config() -> dict:
    defaults = {
        "virtual_mode":           True,
        "scan_interval":          600,
        "min_hours":              2.0,
        "max_hours":              720.0,
        "min_volume_24h":         500.0,
        "min_yes_price":          0.10,
        "max_yes_price":          0.90,
        "min_edge":               0.12,
        "max_llm_disagreement":   0.20,
        "max_move_pct":           60.0,
        "bet_abs":                5.0,
        "max_daily_bets":         5,
        "max_per_scan":           10,
        "symbols":                ["BTC", "ETH", "SOL"],
    }
    try:
        import yaml
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return {**defaults, **loaded}
    except Exception:
        return defaults


# ── Active models (reuse bot/config pattern) ───────────────────────────────────

def _get_models() -> list[dict]:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key   = os.getenv("OPENAI_API_KEY")
    qwen_key     = os.getenv("QWEN_API_KEY")

    all_models = [
        {
            "name": "deepseek",
            "provider": "openai-compat",
            "model": "deepseek-chat",
            "api_key": deepseek_key,
            "base_url": "https://api.deepseek.com/v1",
        },
        {
            "name": "qwen",
            "provider": "openai-compat",
            "model": "qwen-plus",
            "api_key": qwen_key,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        {
            "name": "gpt-4o-mini",
            "provider": "openai-compat",
            "model": "gpt-4o-mini",
            "api_key": openai_key,
            "base_url": "https://api.openai.com/v1",
        },
    ]
    active = [m for m in all_models if m.get("api_key")]
    if len(active) < 2:
        log.warning("Only %d active model(s) — need 2+ for consensus", len(active))
    return active


# ── Main loop ──────────────────────────────────────────────────────────────────

async def run() -> None:
    cfg    = _load_config()
    models = _get_models()
    virtual_mode = (os.getenv("OVER_BELOW_VIRTUAL_MODE", "true").lower() != "false"
                    and cfg.get("virtual_mode", True))

    log.info(
        "Over/Below scanner started | mode=%s | interval=%ds | min_edge=%.2f | "
        "models=%s",
        "PAPER" if virtual_mode else "LIVE",
        cfg["scan_interval"],
        cfg["min_edge"],
        [m["name"] for m in models],
    )

    from over_below.scanner import scan_once
    from over_below.paper   import (load_trades, save_trades, load_executed,
                                    save_executed, prune_executed,
                                    record_trade, settle_open_trades, print_summary)

    trades   = load_trades()
    executed = prune_executed(load_executed())

    while True:
        # ── Settle open trades ────────────────────────────────────────────────
        n_settled = settle_open_trades(trades)
        if n_settled:
            save_trades(trades)
            log.info("Settled %d trade(s) this cycle", n_settled)
            print_summary(trades)

        # ── Scan for new signals (open-position cap, not daily cap) ───────────
        open_count  = sum(1 for t in trades if not t.settled)
        max_open    = cfg.get("max_open_positions", 5)
        slots       = max_open - open_count

        if slots > 0:
            skip_ids = set(executed.keys())
            signals  = await scan_once(cfg, models, skip_ids)

            for sig in signals:
                open_count = sum(1 for t in trades if not t.settled)
                if open_count >= max_open:
                    log.info("Open position cap reached (%d/%d) — skipping", open_count, max_open)
                    break

                bet_size = cfg.get("bet_abs", 5.0)
                log.info(
                    "=== %s SIGNAL === %s  llm=%.2f mkt=%.2f edge=%+.3f  "
                    "models: %s=%.2f %s=%.2f",
                    "PAPER" if virtual_mode else "LIVE",
                    sig.market.question[:70],
                    sig.avg_prob, sig.market_prob, sig.edge,
                    sig.model_1_name, sig.llm_prob_1,
                    sig.model_2_name, sig.llm_prob_2,
                )

                if virtual_mode:
                    # Append to the in-memory list so the settler (which only
                    # sees `trades`) can resolve trades recorded after startup.
                    trades.append(record_trade(sig, bet_size))
                else:
                    result = await _execute_live(sig, cfg)
                    if result:
                        trades.append(record_trade(sig, result["bet"]))
                    else:
                        log.warning("Live execution failed for %s", sig.market.question[:60])
                        continue

                # Mark as executed (dedup for 7 days)
                executed[sig.market.market_id] = time.time()
                executed = prune_executed(executed)
                save_executed(executed)
        else:
            log.info("Open cap full (%d/%d) — skipping scan", open_count, max_open)

        log.info("Cycle done. open=%d/%d Sleeping %ds...", open_count, max_open, cfg["scan_interval"])
        await asyncio.sleep(cfg["scan_interval"])


# ── Live execution ─────────────────────────────────────────────────────────────

async def _execute_live(sig, cfg: dict) -> dict | None:
    """Place a real CLOB order via bot/execution.py helpers."""
    try:
        import httpx as _hx
        from over_below.scanner import OBMarket

        m         = sig.market
        token_id  = m.yes_token_id if sig.bet_direction == "YES" else m.no_token_id
        if not token_id:
            log.warning("No token_id for %s %s", sig.bet_direction, m.question[:60])
            return None

        # Fetch live ask
        async with _hx.AsyncClient(timeout=8) as h:
            r    = await h.get("https://clob.polymarket.com/book",
                               params={"token_id": token_id})
            r.raise_for_status()
            asks = r.json().get("asks", [])
        if not asks:
            log.warning("Empty order book for %s", token_id[:12])
            return None
        ask = min(float(a["price"]) for a in asks)

        import math
        bet_size = cfg.get("bet_abs", 5.0)
        size     = math.ceil(bet_size / ask * 10000) / 10000
        if size < 5.0:
            log.info("SKIP: order size %.4f < 5 token minimum", size)
            return None

        # Reuse bot execution infra
        from bot.execution import _place_order_sync, _get_balance_sync
        import asyncio as _aio
        loop = _aio.get_event_loop()

        raw_bal = await loop.run_in_executor(None, _get_balance_sync)
        if raw_bal < bet_size:
            log.warning("Insufficient balance $%.2f < bet $%.2f", raw_bal, bet_size)
            return None

        ok = await loop.run_in_executor(None, _place_order_sync, token_id, ask, size)
        if ok:
            log.info("LIVE ORDER placed: %s %s @ %.3f  bet=$%.2f  tokens=%.4f",
                     sig.bet_direction, m.question[:55], ask, bet_size, size)
            return {"ask": ask, "bet": bet_size}
        return None

    except Exception as exc:
        log.error("Live execution error: %s", exc)
        return None


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Over/Below scanner stopped by user")
