"""
llm_updown_main.py — LLM-primary strategy for Polymarket 5-minute up/down markets.

Asks 2 LLMs for a probability estimate on each active window.
Bets when LLM diverges from market price by >= min_edge.

Run:
    screen -dmS llm_ud bash -c 'source .venv/bin/activate && python src/llm_updown_main.py >> logs/llm_updown.log 2>&1'
"""
from __future__ import annotations

import asyncio, logging, logging.handlers, os, sys, time
from pathlib import Path

_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
for _p in [_ROOT, _ROOT / "polymarket-mcp-main" / "polymarket-mcp-main"]:
    if (_p / ".env").exists(): load_dotenv(_p / ".env"); break

_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_h = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "llm_updown.log", maxBytes=10*1024*1024, backupCount=3)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_h, logging.StreamHandler(sys.stdout)])
log = logging.getLogger("llm_updown_main")

_CONFIG_PATH = _ROOT / "config" / "llm_updown_params.yaml"


def _load_cfg() -> dict:
    defaults = {
        "virtual_mode": True,
        "scan_interval": 20,
        "symbols": ["BTC"],
        "min_elapsed": 60.0,
        "max_elapsed": 200.0,
        "min_edge": 0.15,
        "max_llm_disagreement": 0.25,
        "min_volume_24h": 100.0,
        "min_up_price": 0.10,
        "max_up_price": 0.90,
        "bet_abs": 3.0,
        "max_open_positions": 3,
    }
    try:
        import yaml
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return {**defaults, **(yaml.safe_load(f) or {})}
    except Exception:
        return defaults


def _get_models() -> list[dict]:
    candidates = [
        ("deepseek",    "DEEPSEEK_API_KEY", "deepseek-chat",  "https://api.deepseek.com/v1"),
        ("gpt-4o-mini", "OPENAI_API_KEY",   "gpt-4o-mini",   "https://api.openai.com/v1"),
    ]
    models = []
    for name, key_var, model_id, base_url in candidates:
        key = os.getenv(key_var)
        if key:
            models.append({"name": name, "api_key": key,
                           "model": model_id, "base_url": base_url})
    return models


async def run() -> None:
    cfg    = _load_cfg()
    models = _get_models()
    virtual = (os.getenv("LLM_UPDOWN_VIRTUAL_MODE", "true").lower() != "false"
               and cfg.get("virtual_mode", True))

    log.info("LLM UpDown started | mode=%s | symbols=%s | window=[%.0f-%.0fs] | "
             "min_edge=%.2f | models=%s",
             "PAPER" if virtual else "LIVE",
             cfg["symbols"], cfg["min_elapsed"], cfg["max_elapsed"],
             cfg["min_edge"], [m["name"] for m in models])

    if not models:
        log.error("No models with API keys — exiting"); return

    from llm_updown.scanner import scan_once
    from llm_updown.paper   import (load_trades, save_trades, load_executed,
                                    save_executed, prune_executed,
                                    record_trade, settle_open_trades, print_summary)

    trades          = load_trades()
    executed        = prune_executed(load_executed())
    evaluated_slugs: set[str] = set()   # cleared each new 5-min window
    last_window_ts  = 0

    while True:
        # Clear per-window dedup when a new 5-min window opens
        current_window = (int(time.time()) // 300) * 300
        if current_window != last_window_ts:
            evaluated_slugs.clear()
            last_window_ts = current_window

        # Settle resolved windows
        n = settle_open_trades(trades)
        if n:
            save_trades(trades)
            print_summary(trades)

        # Check open slot
        open_n   = sum(1 for t in trades if not t.settled)
        max_open = cfg.get("max_open_positions", 3)

        if open_n < max_open:
            skip_ids = set(executed.keys())
            signals  = await scan_once(cfg, models, skip_ids, evaluated_slugs)

            for sig in signals:
                if sum(1 for t in trades if not t.settled) >= max_open:
                    break

                bet_size = cfg.get("bet_abs", 3.0)
                log.info("=== %s === %s %s | llm=%.2f mkt=%.2f edge=%+.3f "
                         "elapsed=%.0fs | %s=%.2f %s=%.2f",
                         "PAPER" if virtual else "LIVE",
                         sig.bet_direction, sig.market.symbol,
                         sig.avg_prob, sig.market_prob, sig.edge,
                         sig.market.window_elapsed,
                         sig.model_1_name, sig.llm_prob_1,
                         sig.model_2_name, sig.llm_prob_2)

                if virtual:
                    # Append to the in-memory list so the settler (which only
                    # sees `trades`) can resolve trades recorded after startup.
                    trades.append(record_trade(sig, bet_size))
                else:
                    result = await _execute_live(sig, bet_size)
                    if result:
                        # Record the ACTUAL fill price, not the Gamma mid.
                        trades.append(record_trade(sig, result["bet"], fill_ask=result["ask"]))
                    else:
                        continue

                executed[sig.market.market_id] = time.time()
                executed = prune_executed(executed)
                save_executed(executed)
        else:
            log.info("Open cap full (%d/%d) — skipping", open_n, max_open)

        await asyncio.sleep(cfg["scan_interval"])


async def _execute_live(sig, bet_size: float) -> dict | None:
    try:
        import httpx as _hx, math
        m        = sig.market
        token_id = m.up_token_id if sig.bet_direction == "YES" else m.down_token_id
        if not token_id:
            log.warning("No token_id for %s", m.question[:55]); return None

        async with _hx.AsyncClient(timeout=8) as h:
            r    = await h.get("https://clob.polymarket.com/book", params={"token_id": token_id})
            asks = r.json().get("asks", [])
        if not asks: return None
        ask  = min(float(a["price"]) for a in asks)
        size = math.ceil(bet_size / ask * 10000) / 10000
        if size < 5.0: return None

        from bot.execution import _place_order_sync, _get_balance_sync
        loop = asyncio.get_event_loop()
        bal  = await loop.run_in_executor(None, _get_balance_sync)
        if bal < bet_size: return None
        ok = await loop.run_in_executor(None, _place_order_sync, token_id, ask, size)
        return {"ask": ask, "bet": bet_size} if ok else None
    except Exception as e:
        log.error("Live exec error: %s", e); return None


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Stopped")
