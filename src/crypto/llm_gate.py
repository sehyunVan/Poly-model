"""
crypto/llm_gate.py — Background LLM direction signal for the 5-minute crypto loop.

Runs as a daemon thread. Every refresh_interval seconds, calls 2 LLMs in parallel
with multi-scale Binance price context (5m / 1h / 1d) and caches the directional
verdict (UP / DOWN / UNCERTAIN) per symbol.

Two modes (controlled by llm_gate_enabled in crypto_params.yaml):

  Ghost mode (default, llm_gate_enabled=false):
    The gate never blocks trades.  loop.py still records llm_direction and
    llm_agrees in every signal_log entry for post-hoc WR analysis.

  Gate mode (llm_gate_enabled=true):
    When both LLMs return a direction that OPPOSES the flow signal AND both have
    confidence >= llm_gate_min_conf, the trade is skipped.
    Only activates after enough ghost data confirms LLM-agree WR lift.

Integration in loop.py:
    from crypto.llm_gate import llm_gate as _llm_gate

    # startup (after other feeds):
    _llm_gate.start(models, refresh_interval=60)

    # at execution decision:
    llm_dir, llm_conf = _llm_gate.get_signal("BTC", max_age_s=120)
    llm_agrees = (llm_dir == sig.direction) if llm_dir else None
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from typing import Optional

log = logging.getLogger("crypto.llm_gate")

_BINANCE_BASE = "https://api.binance.com/api/v3"
_BINANCE_SYM  = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = """\
You are a quantitative analyst assessing BTC/USD short-term momentum.

Price context (Binance spot, oldest→newest, 10 data points each):
{ctx_5m}
{ctx_1h}
{ctx_1d}

Current BTC price: ${current_price:,.0f}

Question: Over the NEXT 5 MINUTES, is BTC price more likely to go UP or DOWN?
Use the 5-minute candles for momentum and 1h/1d to confirm or contradict the trend.
If you genuinely cannot tell, say UNCERTAIN.

Reply EXACTLY (no other text):
DIRECTION: UP or DOWN or UNCERTAIN
CONFIDENCE: <integer 0-100>
REASONING: <one sentence, max 15 words>"""


# ── Context builder (sync, called from daemon thread) ─────────────────────────

def _fetch_candles(binance_sym: str, interval: str, limit: int = 10) -> list[dict]:
    try:
        import httpx
        r = httpx.get(f"{_BINANCE_BASE}/klines",
                      params={"symbol": binance_sym, "interval": interval, "limit": limit},
                      timeout=8)
        r.raise_for_status()
        return [{"open": float(row[1]), "high": float(row[2]),
                 "low": float(row[3]), "close": float(row[4])} for row in r.json()]
    except Exception as exc:
        log.debug("Candle fetch failed (%s %s): %s", binance_sym, interval, exc)
        return []


def _summarize(candles: list[dict], label: str) -> str:
    if not candles:
        return f"{label}: (no data)"
    closes = [c["close"] for c in candles]
    opens  = [c["open"]  for c in candles]
    pct    = (closes[-1] / closes[0] - 1) * 100
    green  = sum(1 for o, c in zip(opens, closes) if c > o)
    if len(closes) >= 6:
        mid   = len(closes) // 2
        e = (closes[mid] - closes[0]) / (closes[0] + 1e-9) * 100
        l = (closes[-1] - closes[mid]) / (closes[mid] + 1e-9) * 100
        mom = "accel" if (e != 0 and l > e * 1.5) else "decel" if (e != 0 and l < e * 0.4) else "steady"
    else:
        mom = "n/a"
    return f"{label}: ${closes[0]:,.0f}→${closes[-1]:,.0f} ({pct:+.2f}%) {green}/{len(closes)} up | {mom}"


def _build_context(symbol: str) -> Optional[str]:
    bs = _BINANCE_SYM.get(symbol)
    if not bs:
        return None
    try:
        import httpx
        price = float(httpx.get(f"{_BINANCE_BASE}/ticker/price",
                                params={"symbol": bs}, timeout=6).json()["price"])
    except Exception:
        return None
    c5m = _fetch_candles(bs, "5m",  10)
    c1h = _fetch_candles(bs, "1h",  10)
    c1d = _fetch_candles(bs, "1d",  10)
    return _PROMPT.format(
        ctx_5m=_summarize(c5m, "Last 50min (5m candles)"),
        ctx_1h=_summarize(c1h, "Last 10h  (1h candles)"),
        ctx_1d=_summarize(c1d, "Last 10d  (1d candles)"),
        current_price=price,
    )


# ── LLM caller (async) ────────────────────────────────────────────────────────

def _parse(text: str) -> tuple[Optional[str], int, str]:
    direction = conf = None
    reasoning = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DIRECTION:"):
            val = line.split(":", 1)[1].strip().upper()
            if "UP" in val and "DOWN" not in val:
                direction = "UP"
            elif "DOWN" in val:
                direction = "DOWN"
            else:
                direction = None   # UNCERTAIN
        elif line.upper().startswith("CONFIDENCE:"):
            m = re.search(r"\d+", line)
            if m:
                conf = max(0, min(100, int(m.group())))
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return direction, conf or 0, reasoning


async def _call_llm(model: dict, prompt: str) -> tuple[Optional[str], int, str]:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=model["api_key"],
            base_url=model.get("base_url", "https://api.openai.com/v1"),
            timeout=20.0,
        )
        resp = await client.chat.completions.create(
            model=model["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
        )
        return _parse(resp.choices[0].message.content or "")
    except Exception as exc:
        log.debug("LLM call failed (%s): %s", model.get("name"), exc)
        return None, 0, ""


async def _refresh_async(symbol: str, models: list[dict]) -> tuple[Optional[str], int, str]:
    """Call 2 models in parallel, return consensus direction."""
    prompt = _build_context(symbol)
    if prompt is None:
        return None, 0, ""

    if len(models) < 2:
        d, c, r = await _call_llm(models[0], prompt)
        return d, c, r

    (d1, c1, r1), (d2, c2, r2) = await asyncio.gather(
        _call_llm(models[0], prompt),
        _call_llm(models[1], prompt),
    )

    log.info("LLM gate refresh %s: %s=%s(c%d) %s=%s(c%d)",
             symbol,
             models[0]["name"], d1 or "UNC", c1,
             models[1]["name"], d2 or "UNC", c2)

    # Both must agree on the same non-None direction
    if d1 is None or d2 is None or d1 != d2:
        return None, 0, f"{r1} / {r2}"

    avg_conf = (c1 + c2) // 2
    return d1, avg_conf, f"{r1} / {r2}"


# ── Gate class ────────────────────────────────────────────────────────────────

class LLMGate:
    """Background daemon that refreshes LLM direction per symbol every N seconds."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple] = {}   # symbol → (direction, conf, reasoning, ts)
        self._lock    = threading.Lock()
        self._models:  list[dict]    = []
        self._symbols: list[str]     = ["BTC"]
        self._interval: int          = 60
        self._thread: Optional[threading.Thread] = None
        self._loop:   Optional[asyncio.AbstractEventLoop] = None

    def start(self, models: list[dict], symbols: list[str] = None,
              refresh_interval: int = 60) -> None:
        if not models:
            log.warning("LLMGate.start: no models — gate disabled")
            return
        self._models   = models
        self._symbols  = symbols or ["BTC"]
        self._interval = refresh_interval
        self._thread   = threading.Thread(target=self._run, daemon=True,
                                           name="llm_gate")
        self._thread.start()
        log.info("LLM gate started | symbols=%s models=%s interval=%ds",
                 self._symbols, [m["name"] for m in models], refresh_interval)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        while True:
            for sym in self._symbols:
                try:
                    d, c, r = self._loop.run_until_complete(
                        _refresh_async(sym, self._models)
                    )
                    with self._lock:
                        self._cache[sym] = (d, c, r, time.monotonic())
                except Exception as exc:
                    log.debug("LLMGate refresh error (%s): %s", sym, exc)
            time.sleep(self._interval)

    def get_signal(self, symbol: str, max_age_s: int = 120) -> tuple[Optional[str], int]:
        """
        Return (direction, confidence) from cache.
        direction is "UP", "DOWN", or None (uncertain / stale / unavailable).
        """
        with self._lock:
            entry = self._cache.get(symbol)
        if entry is None:
            return None, 0
        d, c, _r, ts = entry
        if time.monotonic() - ts > max_age_s:
            return None, 0   # stale
        return d, c

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# Module singleton — imported by loop.py
llm_gate = LLMGate()
