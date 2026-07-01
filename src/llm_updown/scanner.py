"""
llm_updown/scanner.py — LLM-primary strategy for Polymarket up/down markets.

Supports multiple timeframes (5m / 15m / 4h) and multiple symbols (BTC / ETH / SOL).
For each active window in the entry range, asks 2 LLMs for a directional call.
Bets when LLM conviction diverges from market price by >= min_edge.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("llm_updown.scanner")

GAMMA_BASE   = "https://gamma-api.polymarket.com"
BINANCE_BASE = "https://api.binance.com/api/v3"
CLOB_BASE    = "https://clob.polymarket.com"
_HTTP        = httpx.Client(timeout=12.0)

_SYMBOL_SLUG = {"BTC": "btc", "ETH": "eth", "SOL": "sol"}
_BINANCE_SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class UDMarket:
    market_id:    str
    condition_id: str
    slug:         str
    question:     str
    symbol:       str
    timeframe:    str     # "5m", "15m", "4h"
    window_secs:  int     # total window length in seconds
    up_price:     float   # YES token price = P(UP)
    down_price:   float
    up_token_id:  str
    down_token_id: str
    window_elapsed: float  # seconds since window open
    volume_24h:   float


@dataclass
class UDSignal:
    market:        UDMarket
    current_price: float
    open_price:    float   # estimated BTC price at window open
    price_move:    float   # current - open (absolute)
    llm_prob_1:    float   # P(UP) from model 1
    llm_prob_2:    float   # P(UP) from model 2
    avg_prob:      float
    market_prob:   float   # up_price
    edge:          float   # avg_prob - market_prob
    bet_direction: str     # "YES" (bet UP) or "NO" (bet DOWN)
    model_1_name:  str
    model_2_name:  str
    reasoning_1:   str = ""
    reasoning_2:   str = ""
    conf_1:        int = 0
    conf_2:        int = 0
    entry_ask:     float = 0.5   # executable CLOB ask used for the bet side


# ── Fetch active up/down markets ───────────────────────────────────────────────

def _parse_prices(raw) -> tuple[float, float]:
    import json as _j
    if isinstance(raw, str):
        try: raw = _j.loads(raw)
        except: return 0.5, 0.5
    p = raw or []
    up   = float(p[0]) if p else 0.5
    down = float(p[1]) if len(p) > 1 else 1.0 - up
    return up, down


def _parse_tokens(raw) -> tuple[str, str]:
    import json as _j
    if isinstance(raw, str):
        try: raw = _j.loads(raw)
        except: raw = []
    if not isinstance(raw, list) or not raw:
        return "", ""
    if isinstance(raw[0], dict):
        up_id = down_id = ""
        for t in raw:
            if t.get("outcome","").lower() == "yes": up_id   = t.get("token_id","")
            if t.get("outcome","").lower() == "no":  down_id = t.get("token_id","")
        return up_id, down_id
    return str(raw[0]), str(raw[1]) if len(raw) > 1 else ""


def _clob_ask(token_id: str) -> Optional[float]:
    """Best (lowest) executable ask for a token from the CLOB book.

    This is what a taker actually PAYS. Gamma outcomePrices is a lagging mid that
    can sit 10-30% away from the book on fast up/down windows — measuring edge or
    PnL against it is fictional, because you fill here, not there.
    """
    if not token_id:
        return None
    try:
        b = _HTTP.get(f"{CLOB_BASE}/book", params={"token_id": str(token_id)}, timeout=8).json()
        asks = [float(a["price"]) for a in b.get("asks", [])]
        return min(asks) if asks else None
    except Exception as exc:
        log.debug("CLOB book fetch failed (%s): %s", str(token_id)[:8], exc)
        return None


def _window_elapsed(slug: str) -> Optional[float]:
    """Estimate seconds elapsed in the current window from the slug timestamp."""
    m = re.search(r"(\d{9,10})$", slug)
    if not m:
        return None
    window_open_ts = int(m.group(1))
    return time.time() - window_open_ts


def _current_window_timestamps(window_secs: int, look_back: int = 3) -> list[int]:
    """Return recent window start timestamps for a given window length."""
    now     = int(time.time())
    current = (now // window_secs) * window_secs
    return [current - i * window_secs for i in range(look_back)]


def _fetch_event_market(slug: str) -> Optional[dict]:
    """Fetch market via /events?slug= — same method as the main crypto loop."""
    try:
        r = _HTTP.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=8)
        r.raise_for_status()
        events = r.json()
        if not events:
            return None
        ev = events[0]
        if ev.get("closed", True):
            return None
        markets = ev.get("markets", [])
        if not markets:
            return None
        m = markets[0]
        m["_slug"] = slug
        # Propagate event-level volume to market dict (market dict lacks volume24hr)
        if "volume24hr" not in m:
            m["volume24hr"] = ev.get("volume24hr", 0)
        return m
    except Exception as exc:
        log.debug("Event fetch failed (%s): %s", slug, exc)
        return None


def fetch_ud_markets(cfg: dict) -> list[UDMarket]:
    """
    Fetch active up/down markets across all configured timeframes and symbols.
    Each timeframe has its own min/max_elapsed entry window.
    """
    symbols   = list(cfg.get("symbols", ["BTC"]))
    min_up    = cfg.get("min_up_price", 0.10)
    max_up    = cfg.get("max_up_price", 0.90)
    tf_cfg    = cfg.get("timeframes", {
        "5m": {"window_secs": 300,   "min_elapsed": 60,   "max_elapsed": 200},
    })

    results: list[UDMarket] = []
    for tf_name, tf in tf_cfg.items():
        win_secs    = int(tf.get("window_secs", 300))
        min_elapsed = float(tf.get("min_elapsed", 60))
        max_elapsed = float(tf.get("max_elapsed", 200))

        for ts in _current_window_timestamps(win_secs, look_back=3):
            elapsed = time.time() - ts
            if not (min_elapsed <= elapsed <= max_elapsed):
                continue

            for sym in symbols:
                prefix = _SYMBOL_SLUG.get(sym, sym.lower())
                slug   = f"{prefix}-updown-{tf_name}-{ts}"

                row = _fetch_event_market(slug)
                if row is None:
                    continue

                up_price, down_price = _parse_prices(row.get("outcomePrices","[0.5,0.5]"))
                if not (min_up <= up_price <= max_up):
                    continue

                up_tok, down_tok = _parse_tokens(row.get("clobTokenIds") or [])
                cid = row.get("conditionId","")
                results.append(UDMarket(
                    market_id=row.get("id", cid),
                    condition_id=cid,
                    slug=slug,
                    question=row.get("question", slug),
                    symbol=sym,
                    timeframe=tf_name,
                    window_secs=win_secs,
                    up_price=up_price,
                    down_price=down_price,
                    up_token_id=up_tok,
                    down_token_id=down_tok,
                    window_elapsed=elapsed,
                    volume_24h=float(row.get("volume24hr") or 0),
                ))

    log.info("Found %d up/down markets (%s × %s)",
             len(results), list(tf_cfg.keys()), symbols)
    return results


# ── Price context ──────────────────────────────────────────────────────────────

def _get_candles(sym: str, interval: str, limit: int = 10) -> list[dict]:
    bs = _BINANCE_SYM.get(sym)
    if not bs:
        return []
    try:
        r = _HTTP.get(f"{BINANCE_BASE}/klines",
                      params={"symbol": bs, "interval": interval, "limit": limit},
                      timeout=8)
        r.raise_for_status()
        return [{"open": float(x[1]), "high": float(x[2]),
                 "low": float(x[3]), "close": float(x[4])} for x in r.json()]
    except Exception as exc:
        log.debug("Candle fetch failed (%s %s): %s", sym, interval, exc)
        return []


def _summarize(candles: list[dict], label: str) -> str:
    if not candles:
        return f"{label}: (no data)"
    closes = [c["close"] for c in candles]
    opens  = [c["open"]  for c in candles]
    pct    = (closes[-1] / closes[0] - 1) * 100
    green  = sum(1 for o, c in zip(opens, closes) if c > o)
    if len(closes) >= 6:
        mid = len(closes) // 2
        e = (closes[mid] - closes[0]) / (closes[0] + 1e-9) * 100
        l = (closes[-1] - closes[mid]) / (closes[mid] + 1e-9) * 100
        mom = "accel" if (e != 0 and l > e * 1.5) else "decel" if (e != 0 and l < e * 0.4) else "steady"
    else:
        mom = "n/a"
    return f"{label}: ${closes[0]:,.0f}→${closes[-1]:,.0f} ({pct:+.2f}%) {green}/{len(closes)} up | {mom}"


_TF_CANDLES = {
    "5m":  [("1m", 10), ("5m", 6),  ("1h", 4)],
    "15m": [("5m", 10), ("15m", 6), ("1h", 6)],
    "4h":  [("1h", 10), ("4h", 6),  ("1d", 6)],
}
# Shortest interval per timeframe for momentum calculation
_TF_SHORT_INTERVAL = {"5m": "1m", "15m": "5m", "4h": "1h"}


def build_ud_context(market: UDMarket, current_price: float, cfg: dict) -> str:
    """Build rich technical context scaled to the window's timeframe."""
    tf       = market.timeframe
    candle_specs = cfg.get("timeframes", {}).get(tf, {}).get(
        "candles", [(s, n) for s, n in _TF_CANDLES.get(tf, _TF_CANDLES["5m"])]
    )
    # candle_specs can be list of [interval, count] from yaml or tuples from default
    def _spec(s):
        return (s[0], int(s[1])) if isinstance(s, (list, tuple)) else s

    candles_data = [(_spec(s)[0], _get_candles(market.symbol, _spec(s)[0], _spec(s)[1]))
                    for s in candle_specs]

    remaining = max(0, market.window_secs - market.window_elapsed)

    # Momentum using shortest-interval candles
    short_candles = candles_data[0][1] if candles_data else []
    win_open_price = short_candles[0]["open"] if short_candles else current_price
    win_move_pct   = (current_price / win_open_price - 1) * 100

    def bar(c): return "▲" if c["close"] >= c["open"] else "▼"
    last3 = "".join(bar(c) for c in short_candles[-3:]) if len(short_candles) >= 3 else "?"
    ret_1 = (short_candles[-1]["close"]/short_candles[-2]["close"]-1)*100 if len(short_candles)>=2 else 0.0
    ret_3 = (short_candles[-1]["close"]/short_candles[-4]["close"]-1)*100 if len(short_candles)>=4 else 0.0

    lines = [
        f"{market.symbol}/USD [{tf} window]  current=${current_price:,.2f}",
        f"Window: {market.window_elapsed:.0f}s elapsed | {remaining:.0f}s remaining of {market.window_secs}s",
        f"Since window open: {win_move_pct:+.3f}%  (open≈${win_open_price:,.2f})",
        f"Last 3 candles: {last3}",
        f"Short return: {ret_1:+.3f}%  ({_TF_SHORT_INTERVAL.get(tf,'1m')} ago)   3-bar: {ret_3:+.3f}%",
        "",
        "Price history (oldest→newest):",
    ]
    for interval, cdata in candles_data:
        lines.append(_summarize(cdata, f"{interval} candles ({len(cdata)} bars)"))
    return "\n".join(lines)


# Direction prompt — forces UP/DOWN commitment, no probability hedging.
# Confidence 1-10 encodes magnitude; converted to probability internally.
_UD_PROMPT = """\
You are a momentum trader. Make a decisive call on this {tf} {symbol} window.

{context}

The window resolves UP if {symbol} price at close > price at window open.
The window resolves DOWN if {symbol} price at close <= price at window open.

Look at the momentum signals above. Is the current move likely to CONTINUE or REVERSE?
Pick the stronger side. Even small conviction beats saying nothing.

Reply EXACTLY (no other text):
DIRECTION: UP or DOWN
CONFIDENCE: <1-10>  (5=coin flip, 7=clear momentum, 9=strong conviction)
REASONING: <one sentence, max 12 words>"""


# ── LLM caller ─────────────────────────────────────────────────────────────────

def _direction_to_prob(direction: str, confidence: int) -> float:
    """Convert UP/DOWN + confidence (1-10) to P(UP) probability.
    conf=5 → 0.50 (no edge), conf=10 → 0.80 UP or 0.20 DOWN.
    """
    delta = max(0, confidence - 5) / 5.0 * 0.30   # 0 to 0.30
    if direction.upper() == "UP":
        return min(0.95, 0.50 + delta)
    else:
        return max(0.05, 0.50 - delta)


def _parse(text: str) -> tuple[Optional[float], int, str]:
    """Parse DIRECTION/CONFIDENCE/REASONING format → (prob, conf, reasoning)."""
    direction = conf = None
    reasoning = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DIRECTION:"):
            val = line.split(":", 1)[1].strip().upper()
            direction = "UP" if "UP" in val else "DOWN" if "DOWN" in val else None
        elif line.upper().startswith("CONFIDENCE:"):
            m = re.search(r"\d+", line)
            if m: conf = max(1, min(10, int(m.group())))
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    if direction is None or conf is None:
        return None, 0, reasoning
    return _direction_to_prob(direction, conf), conf * 10, reasoning  # scale conf to 0-100


async def _call_llm(model: dict, prompt: str) -> tuple[Optional[float], int, str]:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=model["api_key"],
                             base_url=model.get("base_url","https://api.openai.com/v1"),
                             timeout=15.0)
        resp = await client.chat.completions.create(
            model=model["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=100,
        )
        return _parse(resp.choices[0].message.content or "")
    except Exception as exc:
        log.warning("LLM call failed (%s): %s", model.get("name"), exc)
        return None, 0, ""


# ── Signal generator ───────────────────────────────────────────────────────────

async def get_ud_signal(market: UDMarket, cfg: dict,
                        models: list[dict]) -> Optional[UDSignal]:
    if len(models) < 1:
        return None

    # Get current price
    bs = _BINANCE_SYM.get(market.symbol)
    if not bs:
        return None
    try:
        current_price = float(_HTTP.get(
            f"{BINANCE_BASE}/ticker/price", params={"symbol": bs}, timeout=6
        ).json()["price"])
    except Exception:
        return None

    context = build_ud_context(market, current_price, cfg)
    prompt  = _UD_PROMPT.format(context=context, symbol=market.symbol, tf=market.timeframe)

    # Call models in parallel (use up to 2)
    m1 = models[0]
    m2 = models[1] if len(models) >= 2 else None

    if m2:
        (p1, c1, r1), (p2, c2, r2) = await asyncio.gather(
            _call_llm(m1, prompt), _call_llm(m2, prompt)
        )
    else:
        p1, c1, r1 = await _call_llm(m1, prompt)
        p2, c2, r2 = p1, c1, r1   # solo model: both slots same

    if p1 is None or p2 is None:
        log.info("SKIP %s: model call failed", market.slug[-20:])
        return None

    # Models must agree within tolerance
    max_disagree = cfg.get("max_llm_disagreement", 0.25)
    if abs(p1 - p2) > max_disagree:
        log.info("SKIP %s: models disagree %.2f vs %.2f (gap=%.2f > %.2f)",
                 market.slug[-20:], p1, p2, abs(p1-p2), max_disagree)
        return None

    avg_prob    = (p1 + p2) / 2.0
    market_prob = market.up_price   # Gamma mid — kept for logging/back-compat only

    # ── Executable edge ──────────────────────────────────────────────────────
    # Compare our probability to the price we'd ACTUALLY pay on the CLOB, not the
    # lagging Gamma mid. YES is +EV only if P(UP) > ask_up; NO is +EV only if
    # P(DOWN)=1-P(UP) > ask_down. The old `avg_prob - up_price` edge was measured
    # against an untradeable price, so most of it was just Gamma-vs-CLOB lag that
    # evaporates on execution. Pick the side with the larger positive edge.
    ask_up   = _clob_ask(market.up_token_id)   or market.up_price
    ask_down = _clob_ask(market.down_token_id) or market.down_price
    yes_edge = avg_prob         - ask_up
    no_edge  = (1.0 - avg_prob) - ask_down

    min_edge = cfg.get("min_edge", 0.15)
    if yes_edge >= no_edge and yes_edge >= min_edge:
        bet_direction, edge, entry_ask = "YES", yes_edge, ask_up
    elif no_edge > yes_edge and no_edge >= min_edge:
        bet_direction, edge, entry_ask = "NO", no_edge, ask_down
    else:
        log.info("SKIP %s: no executable edge (yes=%.3f no=%.3f < %.3f | llm=%.2f ask_up=%.3f ask_dn=%.3f)",
                 market.slug[-20:], yes_edge, no_edge, min_edge, avg_prob, ask_up, ask_down)
        return None

    # Estimate open price from 5-min candles
    c5m = _get_candles(market.symbol, "5m", 2)
    open_price = c5m[0]["open"] if c5m else current_price

    log.info("SIGNAL %s %s[%s] | llm=%.2f gmid=%.2f ask=%.3f edge=%+.3f | elapsed=%.0fs | %s",
             bet_direction, market.symbol, market.timeframe,
             avg_prob, market_prob, entry_ask, edge,
             market.window_elapsed, market.question[:55])

    return UDSignal(
        market=market,
        current_price=current_price,
        open_price=open_price,
        price_move=current_price - open_price,
        llm_prob_1=p1, llm_prob_2=p2,
        avg_prob=avg_prob,
        market_prob=market_prob,
        edge=edge,
        bet_direction=bet_direction,
        model_1_name=m1["name"],
        model_2_name=m2["name"] if m2 else m1["name"],
        reasoning_1=r1, reasoning_2=r2,
        conf_1=c1, conf_2=c2,
        entry_ask=entry_ask,
    )


# ── Full scan ──────────────────────────────────────────────────────────────────

async def scan_once(cfg: dict, models: list[dict],
                    skip_ids: set[str],
                    evaluated_slugs: set[str]) -> list[UDSignal]:
    """
    evaluated_slugs: slugs already called this window — prevents re-calling
    LLMs on the same window every 20s loop cycle.
    """
    markets = fetch_ud_markets(cfg)
    candidates = [m for m in markets
                  if m.market_id not in skip_ids
                  and m.slug not in evaluated_slugs]
    log.info("Evaluating %d/%d candidates (dedup skipped %d)",
             len(candidates), len(markets),
             len(markets) - len(candidates))

    signals: list[UDSignal] = []
    for market in candidates:
        evaluated_slugs.add(market.slug)
        sig = await get_ud_signal(market, cfg, models)
        if sig is not None:
            signals.append(sig)
    return signals
