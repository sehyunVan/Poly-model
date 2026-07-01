"""
over_below/scanner.py — LLM-powered scanner for Polymarket over/below price markets.

Targets markets asking whether a crypto asset will cross a specific price threshold
by a given date (e.g. "Will BTC be above $70,000 on June 30?").

Strategy:
1. Scan Gamma API for over/below price markets on BTC/ETH/SOL
2. Build multi-timeframe Binance price context (5m / 1h / 1d candles)
3. Ask 2 LLMs independently for a probability estimate (0–100)
4. When both agree within max_disagreement AND diverge from market price
   by > min_edge, generate a bet signal

Paper mode: the main loop reads signals and logs them; execution is separate.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

log = logging.getLogger("over_below.scanner")

GAMMA_BASE  = "https://gamma-api.polymarket.com"
_HTTP       = httpx.Client(timeout=15.0)

# ── Symbol mapping ─────────────────────────────────────────────────────────────

_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum", "ether"],
    "SOL": ["sol", "solana"],
    "BNB": ["bnb"],
    "XRP": ["xrp", "ripple"],
}

_BINANCE_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}

# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class OBMarket:
    """Parsed over/below Polymarket market."""
    market_id:    str
    condition_id: str
    slug:         str
    question:     str
    symbol:       str        # "BTC", "ETH", etc.
    direction:    str        # "above" or "below"
    threshold:    float      # price level, e.g. 70000.0
    yes_price:    float      # Gamma yes token price (market consensus)
    no_price:     float
    yes_token_id: str
    no_token_id:  str
    volume_24h:   float
    hours_to_close: float


@dataclass
class OBSignal:
    """LLM consensus signal — bet candidate."""
    market:       OBMarket
    current_price: float
    move_pct:     float      # % move required from current to threshold
    llm_prob_1:   float      # probability from model 1 (0–1)
    llm_prob_2:   float      # probability from model 2 (0–1)
    avg_prob:     float      # (prob1 + prob2) / 2
    market_prob:  float      # market yes_price (what we're comparing against)
    edge:         float      # avg_prob - market_prob (positive = bet YES, negative = bet NO)
    bet_direction: str       # "YES" or "NO"
    model_1_name: str
    model_2_name: str
    reasoning_1:  str = ""
    reasoning_2:  str = ""
    conf_1:       int = 0
    conf_2:       int = 0


# ── Market title parser ────────────────────────────────────────────────────────

_ABOVE_WORDS = {"above", "reach", "exceed", "over", "surpass", "hit", "break",
                "cross", "top", "at least", "or higher"}
_BELOW_WORDS = {"below", "under", "fall", "drop", "decline", "less than",
                "not above", "or lower"}


def parse_ob_question(question: str) -> Optional[tuple[str, str, float]]:
    """
    Parse a market title into (symbol, direction, threshold).
    Returns None if the market is not a clean over/below price question.
    """
    q = question.lower()

    # Identify symbol
    symbol = None
    for sym, keywords in _SYMBOL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            symbol = sym
            break
    if symbol is None:
        return None

    # Extract threshold price — first $X,XXX or $X pattern (without consuming k suffix)
    m = re.search(r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|\d+(?:\.[0-9]+)?)', question)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    threshold = float(raw)
    # Handle shorthand: "$70k" / "$100K" → multiply by 1000
    suffix = question[m.end():m.end()+1].lower()
    if suffix == "k":
        threshold *= 1000
    elif suffix == "m":
        threshold *= 1_000_000
    if threshold < 1:
        return None  # probably a probability, not a price

    # Identify direction — "above or below" is ambiguous, skip
    has_above = any(w in q for w in _ABOVE_WORDS)
    has_below = any(w in q for w in _BELOW_WORDS)
    if has_above and has_below:
        return None   # "above or below X" — can't determine direction
    elif has_above:
        direction = "above"
    elif has_below:
        direction = "below"
    else:
        return None

    return symbol, direction, threshold


# ── Gamma market fetch + filter ────────────────────────────────────────────────

def _parse_prices(raw_prices) -> tuple[float, float]:
    import json as _j
    if isinstance(raw_prices, str):
        try:
            prices = _j.loads(raw_prices)
        except Exception:
            return 0.5, 0.5
    else:
        prices = raw_prices or []
    yes = float(prices[0]) if prices else 0.5
    no  = float(prices[1]) if len(prices) > 1 else 1.0 - yes
    return yes, no


def _parse_tokens(raw_tokens) -> tuple[str, str]:
    import json as _j
    if isinstance(raw_tokens, str):
        try:
            raw_tokens = _j.loads(raw_tokens)
        except Exception:
            raw_tokens = []
    if not isinstance(raw_tokens, list) or not raw_tokens:
        return "", ""
    if isinstance(raw_tokens[0], dict):
        yes_id = no_id = ""
        for tok in raw_tokens:
            outcome = tok.get("outcome", "").lower()
            if outcome == "yes":
                yes_id = tok.get("token_id", "")
            elif outcome == "no":
                no_id = tok.get("token_id", "")
        return yes_id, no_id
    yes_id = str(raw_tokens[0]) if len(raw_tokens) > 0 else ""
    no_id  = str(raw_tokens[1]) if len(raw_tokens) > 1 else ""
    return yes_id, no_id


def _hours_to_close(end_date_str: str) -> float:
    try:
        end   = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        delta = end - datetime.now(tz=timezone.utc)
        return delta.total_seconds() / 3600
    except Exception:
        return -1.0


def fetch_ob_markets(
    cfg: dict,
    total_limit: int = 500,
) -> list[OBMarket]:
    """
    Fetch active markets from Gamma API (paginated), parse titles for over/below
    patterns, and return filtered OBMarket list.
    Gamma caps each response at ~100 markets, so we paginate with offset.
    """
    symbols  = set(cfg.get("symbols", ["BTC", "ETH", "SOL"]))
    min_h    = cfg.get("min_hours", 2.0)
    max_h    = cfg.get("max_hours", 720.0)
    min_vol  = cfg.get("min_volume_24h", 100.0)   # lower floor for OB markets
    min_yes  = cfg.get("min_yes_price", 0.10)
    max_yes  = cfg.get("max_yes_price", 0.90)

    raw_list: list[dict] = []
    page_size = 100
    offset    = 0
    while len(raw_list) < total_limit:
        try:
            resp = _HTTP.get(
                f"{GAMMA_BASE}/markets",
                params={"active": "true", "closed": "false",
                        "limit": page_size, "offset": offset,
                        "order": "volume24hr", "ascending": "false"},
            )
            resp.raise_for_status()
            page = resp.json()
        except Exception as exc:
            log.error("Gamma API fetch failed (offset=%d): %s", offset, exc)
            break
        if not page:
            break
        raw_list.extend(page)
        if len(page) < page_size:
            break   # last page
        offset += page_size

    results: list[OBMarket] = []
    for raw in raw_list:
        question = (raw.get("question") or "").strip()
        if not question:
            continue

        parsed = parse_ob_question(question)
        if parsed is None:
            continue
        sym, direction, threshold = parsed
        if sym not in symbols:
            continue

        end_date = raw.get("endDate", "")
        hours    = _hours_to_close(end_date) if end_date else -1.0
        if not (min_h <= hours <= max_h):
            continue

        vol = float(raw.get("volume24hr") or raw.get("volume24h") or 0)
        if vol < min_vol:
            continue

        yes_price, no_price = _parse_prices(raw.get("outcomePrices", "[0.5,0.5]"))
        if not (min_yes <= yes_price <= max_yes):
            continue

        yes_tok, no_tok = _parse_tokens(raw.get("clobTokenIds") or raw.get("tokens") or [])
        condition_id    = raw.get("conditionId", "")
        market_id       = raw.get("id", condition_id)
        slug            = raw.get("slug", "")

        results.append(OBMarket(
            market_id=market_id,
            condition_id=condition_id,
            slug=slug,
            question=question,
            symbol=sym,
            direction=direction,
            threshold=threshold,
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=yes_tok,
            no_token_id=no_tok,
            volume_24h=vol,
            hours_to_close=hours,
        ))

    log.info("Parsed %d over/below markets from %d fetched (min_vol=$%.0f)",
             len(results), len(raw_list), min_vol)
    return results


# ── Price context builder ──────────────────────────────────────────────────────

def _get_binance_price(symbol: str) -> Optional[float]:
    bs = _BINANCE_SYMBOL.get(symbol)
    if not bs:
        return None
    try:
        r = _HTTP.get("https://api.binance.com/api/v3/ticker/price",
                      params={"symbol": bs}, timeout=6)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as exc:
        log.warning("Binance price fetch failed (%s): %s", symbol, exc)
        return None


def _get_candles(symbol: str, interval: str, limit: int) -> list[dict]:
    bs = _BINANCE_SYMBOL.get(symbol)
    if not bs:
        return []
    try:
        r = _HTTP.get("https://api.binance.com/api/v3/klines",
                      params={"symbol": bs, "interval": interval, "limit": limit},
                      timeout=10)
        r.raise_for_status()
        candles = []
        for row in r.json():
            candles.append({
                "open": float(row[1]),
                "high": float(row[2]),
                "low":  float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return candles
    except Exception as exc:
        log.warning("Candle fetch failed (%s %s): %s", symbol, interval, exc)
        return []


def _summarize_candles(candles: list[dict], label: str) -> str:
    if not candles:
        return f"{label}: (no data)"
    closes = [c["close"] for c in candles]
    opens  = [c["open"]  for c in candles]
    pct    = (closes[-1] / closes[0] - 1) * 100
    green  = sum(1 for o, c in zip(opens, closes) if c > o)

    # Momentum: compare late vs early rate of change
    if len(closes) >= 6:
        mid   = len(closes) // 2
        early = (closes[mid] - closes[0]) / (closes[0] + 1e-9) * 100
        late  = (closes[-1] - closes[mid]) / (closes[mid] + 1e-9) * 100
        if early == 0:
            mom = "steady"
        elif late > early * 1.5:
            mom = "accelerating"
        elif late < early * 0.4:
            mom = "decelerating"
        else:
            mom = "steady"
    else:
        mom = "n/a"

    return (f"{label}: ${closes[0]:,.0f}→${closes[-1]:,.0f} "
            f"({pct:+.2f}%) | {green}/{len(closes)} up | momentum={mom}")


def build_price_context(
    symbol: str,
    current_price: float,
    threshold: float,
    direction: str,
    days_left: float,
    market_yes_price: float,
) -> str:
    """Fetch multi-scale candles and format as LLM-readable text."""
    c5m = _get_candles(symbol, "5m", 10)
    c1h = _get_candles(symbol, "1h", 10)
    c1d = _get_candles(symbol, "1d", 10)

    move_pct = (threshold / current_price - 1) * 100
    hours_left = days_left * 24

    # Daily trend: average % change per day over the last 10 days
    if c1d and len(c1d) >= 2:
        daily_pct = (c1d[-1]["close"] / c1d[0]["close"] - 1) * 100 / max(len(c1d) - 1, 1)
    else:
        daily_pct = 0.0

    projected = current_price * (1 + daily_pct / 100) ** days_left if days_left > 0 else current_price

    lines = [
        f"MARKET: Will {symbol} be {direction} ${threshold:,.0f} by "
        f"{'~' + str(int(days_left)) + 'd' if days_left >= 1 else str(int(hours_left)) + 'h'}?",
        "",
        f"Current {symbol}: ${current_price:,.2f}",
        f"Required move: {move_pct:+.1f}% ({direction} ${threshold:,.0f})",
        f"Time remaining: {days_left:.1f} days",
        f"At current 10-day pace ({daily_pct:+.2f}%/day): projected ${projected:,.0f}",
        "",
        "Price history (Binance, oldest→newest, 10 data points each):",
        _summarize_candles(c5m, "Last 50min (5m candles)"),
        _summarize_candles(c1h, "Last 10h  (1h candles)"),
        _summarize_candles(c1d, "Last 10d  (1d candles)"),
        "",
        f"Market currently prices this at {market_yes_price*100:.0f}%. "
        "Form your OWN independent estimate — do not anchor to this number.",
    ]
    return "\n".join(lines)


_LLM_PROMPT = """\
You are a quantitative analyst estimating a probability for a crypto prediction market.

{context}

Based ONLY on the price data above, estimate the probability (0–100) that this market resolves YES.

Key considerations:
- How large is the required move relative to recent volatility?
- Is the current momentum pointing toward or away from the threshold?
- Is there enough time for the move to occur at the current pace?

Reply in EXACTLY this format (no other text):
PROBABILITY: <integer 0-100>
CONFIDENCE: <integer 0-100>
REASONING: <one sentence, max 20 words>"""


# ── LLM caller ─────────────────────────────────────────────────────────────────

def _parse_llm_response(text: str) -> tuple[Optional[float], int, str]:
    """Return (probability 0-1, confidence 0-100, reasoning)."""
    prob = conf = None
    reasoning = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("PROBABILITY:"):
            m = re.search(r"\d+", line)
            if m:
                val = int(m.group())
                prob = max(0, min(100, val)) / 100.0
        elif line.upper().startswith("CONFIDENCE:"):
            m = re.search(r"\d+", line)
            if m:
                conf = max(0, min(100, int(m.group())))
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return prob, conf or 0, reasoning


async def _call_llm(model_cfg: dict, prompt: str) -> tuple[Optional[float], int, str]:
    """Call one LLM, return (prob, confidence, reasoning)."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg.get("base_url", "https://api.openai.com/v1"),
            timeout=25.0,
        )
        resp = await client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        text = resp.choices[0].message.content or ""
        return _parse_llm_response(text)
    except Exception as exc:
        log.warning("LLM call failed (%s): %s", model_cfg.get("name"), exc)
        return None, 0, ""


# ── Signal generator ───────────────────────────────────────────────────────────

async def get_llm_signal(
    market: OBMarket,
    cfg: dict,
    models: list[dict],
) -> Optional[OBSignal]:
    """
    Fetch current price, build context, call 2 LLMs in parallel.
    Returns OBSignal if there's a tradeable edge, None otherwise.
    """
    if len(models) < 2:
        log.warning("Need at least 2 active models for consensus; got %d", len(models))
        return None

    current_price = _get_binance_price(market.symbol)
    if current_price is None:
        return None

    days_left = market.hours_to_close / 24.0
    move_pct  = (market.threshold / current_price - 1) * 100

    # Skip markets where the required move is impossibly large or already done
    max_move_pct = cfg.get("max_move_pct", 60.0)
    if abs(move_pct) > max_move_pct:
        log.info("SKIP %s: move %.1f%% exceeds max %.1f%%",
                 market.question[:60], move_pct, max_move_pct)
        return None

    # Skip already-resolved direction: price already past the threshold
    if market.direction == "above" and current_price >= market.threshold:
        log.info("SKIP %s: BTC $%.0f already above $%.0f",
                 market.question[:60], current_price, market.threshold)
        return None
    if market.direction == "below" and current_price <= market.threshold:
        log.info("SKIP %s: BTC $%.0f already below $%.0f",
                 market.question[:60], current_price, market.threshold)
        return None

    log.info("EVALUATING %s | price=$%.0f threshold=$%.0f move=%.1f%% ttc=%.1fh mkt=%.3f",
             market.question[:60], current_price, market.threshold,
             move_pct, market.hours_to_close, market.yes_price)

    context = build_price_context(
        market.symbol, current_price, market.threshold,
        market.direction, days_left, market.yes_price,
    )
    prompt = _LLM_PROMPT.format(context=context)

    # Call models in parallel — use first two active models
    m1, m2 = models[0], models[1]
    (p1, c1, r1), (p2, c2, r2) = await asyncio.gather(
        _call_llm(m1, prompt),
        _call_llm(m2, prompt),
    )

    if p1 is None or p2 is None:
        log.info("SKIP %s: one or both models failed", market.question[:60])
        return None

    # Check that models agree within tolerance
    max_disagree = cfg.get("max_llm_disagreement", 0.20)
    if abs(p1 - p2) > max_disagree:
        log.info(
            "SKIP %s: model disagreement %.2f > %.2f (m1=%.2f m2=%.2f)",
            market.question[:60], abs(p1 - p2), max_disagree, p1, p2,
        )
        return None

    avg_prob    = (p1 + p2) / 2.0
    market_prob = market.yes_price
    edge        = avg_prob - market_prob

    min_edge = cfg.get("min_edge", 0.12)
    if abs(edge) < min_edge:
        log.info(
            "SKIP %s: edge %.3f < %.3f (llm=%.2f mkt=%.2f)",
            market.question[:60], abs(edge), min_edge, avg_prob, market_prob,
        )
        return None

    bet_direction = "YES" if edge > 0 else "NO"
    log.info(
        "SIGNAL %s %s | llm=%.2f mkt=%.2f edge=%+.3f | m1=%.2f m2=%.2f",
        bet_direction, market.question[:55], avg_prob, market_prob,
        edge, p1, p2,
    )

    return OBSignal(
        market=market,
        current_price=current_price,
        move_pct=move_pct,
        llm_prob_1=p1,
        llm_prob_2=p2,
        avg_prob=avg_prob,
        market_prob=market_prob,
        edge=edge,
        bet_direction=bet_direction,
        model_1_name=m1["name"],
        model_2_name=m2["name"],
        reasoning_1=r1,
        reasoning_2=r2,
        conf_1=c1,
        conf_2=c2,
    )


# ── Full scan cycle ────────────────────────────────────────────────────────────

async def scan_once(cfg: dict, models: list[dict],
                    skip_ids: set[str]) -> list[OBSignal]:
    """
    Run one scan cycle: fetch markets, filter, call LLMs, return signals.
    skip_ids: market_ids already bet on this session (dedup).
    """
    markets = fetch_ob_markets(cfg)
    if not markets:
        return []

    # Exclude already-executed markets
    candidates = [m for m in markets if m.market_id not in skip_ids]
    log.info("Evaluating %d/%d candidates (after dedup)", len(candidates), len(markets))

    signals: list[OBSignal] = []
    max_per_scan = cfg.get("max_per_scan", 10)
    for market in candidates[:max_per_scan]:
        sig = await get_llm_signal(market, cfg, models)
        if sig is not None:
            signals.append(sig)

    return signals
