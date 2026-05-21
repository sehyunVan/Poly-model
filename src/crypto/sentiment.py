"""
Crypto sentiment signals:
  1. Alternative.me Fear & Greed Index (0=extreme fear, 100=extreme greed)
  2. RSS headlines from CoinDesk + CoinTelegraph → LLM sentiment score
     (CryptoPanic now requires a paid API key; replaced with free RSS feeds)
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

log = logging.getLogger("crypto.sentiment")

_CLIENT = httpx.Client(timeout=10.0)

# ── Fear & Greed ──────────────────────────────────────────────────────────────

_fg_cache: dict = {"value": None, "ts": 0.0}
_FG_TTL = 300  # refresh every 5 minutes


def get_fear_greed() -> Optional[float]:
    """
    Returns Fear & Greed index 0–100, normalised to 0.0–1.0.
    Cached for 5 minutes.
    """
    now = time.time()
    if _fg_cache["value"] is not None and now - _fg_cache["ts"] < _FG_TTL:
        return _fg_cache["value"]
    try:
        resp = _CLIENT.get("https://api.alternative.me/fng/?limit=1")
        resp.raise_for_status()
        raw = int(resp.json()["data"][0]["value"])
        normalised = raw / 100.0
        _fg_cache["value"] = normalised
        _fg_cache["ts"] = now
        return normalised
    except Exception as exc:
        log.warning("Fear&Greed fetch failed: %s", exc)
        return _fg_cache["value"]  # return stale if available


# ── RSS headline fetcher (CoinDesk + CoinTelegraph) ──────────────────────────

_RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]

_news_cache: dict = {"headlines": [], "ts": 0.0}
_NEWS_TTL = 120  # refresh every 2 minutes


def get_recent_headlines(limit: int = 10) -> list[str]:
    """
    Fetch recent crypto news headlines from free RSS feeds.
    Falls back to cached/empty list on error.
    """
    now = time.time()
    if _news_cache["headlines"] and now - _news_cache["ts"] < _NEWS_TTL:
        return _news_cache["headlines"]

    headlines: list[str] = []
    for feed_url in _RSS_FEEDS:
        try:
            resp = _CLIENT.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            # RSS <item><title> or Atom <entry><title>
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item/title") or root.findall(".//atom:entry/atom:title", ns)
            for el in items[:limit]:
                if el.text and el.text.strip():
                    headlines.append(el.text.strip())
            if headlines:
                break  # one working feed is enough
        except Exception as exc:
            log.debug("RSS feed %s failed: %s", feed_url, exc)

    if headlines:
        _news_cache["headlines"] = headlines[:limit]
        _news_cache["ts"] = now
    return _news_cache["headlines"]


# ── LLM sentiment scorer ──────────────────────────────────────────────────────

_sentiment_cache: dict = {"score": None, "ts": 0.0}
_SENTIMENT_TTL = 120  # same cadence as news


def get_llm_sentiment(headlines: list[str]) -> float:
    """
    Ask Claude to score current crypto market sentiment.
    Returns -1.0 (very bearish) to +1.0 (very bullish), 0.0 = neutral.
    Cached for 2 minutes.
    """
    import anthropic

    now = time.time()
    if _sentiment_cache["score"] is not None and now - _sentiment_cache["ts"] < _SENTIMENT_TTL:
        return _sentiment_cache["score"]

    if not headlines:
        return 0.0

    try:
        client = anthropic.Anthropic()
        headline_text = "\n".join(f"- {h}" for h in headlines[:10])
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": (
                    "Given these recent crypto news headlines, rate the overall "
                    "short-term market sentiment for Bitcoin on a scale from -1.0 "
                    "(very bearish) to +1.0 (very bullish). "
                    "Reply with ONLY a number between -1.0 and 1.0.\n\n"
                    f"{headline_text}"
                ),
            }],
        )
        raw = msg.content[0].text.strip()
        score = max(-1.0, min(1.0, float(raw)))
        _sentiment_cache["score"] = score
        _sentiment_cache["ts"] = now
        return score
    except Exception as exc:
        log.warning("LLM sentiment failed: %s", exc)
        return _sentiment_cache["score"] if _sentiment_cache["score"] is not None else 0.0


def build_sentiment_features() -> dict[str, float]:
    """
    Build all sentiment features. Designed to be fast — both sources are cached.
    """
    headlines = get_recent_headlines()
    fear_greed = get_fear_greed()
    llm_score = get_llm_sentiment(headlines)

    return {
        "fear_greed":     fear_greed if fear_greed is not None else 0.5,
        "llm_sentiment":  llm_score,
        "news_count":     float(len(headlines)),
    }
