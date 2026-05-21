"""
Bootstrap model training from settled Polymarket markets.

Enhanced version — v2 changes over v1:
  - Gamma API as primary data source (richer data, outcomePrices field for
    reliable outcome detection, better historical market coverage)
  - Default lookback 180 days / 500 markets (was 60 days / 200)
  - llm_intuition_score = 0.5 for ALL real historical records
    (Claude's training data extends to Aug 2025; it knows past outcomes, so
    using Claude to score historical text would contaminate training labels)
  - Point-in-time news for markets closed within the GNews free-plan window
    (~28 days); keyword-based sentiment analysis (no LLM call)
  - Prediction snapshot at as_of_date = close_time - as_of_offset_days (default 3)
    instead of close_time - 1 hour, giving more realistic pre-decision features

Two-stage feature construction:
  Stage 1 (preferred): Real CLOB price history + optional historical news.
  Stage 2 (fallback):  Calibrated synthetic features when price history is
                       unavailable (common for older settled markets).

Usage:
    python scripts/bootstrap_training.py
    python scripts/bootstrap_training.py --days 180 --limit 500
    python scripts/bootstrap_training.py --as-of-offset 7   # 7 days before close
    python scripts/bootstrap_training.py --no-news           # skip GNews (faster)
    python scripts/bootstrap_training.py --dry-run           # build cache, skip training
    python scripts/bootstrap_training.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
_ROOT    = _SCRIPTS.parent
_SRC     = _ROOT / "src"
sys.path.insert(0, str(_SRC))

# load .env before importing project modules (CLOB_HOST, GNEWS_API_KEY, etc.)
from dotenv import load_dotenv
for _p in [_ROOT, _ROOT / "polymarket-mcp-main" / "polymarket-mcp-main"]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

# ── project imports ───────────────────────────────────────────────────────────
import httpx

from data.market import CLOB_HOST, get_market_history, get_trades
from data.external import get_news
from features.schemas import FeatureVector, StructuredFeatures, TextFeatures
from prediction.training import RollingTrainer, save_feature_to_cache

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("bootstrap")

# ── constants ─────────────────────────────────────────────────────────────────
_RATE_LIMIT_SEC      = 0.3    # polite delay between per-market API calls
_GAMMA_HOST          = "https://gamma-api.polymarket.com"
_GNEWS_LOOKBACK_DAYS = 28     # GNews free plan history window (~1 month)

# Keyword lists for text sentiment analysis (no LLM — avoids contamination).
# These are coarse signals; the model learns to weight them via regression.
_POS_WORDS = frozenset({
    "yes", "win", "wins", "won", "pass", "passes", "passed", "approve", "approved",
    "increase", "increases", "rise", "rises", "rose", "bullish", "likely", "confirmed",
    "lead", "leads", "ahead", "victory", "victories", "succeed", "succeeds", "success",
    "gain", "gains", "higher", "surge", "surged",
})
_NEG_WORDS = frozenset({
    "no", "fail", "fails", "failed", "failure", "reject", "rejected", "rejects",
    "decrease", "decreases", "fall", "falls", "fell", "bearish", "unlikely", "denied",
    "behind", "loss", "losses", "defeat", "defeated", "collapse", "withdraw", "ban",
    "lower", "drop", "drops", "dropped", "miss", "misses", "missed",
})
_UNCERTAINTY_PHRASES = (
    "maybe", "possibly", "might", "could", "unclear", "uncertain", "close race",
    "narrow", "toss-up", "tossup", "undecided", "polling", "neck and neck",
    "too close to call", "could go either way",
)
_RISK_WORDS = frozenset({
    "fail", "crisis", "reject", "ban", "crash", "collapse", "scandal",
    "controversy", "arrest", "indicted", "charged", "killed", "attacked", "fraud",
    "default", "bankrupt", "bankruptcy",
})

# Category keyword lists (mirrors market.py to avoid cross-import)
_CRYPTO_KW   = frozenset({"bitcoin", "btc", "ethereum", "eth", "crypto", "defi",
                           "solana", "sol", "bnb", "xrp", "doge", "token", "nft",
                           "blockchain", "coinbase", "binance", "web3"})
_SPORTS_KW   = frozenset({"nba", "nfl", "nhl", "mlb", "nascar", "fifa",
                           "premier league", "champions league", "world cup",
                           "super bowl", "stanley cup", "masters", "wimbledon",
                           "ufc", "mma", "tennis", "golf", "basketball", "soccer"})
_POLITICS_KW = frozenset({"president", "election", "senate", "congress", "parliament",
                           "prime minister", "governor", "minister", "vote", "ballot",
                           "democrat", "republican", "trump", "biden", "fed ",
                           "federal reserve", "interest rate", "nato", "united nations"})


# ── Gamma API helpers ──────────────────────────────────────────────────────────

def _fetch_settled_markets_gamma(days_back: int, limit: int) -> list[dict]:
    """
    Fetch settled markets from the Gamma API.

    Gamma provides `outcomePrices` (e.g. '["1","0"]' = YES won) which is more
    reliable than the CLOB tokens[].winner field for historical markets.

    NOTE: Gamma's endDateIso is the *historical event date*, not the date the
    market was actively traded on Polymarket.  We therefore do NOT filter by
    date here — instead we rely on _determine_outcome_gamma() to verify that
    the market was actually resolved (outcomePrices near 0 or 1).

    Paginates via offset up to _MAX_GAMMA_PAGES pages.
    """
    _MAX_GAMMA_PAGES = 10   # hard cap to avoid very long runs
    results: list[dict] = []
    offset  = 0
    page    = 0

    with httpx.Client(timeout=30.0) as http:
        while len(results) < limit and page < _MAX_GAMMA_PAGES:
            page += 1
            params: dict = {
                "closed": "true",
                "limit":  500,
                "offset": offset,
            }
            try:
                r = http.get(f"{_GAMMA_HOST}/markets", params=params)
                r.raise_for_status()
                payload = r.json()
            except Exception as exc:
                log.warning("Gamma API error (page %d, offset %d): %s", page, offset, exc)
                break

            batch = payload if isinstance(payload, list) else payload.get("data", [])
            if not batch:
                break

            accepted = 0
            for raw in batch:
                if not (raw.get("closed", False) or raw.get("resolved", False)):
                    continue
                # Quick pre-filter: outcomePrices must show a clear winner
                if _determine_outcome_gamma(raw) == -1:
                    continue
                results.append(raw)
                accepted += 1
                if len(results) >= limit:
                    break

            log.debug(
                "Gamma page %d (offset=%d): batch=%d accepted=%d total=%d",
                page, offset, len(batch), accepted, len(results),
            )

            if len(results) >= limit or len(batch) < 500:
                break

            offset += len(batch)

    log.info("Gamma API: fetched %d settled markets", len(results))
    return results[:limit]


def _fetch_settled_markets_clob(days_back: int, limit: int) -> list[dict]:
    """
    Fetch settled markets from the CLOB API (fallback when Gamma is unavailable).

    NOTE: CLOB returns oldest markets first.  With days_back=180 the target
    window is only 6 months old, but the API may return thousands of older
    markets before reaching it.  We cap at _MAX_CLOB_PAGES to prevent
    multi-hour runtimes.
    """
    _MAX_CLOB_PAGES = 5
    cutoff      = datetime.now(timezone.utc) - timedelta(days=days_back)
    results: list[dict] = []
    next_cursor: Optional[str] = None
    page = 0

    with httpx.Client(timeout=30.0) as http:
        while len(results) < limit and page < _MAX_CLOB_PAGES:
            page += 1
            params: dict = {"closed": "true", "limit": 1000}
            if next_cursor:
                params["next_cursor"] = next_cursor

            try:
                r = http.get(f"{CLOB_HOST}/markets", params=params)
                r.raise_for_status()
                payload = r.json()
            except Exception as exc:
                log.warning("CLOB fetch error (page %d): %s", page, exc)
                break

            batch       = payload.get("data", []) if isinstance(payload, dict) else payload
            next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None

            if not batch:
                break

            accepted = 0
            for raw in batch:
                if not raw.get("closed", False):
                    continue
                end_str = raw.get("end_date_iso") or raw.get("resolution_date", "")
                try:
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue
                if end_dt < cutoff:
                    continue
                results.append(raw)
                accepted += 1
                if len(results) >= limit:
                    break

            log.debug("CLOB page %d: batch=%d accepted=%d total=%d",
                      page, len(batch), accepted, len(results))

            if not next_cursor or len(results) >= limit:
                break

    log.info("CLOB API: fetched %d settled markets (days_back=%d)", len(results), days_back)
    return results[:limit]


def _fetch_settled_markets(days_back: int, limit: int) -> tuple[list[dict], str]:
    """Try Gamma API first, then CLOB. Returns (markets, source)."""
    markets = _fetch_settled_markets_gamma(days_back, limit)
    if markets:
        return markets, "gamma"
    log.warning("Gamma returned no results — falling back to CLOB API.")
    markets = _fetch_settled_markets_clob(days_back, limit)
    return markets, "clob"


# ── Outcome detection ──────────────────────────────────────────────────────────

def _determine_outcome_gamma(raw: dict) -> int:
    """
    Determine YES=1 / NO=0 / unknown=-1 from a Gamma API market dict.

    Priority order:
      1. outcomePrices field: ["1","0"] = YES won, ["0","1"] = NO won.
         NOTE: Gamma returns this as a JSON-encoded *string*, not a Python list.
         e.g. '["0.9999998...", "0.0000001..."]' — must json.loads() first.
      2. tokens[].winner with explicit YES/NO label.
      3. Binary market with unambiguous winner.
    """
    # 1. outcomePrices (most reliable for resolved markets)
    outcome_prices = raw.get("outcomePrices")
    # Gamma returns this as a JSON string — parse it if needed
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            outcome_prices = None
    if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
        try:
            yes_p = float(outcome_prices[0])
            no_p  = float(outcome_prices[1])
            if yes_p >= 0.9:
                return 1
            if no_p >= 0.9:
                return 0
        except (TypeError, ValueError):
            pass

    # 2. Explicit YES/NO winner label in token list
    tokens = raw.get("tokens") or []
    if isinstance(tokens, list):
        for tok in tokens:
            if not isinstance(tok, dict):
                continue
            if tok.get("winner", False):
                s = str(tok.get("outcome", "")).lower().strip()
                if s in {"yes", "true", "1"}:
                    return 1
                if s in {"no", "false", "0"}:
                    return 0

    # 3. Binary market with exactly one winner
    valid = [t for t in tokens if isinstance(t, dict)] if isinstance(tokens, list) else []
    if len(valid) == 2:
        winners = [t for t in valid if t.get("winner", False)]
        losers  = [t for t in valid if not t.get("winner", False)]
        if len(winners) == 1 and len(losers) == 1:
            return 1 if valid[0].get("winner", False) else 0

    return -1


def _determine_outcome_clob(raw: dict) -> int:
    """Determine outcome from CLOB market dict."""
    tokens = [t for t in raw.get("tokens", []) if isinstance(t, dict)]
    for tok in tokens:
        if tok.get("winner", False):
            s = str(tok.get("outcome", "")).lower().strip()
            if s in {"yes", "true", "1"}:
                return 1
            if s in {"no", "false", "0"}:
                return 0
    if len(tokens) == 2:
        winners = [t for t in tokens if t.get("winner", False)]
        losers  = [t for t in tokens if not t.get("winner", False)]
        if len(winners) == 1 and len(losers) == 1:
            return 1 if tokens[0].get("winner", False) else 0
    return -1


def _get_market_meta(raw: dict, source: str) -> tuple[str, str, str, str]:
    """
    Extract (condition_id, title, category, yes_token_id) from a raw dict.
    Handles both Gamma and CLOB schemas.
    Returns ("", "", "other", "") on failure.
    """
    if source == "gamma":
        condition_id = raw.get("conditionId") or raw.get("condition_id", "")
        title        = raw.get("question") or raw.get("title", "")
        yes_token_id = ""
        token_ids    = raw.get("clobTokenIds") or []
        # Gamma returns clobTokenIds as a JSON-encoded string, not a Python list
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                token_ids = []
        if isinstance(token_ids, list) and token_ids:
            yes_token_id = str(token_ids[0])

        # Category: events tag → keyword fallback
        category = "other"
        events = raw.get("events") or []
        if isinstance(events, list) and events:
            tag = str(events[0].get("tag", "") or events[0].get("category", "")).lower()
            if tag in {"politics", "crypto", "sports"}:
                category = tag
        if category == "other":
            tl = title.lower()
            if any(kw in tl for kw in _CRYPTO_KW):
                category = "crypto"
            elif any(kw in tl for kw in _SPORTS_KW):
                category = "sports"
            elif any(kw in tl for kw in _POLITICS_KW):
                category = "politics"
    else:
        condition_id = raw.get("condition_id") or raw.get("market_id", "")
        title        = raw.get("question") or raw.get("title", "")
        raw_cat      = str(raw.get("category", "other")).lower().strip()
        category     = raw_cat if raw_cat in {"politics", "crypto", "sports"} else "other"
        yes_token_id = ""
        for tok in raw.get("tokens", []):
            if isinstance(tok, dict):
                s = str(tok.get("outcome", "")).lower()
                if s in {"yes", "true", "1"}:
                    yes_token_id = str(tok.get("token_id", ""))
                    break
        if not yes_token_id:
            tokens = raw.get("tokens", [])
            if tokens and isinstance(tokens[0], dict):
                yes_token_id = str(tokens[0].get("token_id", ""))

    return condition_id, title, category, yes_token_id


# ── Keyword-based text analysis (no LLM) ──────────────────────────────────────

def _keyword_sentiment(text: str) -> float:
    """Simple bag-of-words sentiment: -1.0 (negative) to +1.0 (positive)."""
    words = set(re.findall(r"\w+", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    return 0.0 if total == 0 else max(-1.0, min(1.0, (pos - neg) / total))


def _keyword_uncertainty(text: str) -> float:
    """0 = certain, 1 = very uncertain (capped at 1.0)."""
    tl = text.lower()
    score = sum(1 for phrase in _UNCERTAINTY_PHRASES if phrase in tl)
    return min(1.0, score * 0.25)   # 4+ phrases saturates at 1.0


def _keyword_risk_count(text: str) -> int:
    """Count of negative-risk keywords present in the text."""
    words = set(re.findall(r"\w+", text.lower()))
    return len(words & _RISK_WORDS)


def _build_text_features_from_news(
    market_id: str,
    timestamp: datetime,
    articles: list,   # list[Article]
) -> TextFeatures:
    """
    Build TextFeatures using keyword analysis on article titles/summaries.

    IMPORTANT: llm_intuition_score is ALWAYS 0.5 (neutral) for historical
    bootstrap data.  Claude's training data extends to Aug 2025, so it knows
    the outcomes of past markets — letting it score historical text would leak
    the label into the training features.

    The other text scores (sentiment, uncertainty, risk) are computed purely
    from keyword co-occurrence, which is safe for historical data.
    """
    if not articles:
        return TextFeatures(
            market_id=market_id,
            timestamp=timestamp,
            sentiment_score=0.0,
            uncertainty_score=0.5,
            negative_risk_count=0,
            llm_intuition_score=0.5,   # NEUTRALIZED
            summary="bootstrap: no historical news available",
            source_count=0,
        )

    texts = [(a.title or "") + " " + (a.summary or "") for a in articles]

    sentiments    = [_keyword_sentiment(t)   for t in texts]
    uncertainties = [_keyword_uncertainty(t) for t in texts]
    risk_counts   = [_keyword_risk_count(t)  for t in texts]

    avg_sentiment   = sum(sentiments)    / len(sentiments)
    avg_uncertainty = sum(uncertainties) / len(uncertainties)
    total_risk      = sum(risk_counts)

    return TextFeatures(
        market_id=market_id,
        timestamp=timestamp,
        sentiment_score=max(-1.0, min(1.0, avg_sentiment)),
        uncertainty_score=max(0.0, min(1.0, avg_uncertainty)),
        negative_risk_count=total_risk,
        llm_intuition_score=0.5,   # NEUTRALIZED: Claude knows past outcomes
        summary=f"bootstrap: {len(articles)} historical articles (keyword analysis)",
        source_count=len(articles),
    )


# ── Feature vector builders ────────────────────────────────────────────────────

def _safe_return(cur: float, past: float) -> float:
    return (cur - past) / past if past > 1e-9 else 0.0


def _stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    try:
        return statistics.stdev(vals)
    except statistics.StatisticsError:
        return 0.0


def _build_real_feature_vector(
    condition_id: str,
    title: str,
    category: str,
    yes_token_id: str,
    close_time: datetime,
    as_of_offset_days: int,
    fetch_news: bool,
) -> Optional[FeatureVector]:
    """
    Build a FeatureVector from CLOB price history + optional point-in-time news.

    Snapshot time = close_time - as_of_offset_days.  Using 3 days before close
    (instead of 1 hour) provides:
      - Richer price history (more OHLCV candles available)
      - More realistic trading conditions (avoids last-minute volatility)
      - Consistent seconds_to_close feature (~259,200 s across all samples)

    Returns None if no price data is available for this market.
    """
    prediction_time = close_time - timedelta(days=as_of_offset_days)
    start_24h       = prediction_time - timedelta(hours=24)
    start_1h        = prediction_time - timedelta(hours=1)

    closes_24h: list[float] = []
    closes_1h:  list[float] = []

    try:
        hist_24h   = get_market_history(
            condition_id, start_24h, prediction_time,
            interval="1h", token_id=yes_token_id,
        )
        closes_24h = [p.close for p in hist_24h]
    except Exception:
        pass

    try:
        hist_1h  = get_market_history(
            condition_id, start_1h, prediction_time,
            interval="1m", token_id=yes_token_id,
        )
        closes_1h = [p.close for p in hist_1h]
    except Exception:
        pass

    if not closes_24h and not closes_1h:
        return None   # no data; caller uses synthetic fallback

    price_current = closes_24h[-1] if closes_24h else closes_1h[-1]
    price_current = max(0.001, min(0.999, price_current))

    volume_24h = 0.0
    try:
        trades     = get_trades(condition_id, start_24h, prediction_time)
        volume_24h = sum(t.size for t in trades)
    except Exception:
        pass

    struct = StructuredFeatures(
        market_id           = condition_id,
        timestamp           = prediction_time,
        category            = category,
        seconds_to_close    = max(0.0, (close_time - prediction_time).total_seconds()),
        price_current       = price_current,
        price_1h_return     = (
            _safe_return(price_current, closes_24h[0]) if len(closes_24h) >= 2 else 0.0
        ),
        price_24h_return    = (
            _safe_return(price_current, closes_1h[0]) if len(closes_1h) >= 2 else 0.0
        ),
        volatility_24h      = _stdev(closes_24h),
        volatility_1h       = _stdev(closes_1h),
        volume_24h          = volume_24h,
        spread              = 0.02,          # orderbook not available historically
        orderbook_imbalance = 0.5,
    )

    # ── Point-in-time news (keyword-based; no LLM) ────────────────────────────
    articles = []
    if fetch_news and title:
        now = datetime.now(timezone.utc)
        # GNews free plan only retains ~28 days of history
        if (now - close_time).days <= _GNEWS_LOOKBACK_DAYS:
            news_since = prediction_time - timedelta(days=7)
            try:
                articles = get_news(
                    query=title[:80],
                    since=news_since,
                    limit=10,
                    until=prediction_time,
                )
                if articles:
                    log.debug(
                        "  news: %d articles for %s", len(articles), condition_id[:20]
                    )
            except Exception as exc:
                log.debug("  news fetch error for %s: %s", condition_id[:16], exc)

    text = _build_text_features_from_news(condition_id, prediction_time, articles)
    return FeatureVector(structured=struct, text=text)


def _build_synthetic_feature_vector(
    market_id: str,
    category: str,
    outcome: int,
    rng: random.Random,
) -> FeatureVector:
    """
    Generate a calibrated synthetic FeatureVector for a given outcome.

    Unlike real historical records, synthetic data is not contaminated by
    Claude's knowledge of past events, so llm_intuition_score can safely
    be correlated with price_current.  This teaches the model the expected
    predictive relationship for this feature in live trading.

    price_current is sampled from a Beta distribution biased toward the
    true outcome (YES→mean≈0.70, NO→mean≈0.30) with noise to prevent
    memorisation.
    """
    now = datetime.now(timezone.utc)

    if outcome == 1:
        raw_price = rng.betavariate(7, 3)   # mean ≈ 0.70
    else:
        raw_price = rng.betavariate(3, 7)   # mean ≈ 0.30
    price_current = max(0.02, min(0.98, raw_price))

    seconds_to_close   = rng.uniform(3600, 7 * 86400)
    price_24h_return   = rng.gauss(0, 0.04)
    price_1h_return    = rng.gauss(0, 0.015)
    volatility_24h     = abs(rng.gauss(0.025, 0.015))
    volatility_1h      = abs(rng.gauss(0.008, 0.005))
    volume_24h         = abs(rng.gauss(6000, 4000))
    spread             = max(0.005, abs(rng.gauss(0.02, 0.01)))
    imbalance          = rng.uniform(0.3, 0.7)
    sentiment          = rng.gauss(0, 0.25)
    uncertainty        = rng.uniform(0.2, 0.8)
    risk_count         = rng.randint(0, 3)
    # llm_intuition correlated with outcome for synthetic data (no contamination)
    intuition          = max(0.01, min(0.99, price_current + rng.gauss(0, 0.08)))

    struct = StructuredFeatures(
        market_id           = market_id,
        timestamp           = now,
        category            = category,
        seconds_to_close    = seconds_to_close,
        price_current       = price_current,
        price_1h_return     = price_1h_return,
        price_24h_return    = price_24h_return,
        volatility_1h       = volatility_1h,
        volatility_24h      = volatility_24h,
        volume_24h          = volume_24h,
        spread              = spread,
        orderbook_imbalance = imbalance,
    )
    text = TextFeatures(
        market_id           = market_id,
        timestamp           = now,
        sentiment_score     = sentiment,
        uncertainty_score   = uncertainty,
        negative_risk_count = risk_count,
        llm_intuition_score = intuition,
        summary             = "bootstrap: synthetic sample",
    )
    return FeatureVector(structured=struct, text=text)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap model training from settled Polymarket markets (v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Fetch markets settled within the last N days.",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max number of settled markets to fetch from the API.",
    )
    parser.add_argument(
        "--as-of-offset", type=int, default=3, dest="as_of_offset",
        help=(
            "Days before market close to take the historical price snapshot. "
            "3 (default) = snapshot at T-3d, giving realistic pre-decision features. "
            "Increase to 7 if the CLOB rarely has price data at T-3d."
        ),
    )
    parser.add_argument(
        "--synthetic", type=int, default=200,
        help=(
            "Additional synthetic samples for markets with no price history. "
            "Set 0 to disable. Reduced default (200) vs v1 (300) since Gamma API "
            "provides more real historical data."
        ),
    )
    parser.add_argument(
        "--no-news", action="store_true", dest="no_news",
        help=(
            "Skip GNews fetching entirely. Text features default to neutral values. "
            "Use this if GNEWS_API_KEY is unset or to speed up the run."
        ),
    )
    parser.add_argument(
        "--window-days", type=int, default=90,
        help="Training window (days) passed to RollingTrainer.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the feature cache only; do not train models.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fetch_news = not args.no_news

    log.info(
        "Bootstrap v2 | days=%d limit=%d as_of_offset=%d "
        "synthetic=%d fetch_news=%s window=%d dry_run=%s",
        args.days, args.limit, args.as_of_offset, args.synthetic,
        fetch_news, args.window_days, args.dry_run,
    )

    rng = random.Random(42)   # deterministic for reproducibility

    # ── 1. fetch settled markets ───────────────────────────────────────────────
    raw_markets, source = _fetch_settled_markets(args.days, args.limit)
    if not raw_markets:
        log.warning("No settled markets found from API. Will rely on synthetic data only.")
    else:
        log.info("Using %d markets from %s API", len(raw_markets), source.upper())

    # ── 2. process each settled market ────────────────────────────────────────
    saved_real  = 0
    saved_synth = 0
    skipped     = 0
    news_hits   = 0
    synthetic_queue: list[tuple[str, str, int]] = []   # (condition_id, category, outcome)

    for idx, raw in enumerate(raw_markets, 1):
        # Determine outcome
        outcome = (
            _determine_outcome_gamma(raw)
            if source == "gamma"
            else _determine_outcome_clob(raw)
        )
        if outcome == -1:
            log.debug("[%d/%d] skip: no winner data", idx, len(raw_markets))
            skipped += 1
            continue

        # Parse close time
        end_str = (raw.get("endDateIso") or raw.get("endDate")
                   or raw.get("end_date_iso") or raw.get("resolution_date", ""))
        try:
            close_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            skipped += 1
            continue

        # Extract market metadata
        condition_id, title, category, yes_token_id = _get_market_meta(raw, source)
        if not condition_id:
            skipped += 1
            continue

        mid_short = condition_id[:20]

        # Stage 1: try real price history
        fv = _build_real_feature_vector(
            condition_id, title, category, yes_token_id,
            close_time, args.as_of_offset, fetch_news,
        )

        if fv is not None:
            if fv.text.source_count > 0:
                news_hits += 1
            try:
                save_feature_to_cache(fv, outcome)
                saved_real += 1
                log.info(
                    "[%d/%d] real   %-20s | %s | price=%.3f vol=%6.0f cat=%-8s news=%d",
                    idx, len(raw_markets), mid_short,
                    "YES" if outcome == 1 else "NO ",
                    fv.structured.price_current,
                    fv.structured.volume_24h,
                    category,
                    fv.text.source_count,
                )
            except Exception as exc:
                log.warning("[%d/%d] save error %s: %s", idx, len(raw_markets), mid_short, exc)
            time.sleep(_RATE_LIMIT_SEC)
            continue

        # Stage 2: queue for synthetic fallback
        synthetic_queue.append((condition_id, category, outcome))
        time.sleep(_RATE_LIMIT_SEC)

    log.info(
        "Real data pass: real=%d  news_enriched=%d  synthetic_needed=%d  skipped=%d",
        saved_real, news_hits, len(synthetic_queue), skipped,
    )

    # ── 3. synthetic fallback ─────────────────────────────────────────────────
    if args.synthetic > 0:
        n_from_queue = len(synthetic_queue)
        n_extra      = max(0, args.synthetic - n_from_queue)
        categories   = ["politics", "crypto", "sports", "other"]

        log.info(
            "Generating synthetic: %d from settled-market queue + %d random top-up",
            n_from_queue, n_extra,
        )

        # From settled-market queue (preserves real category/outcome distribution)
        for i, (mid, cat, outcome) in enumerate(synthetic_queue):
            synth_id = f"synth_{mid[:16]}_{i}"
            fv = _build_synthetic_feature_vector(synth_id, cat, outcome, rng)
            try:
                save_feature_to_cache(fv, outcome)
                saved_synth += 1
            except Exception as exc:
                log.warning("Synth save error (%s): %s", synth_id[:12], exc)

        # Random top-up to reach args.synthetic total
        for i in range(n_extra):
            synth_id = f"synth_rnd_{i}"
            cat      = rng.choice(categories)
            outcome  = rng.choice([0, 1])
            fv = _build_synthetic_feature_vector(synth_id, cat, outcome, rng)
            try:
                save_feature_to_cache(fv, outcome)
                saved_synth += 1
            except Exception as exc:
                log.warning("Synth save error (%s): %s", synth_id[:12], exc)

        log.info("Synthetic samples saved: %d", saved_synth)

    total_saved = saved_real + saved_synth
    log.info(
        "Cache total: %d real + %d synthetic = %d samples",
        saved_real, saved_synth, total_saved,
    )

    if args.dry_run:
        log.info("--dry-run: skipping model training. Done.")
        return

    if total_saved == 0:
        log.error("No feature vectors saved. Cannot train.")
        sys.exit(1)

    # ── 4. train models ────────────────────────────────────────────────────────
    log.info("Starting model training (window=%d days)...", args.window_days)
    trainer = RollingTrainer(window_days=args.window_days)
    result  = trainer.run_training_cycle()
    status  = result.get("status", "unknown")

    log.info("Training result: %s", result.get("message", result))

    if status == "success":
        log.info(
            "Models trained: baseline=%s  tree=%s  n_samples=%d",
            "OK" if result["baseline_ok"] else "FAIL",
            "OK" if result["tree_ok"]     else "FAIL",
            result["n_samples"],
        )
        log.info("predict_probability() will now return model-based probabilities.")
    elif status == "skipped":
        log.warning(
            "Training skipped (%s). "
            "Minimum sample requirement is 30. Try increasing --synthetic.",
            result.get("message", ""),
        )
    else:
        log.error("Training failed: %s", result.get("message", "unknown error"))
        sys.exit(1)


if __name__ == "__main__":
    main()
