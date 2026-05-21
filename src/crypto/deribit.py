"""
crypto/deribit.py — Deribit BTC options Put/Call Ratio (PCR) signal.

Free public API, no authentication required.
Endpoint: https://www.deribit.com/api/v2/public/get_book_summary_by_currency

PCR = sum(put_open_interest) / sum(call_open_interest)
Filtered to short-dated options (0–2 DTE) — most sensitive to near-term moves.

Contrarian interpretation (professional options traders are usually wrong on timing):
  PCR > 1.2 → crowd buying puts (fear) → contrarian BULLISH  → score > 0
  PCR < 0.7 → crowd buying calls (greed) → contrarian BEARISH → score < 0
  0.7–1.2   → balanced → score = 0.0

Score normalised to [-1, +1]:
  pcr = 1.5 → score = +1.0
  pcr = 0.4 → score = -1.0

Cached 5 minutes — options OI doesn't change tick-by-tick.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

_log = logging.getLogger("crypto.deribit")

_DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
_HTTP        = httpx.Client(timeout=8.0)

_CACHE_TTL   = 300.0   # 5 minutes
_MAX_DTE     = 2       # short-dated only
_MIN_OI      = 100.0   # ignore tiny strikes (BTC notional)
_BULLISH_PCR = 1.2     # above this → contrarian bullish
_BEARISH_PCR = 0.7     # below this → contrarian bearish
_PCR_SCALE   = 0.3     # 0.3 PCR units → full ±1.0 signal

_cache_score: Optional[float] = None
_cache_ts:    float            = 0.0


def _parse_dte(name: str) -> Optional[int]:
    """Days-to-expiry from instrument name: BTC-20FEB26-95000-P."""
    try:
        parts = name.split("-")
        if len(parts) < 3:
            return None
        expiry_dt = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
        return max(0, (expiry_dt - datetime.now(timezone.utc)).days)
    except Exception:
        return None


def _fetch_pcr() -> Optional[float]:
    try:
        resp = _HTTP.get(_DERIBIT_URL, params={"currency": "BTC", "kind": "option"})
        resp.raise_for_status()
        summaries = resp.json().get("result", [])
    except Exception as exc:
        _log.debug("Deribit fetch failed: %s", exc)
        return None

    put_oi = call_oi = 0.0
    for item in summaries:
        name = item.get("instrument_name", "")
        dte  = _parse_dte(name)
        if dte is None or dte > _MAX_DTE:
            continue
        oi = float(item.get("open_interest") or 0)
        if oi < _MIN_OI:
            continue
        if name.endswith("-P"):
            put_oi  += oi
        elif name.endswith("-C"):
            call_oi += oi

    if call_oi < 1.0:
        return None
    pcr = put_oi / call_oi
    _log.debug("Deribit PCR (0-%dDTE): %.3f  put_oi=%.0f  call_oi=%.0f",
               _MAX_DTE, pcr, put_oi, call_oi)
    return pcr


def get_pcr_score() -> Optional[float]:
    """
    Contrarian PCR score in [-1, +1], cached 5 minutes.
    Positive = crowd is fearful (buying puts) → bet UP.
    Negative = crowd is greedy (buying calls) → bet DOWN.
    Returns None on fetch failure (weight redistributed to drift in flow.py).
    """
    global _cache_score, _cache_ts
    now = time.monotonic()
    if _cache_score is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cache_score

    pcr = _fetch_pcr()
    if pcr is None:
        return None

    if pcr > _BULLISH_PCR:
        score = min(1.0,  (pcr - _BULLISH_PCR) / _PCR_SCALE)
    elif pcr < _BEARISH_PCR:
        score = max(-1.0, -((_BEARISH_PCR - pcr) / _PCR_SCALE))
    else:
        score = 0.0

    _cache_score = score
    _cache_ts    = now
    _log.info("Deribit PCR=%.3f → pcr_score=%.3f", pcr, score)
    return score
