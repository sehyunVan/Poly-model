"""
crypto/flow.py — order-flow signal calculator for Polymarket 5-minute markets.

Strategy: follow crowd momentum rather than predict BTC price direction.

Signals
-------
1. price_drift   — how far the UP token price has moved from 0.50 since the
                   window opened.  Positive = crowd buying UP, negative = DOWN.
2. ob_imbalance  — bid vs ask depth at levels near the current price (±0.15).
                   > 0.5 = more buy pressure on UP side.

Combined score in [-1, +1]:
    score = 0.65 * drift_score + 0.35 * ob_score

Bet when abs(score) > SIGNAL_THRESHOLD and we are 60–180 s into the window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional  # noqa: F401 — used in compute_signal signature

import httpx

_log = logging.getLogger("crypto.flow")
_HTTP = httpx.Client(timeout=6.0)

SIGNAL_THRESHOLD = 0.02   # minimum |score| to generate a trade (emergency low — missing signals reducing score)
DRIFT_SCALE      = 0.06   # price move of ±6 cents maps to ±1.0 drift score
OB_NEAR_BAND     = 0.15   # only look at orderbook levels within ±0.15 of mid
# Contrarian ob threshold: when ob_imb < this, the DOWN side is saturated with
# buyers (nobody is bidding UP).  Empirically: ob≈0 → market goes UP 76.3% of
# the time (n=834) despite the formula's bearish reading.  Flip ob_score to +1.0.
OB_CONTRARIAN_THRESHOLD = 0.05

# Signal weights — must sum to 1.0
# 2026-05-09: added W_MOM_30S — last-30s Binance return is the strongest single
# directional WR predictor in signal_log (40+pp WR swing across mom_30s buckets).
# Other weights scaled by 0.80 to make room (0.30→0.24, 0.10→0.08).
# MOM_30S_SCALE=0.0005 (0.05% move) maps to ±1 score.
# When any signal is unavailable, its weight is redistributed to drift.
W_DRIFT        = 0.24
W_OB           = 0.08
W_CVD          = 0.0
W_MACD         = 0.0
W_LIQ          = 0.0
W_FUNDING      = 0.0
W_CLOB_CVD     = 0.24
W_EXCHANGE_DIV = 0.0
W_PCR          = 0.0
W_TICK_VEL     = 0.0
W_TRADE_IMB    = 0.0
W_ORACLE_LAG   = 0.24
W_HAWKES       = 0.0
W_MLOFI        = 0.0
W_MOM_30S      = 0.20
MOM_30S_SCALE  = 0.0005   # 0.05% move maps to ±1 score


@dataclass
class FlowSignal:
    direction:      str    # "UP", "DOWN", "NO_TRADE"
    score:          float  # [-1, +1] — magnitude = confidence
    price_drift:    float  # current_price - window_open_price
    ob_imbalance:   float  # [0, 1] near-price bid fraction
    cvd_score:      float  # [-1, +1] normalised Binance CVD (0 if unavailable)
    macd_score:     float  # [-1, +1] normalised MACD histogram (0 if unavailable)
    current_price:  float  # current UP token consensus price
    alpha:          float  # abs(score) - threshold (> 0 means trade)
    liq_score:              float = 0.0  # [-1,+1] HL liquidation imbalance (0 if unavailable)
    funding_score:          float = 0.0  # [-1,+1] funding rate fade (0 if unavailable)
    clob_cvd_score:         float = 0.0  # [-1,+1] Polymarket CLOB taker CVD (0 if unavailable)
    exchange_div_score:     float = 0.0  # [-1,+1] Binance vs Coinbase divergence (0 if unavailable)
    pcr_score:              float = 0.0  # [-1,+1] Deribit PCR contrarian (0 if unavailable)
    tick_velocity_score:    float = 0.0  # [-1,+1] CLOB UP-token price velocity (0 if unavailable)
    trade_imbalance_score:  float = 0.0  # [-1,+1] Binance 60s buy/sell imbalance (0 if unavailable)
    oracle_lag_score:       float = 0.0  # [-1,+1] Binance vs Chainlink oracle lag (0 if unavailable)
    hawkes_score:           float = 0.0  # [-1,+1] Hawkes decayed buy/sell ratio (0 if unavailable)
    mlofi_score:            float = 0.0  # [-1,+1] Multi-Level OFI from Binance depth (0 if unavailable)
    mom_30s_score:          float = 0.0  # [-1,+1] last-30s Binance return / MOM_30S_SCALE (0 if unavailable)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _fetch_book(up_token_id: str) -> tuple[list, list]:
    """Return (bids, asks) lists of {price, size} dicts. Empty on error."""
    try:
        r = _HTTP.get(
            "https://clob.polymarket.com/book",
            params={"token_id": up_token_id},
        )
        r.raise_for_status()
        d = r.json()
        return d.get("bids", []), d.get("asks", [])
    except Exception as exc:
        _log.debug("Book fetch failed: %s", exc)
        return [], []


def _get_book(token_id: str, feed=None) -> tuple[list, list]:
    """
    Return (bids, asks) from WS cache when fresh, falling back to REST.
    feed: CLOBFeed instance or None. When None, always uses REST.
    After a REST fallback, seeds the WS cache so price_change events apply.
    """
    if feed is not None:
        bids, asks = feed.get_book(token_id)
        if bids or asks:
            return bids, asks
    bids, asks = _fetch_book(token_id)
    if feed is not None and (bids or asks):
        feed.seed_book(token_id, bids, asks)
    return bids, asks


def _cross_imbalance(up_token_id: str, down_token_id: str,
                     up_price: float, feed=None) -> float:
    """
    Compare buy pressure on the UP token vs the DOWN token.

    Fetch the orderbook for both tokens and sum bid volume near each token's
    current price. Returns up_buy / (up_buy + down_buy):
      > 0.5 → more money flowing into UP  (bullish)
      < 0.5 → more money flowing into DOWN (bearish)
      = 0.5 → no signal (not enough data)
    """
    up_bids,   _ = _get_book(up_token_id, feed)
    down_bids, _ = _get_book(down_token_id, feed)
    down_price    = 1.0 - up_price

    up_buy   = sum(
        float(b["size"]) for b in up_bids
        if abs(float(b["price"]) - up_price) <= OB_NEAR_BAND
    )
    down_buy = sum(
        float(b["size"]) for b in down_bids
        if abs(float(b["price"]) - down_price) <= OB_NEAR_BAND
    )
    total = up_buy + down_buy
    if total < 5:
        return 0.5
    return up_buy / total


def compute_signal(
    up_token_id: str,
    down_token_id: str,
    current_price: float,
    open_price: float,
    cvd_score: Optional[float] = None,
    macd_score: Optional[float] = None,
    liq_score: Optional[float] = None,
    funding_score: Optional[float] = None,
    clob_cvd_score: Optional[float] = None,
    drift_scale: float = DRIFT_SCALE,
    clob_feed=None,
    binance_price: Optional[float] = None,
    coinbase_price: Optional[float] = None,
    pcr_score: Optional[float] = None,
    tick_velocity_score: Optional[float] = None,
    trade_imbalance_score: Optional[float] = None,
    oracle_lag_score: Optional[float] = None,
    hawkes_score: Optional[float] = None,
    mlofi_score: Optional[float] = None,
    mom_30s_raw: Optional[float] = None,
) -> FlowSignal:
    """
    Build a FlowSignal for one BTC/ETH up/down market.

    Parameters
    ----------
    up_token_id   : CLOB token ID for the UP outcome
    down_token_id : CLOB token ID for the DOWN outcome
    current_price : current consensus UP price (from Gamma outcomePrices)
    open_price    : UP price recorded when we first saw this market window
    cvd_score     : normalised CVD from BinanceLiveFeed [-1, +1], or None
    macd_score    : normalised MACD(3,15,3) histogram [-1, +1], or None
    liq_score     : OI-delta liquidation proxy [-1, +1], or None
    funding_score : funding rate contrarian fade [-1, +1], or None
    oracle_lag_score : Binance vs Chainlink oracle lag [-1, +1], or None
    hawkes_score  : Hawkes decayed buy/sell excitement ratio [-1, +1], or None
    mlofi_score   : Multi-Level OFI from Binance depth [-1, +1], or None
    """
    # ── Signal 1: price drift ─────────────────────────────────────────────────
    price_drift = current_price - open_price
    drift_score = _clamp(price_drift / drift_scale)

    # ── Signal 2: cross-token orderbook imbalance ─────────────────────────────
    ob_imb = _cross_imbalance(up_token_id, down_token_id, current_price, feed=clob_feed)
    if ob_imb < OB_CONTRARIAN_THRESHOLD:
        # Saturated DOWN buying: nobody bidding UP means DOWN side is crowded.
        # Data: ob≈0 → 76.3% UP rate across 834 trades (post-live: 73.4%, n=361).
        ob_score = 1.0
        _log.info("OB_CONTRARIAN: ob_imb=%.3f < %.2f → ob_score flipped to +1.0 (bullish)",
                  ob_imb, OB_CONTRARIAN_THRESHOLD)
    else:
        ob_score = _clamp((ob_imb - 0.5) * 4)   # ±2 mapped to ±1

    cvd_val          = cvd_score              if cvd_score              is not None else 0.0
    macd_val         = macd_score             if macd_score             is not None else 0.0
    liq_val          = liq_score              if liq_score              is not None else 0.0
    funding_val      = funding_score          if funding_score          is not None else 0.0
    clob_cvd_val     = clob_cvd_score         if clob_cvd_score         is not None else 0.0
    pcr_val          = pcr_score              if pcr_score              is not None else 0.0
    tick_vel_val     = tick_velocity_score    if tick_velocity_score    is not None else 0.0
    trade_imb_val    = trade_imbalance_score  if trade_imbalance_score  is not None else 0.0
    oracle_lag_val   = oracle_lag_score       if oracle_lag_score       is not None else 0.0
    hawkes_val       = hawkes_score           if hawkes_score           is not None else 0.0
    mlofi_val        = mlofi_score            if mlofi_score            is not None else 0.0
    mom_30s_val      = _clamp(mom_30s_raw / MOM_30S_SCALE) if mom_30s_raw is not None else 0.0

    # ── Signal: cross-exchange divergence (Binance vs Coinbase) ──────────────
    # Kept for data collection only — W_EXCHANGE_DIV=0.00 (replaced by oracle_lag).
    _has_div = binance_price is not None and coinbase_price is not None and coinbase_price > 0
    div_score = 0.0
    if _has_div:
        _div = (binance_price - coinbase_price) / coinbase_price
        div_score = _clamp(_div / 0.005)

    # Redistribute missing signal weights to drift
    w_drift        = W_DRIFT
    w_cvd          = W_CVD          if cvd_score              is not None else 0.0
    w_macd         = W_MACD         if macd_score             is not None else 0.0
    w_liq          = W_LIQ          if liq_score              is not None else 0.0
    w_funding      = W_FUNDING      if funding_score          is not None else 0.0
    w_clob_cvd     = W_CLOB_CVD     if clob_cvd_score         is not None else 0.0
    w_exchange_div = 0.0            # always 0 — deprecated
    w_pcr          = W_PCR          if pcr_score              is not None else 0.0
    w_tick_vel     = W_TICK_VEL     if tick_velocity_score    is not None else 0.0
    w_trade_imb    = W_TRADE_IMB    if trade_imbalance_score  is not None else 0.0
    w_oracle_lag   = W_ORACLE_LAG   if oracle_lag_score       is not None else 0.0
    w_hawkes       = W_HAWKES       if hawkes_score           is not None else 0.0
    w_mlofi        = W_MLOFI        if mlofi_score            is not None else 0.0
    w_mom_30s      = W_MOM_30S      if mom_30s_raw            is not None else 0.0
    w_ob            = W_OB
    w_drift       += ((W_CVD - w_cvd) + (W_MACD - w_macd) + (W_LIQ - w_liq)
                      + (W_FUNDING - w_funding) + (W_CLOB_CVD - w_clob_cvd)
                      + (W_PCR - w_pcr) + (W_TICK_VEL - w_tick_vel)
                      + (W_TRADE_IMB - w_trade_imb)
                      + (W_ORACLE_LAG - w_oracle_lag)
                      + (W_HAWKES - w_hawkes)
                      + (W_MLOFI - w_mlofi)
                      + (W_MOM_30S - w_mom_30s))

    # ── Combined score ────────────────────────────────────────────────────────
    score = (w_drift * drift_score + w_ob * ob_score
             + w_cvd * cvd_val + w_macd * macd_val
             + w_liq * liq_val + w_funding * funding_val
             + w_clob_cvd * clob_cvd_val
             + w_pcr * pcr_val
             + w_tick_vel * tick_vel_val
             + w_trade_imb * trade_imb_val
             + w_oracle_lag * oracle_lag_val
             + w_hawkes * hawkes_val
             + w_mlofi * mlofi_val
             + w_mom_30s * mom_30s_val)
    alpha = abs(score) - SIGNAL_THRESHOLD

    if alpha <= 0:
        direction = "NO_TRADE"
    elif score > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    _log.debug(
        "flow signal: drift=%.3f(%.2f) ob_imb=%.2f(%.2f) cvd=%.2f macd=%.2f "
        "liq=%.2f funding=%.2f clob_cvd=%.2f pcr=%.2f tick=%.2f imb=%.2f "
        "oracle_lag=%.2f hawkes=%.2f mlofi=%.2f mom30s=%.2f "
        "→ score=%.3f alpha=%.3f dir=%s",
        price_drift, drift_score, ob_imb, ob_score, cvd_val, macd_val,
        liq_val, funding_val, clob_cvd_val,
        pcr_val, tick_vel_val, trade_imb_val,
        oracle_lag_val, hawkes_val, mlofi_val, mom_30s_val,
        score, alpha, direction,
    )

    return FlowSignal(
        direction=direction,
        score=score,
        price_drift=price_drift,
        ob_imbalance=ob_imb,
        cvd_score=cvd_val,
        macd_score=macd_val,
        current_price=current_price,
        alpha=alpha,
        liq_score=liq_val,
        funding_score=funding_val,
        clob_cvd_score=clob_cvd_val,
        exchange_div_score=div_score,
        pcr_score=pcr_val,
        tick_velocity_score=tick_vel_val,
        trade_imbalance_score=trade_imb_val,
        oracle_lag_score=oracle_lag_val,
        hawkes_score=hawkes_val,
        mlofi_score=mlofi_val,
        mom_30s_score=mom_30s_val,
    )
