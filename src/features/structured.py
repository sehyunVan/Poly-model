"""
수치·범주형 피처 계산 모듈.

시장 데이터(가격 히스토리, 오더북, 체결 기록)를 가져와
StructuredFeatures를 계산한다.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

from data import (
    get_market,
    get_market_history,
    get_orderbook,
    get_trades,
    get_polls,
    OHLCVPoint,
)
from .schemas import StructuredFeatures

# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _stdev(values: list[float]) -> float:
    """표준편차. 데이터 포인트 2개 미만이면 0 반환."""
    if len(values) < 2:
        return 0.0
    try:
        return statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0


def _safe_return(current: float, past: float) -> float:
    """수익률 = (current - past) / past. past ≈ 0이면 0 반환."""
    if past < 1e-9:
        return 0.0
    return (current - past) / past


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── 카테고리별 추가 피처 ──────────────────────────────────────────────────────

def _politics_features(market_id: str, seconds_to_close: float) -> dict:
    """정치 카테고리 전용 피처."""
    extra: dict = {}

    # 여론조사 YES 비율 평균
    polls = get_polls(market_id)
    if polls:
        avg_yes = sum(p.yes_pct for p in polls) / len(polls)
        extra["polls_yes_pct"] = _clamp(avg_yes)

    # 선거일까지 남은 일수 (seconds_to_close 기반)
    extra["days_to_election"] = max(0.0, seconds_to_close / 86400)

    return extra


def _crypto_features() -> dict:
    """
    크립토 카테고리 전용 피처.
    기초자산 가격·변동성은 별도 가격 피드가 필요하므로
    현재는 stub이다 (TASK-1 external.py 확장 후 연동 예정).
    """
    # TODO: CoinGecko / Binance API 연동
    return {}


def _sports_features(seconds_to_close: float) -> dict:
    """
    스포츠 카테고리 전용 피처.
    경기 시작 시각 = 마켓 마감 시각과 동일하다고 가정 (대부분의 경우).
    """
    return {"seconds_to_game": max(0.0, seconds_to_close)}


# ── 공개 API ──────────────────────────────────────────────────────────────────

def build_structured_features(
    market_id: str,
    now: Optional[datetime] = None,
) -> StructuredFeatures:
    """
    시장 데이터로부터 StructuredFeatures를 계산한다.

    데이터 수집 흐름:
        1. 마켓 메타 정보 조회 (카테고리, 마감 시각)
        2. 최근 24h 1시간봉 → price_24h_return, volatility_24h
        3. 최근 1h  1분봉  → price_1h_return,  volatility_1h
        4. 오더북 스냅샷   → spread, orderbook_imbalance, price_current
        5. 체결 기록 24h   → volume_24h
        6. 카테고리별 추가 피처

    API 실패 / 데이터 부족 시 각 피처는 중립 기본값으로 채워진다.

    Args:
        market_id: 마켓 condition_id
        now:       기준 시각 (None이면 현재 UTC 시각 사용)

    Returns:
        StructuredFeatures 객체.

    예시:
        >>> from datetime import datetime, timezone
        >>> fv = build_structured_features("0xabc123", now=datetime.now(timezone.utc))
        >>> print(fv.price_current, fv.spread)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # ── 1. 마켓 메타 ─────────────────────────────────────────────────────
    market = get_market(market_id)
    category = market.category if market else "other"
    seconds_to_close = market.seconds_to_close if market else 0.0

    # ── 2. 24h 가격 히스토리 (1시간봉) ───────────────────────────────────
    yes_token_id = market.yes_token_id if market else ""
    start_24h = now - timedelta(hours=24)
    history_24h: list[OHLCVPoint] = []
    try:
        history_24h = get_market_history(
            market_id, start_24h, now, interval="1h", token_id=yes_token_id
        )
    except Exception as exc:
        print(f"[features.structured] 24h 히스토리 오류 {market_id}: {exc}")

    closes_24h = [p.close for p in history_24h]

    # ── 3. 1h 가격 히스토리 (1분봉) ──────────────────────────────────────
    start_1h = now - timedelta(hours=1)
    history_1h: list[OHLCVPoint] = []
    try:
        history_1h = get_market_history(
            market_id, start_1h, now, interval="1m", token_id=yes_token_id
        )
    except Exception as exc:
        print(f"[features.structured] 1h 히스토리 오류 {market_id}: {exc}")

    closes_1h = [p.close for p in history_1h]

    # ── 4. 오더북 ─────────────────────────────────────────────────────────
    try:
        ob = get_orderbook(market_id)
    except Exception as exc:
        print(f"[features.structured] 오더북 오류 {market_id}: {exc}")
        ob = None

    price_current = 0.5   # 기본값
    spread = 0.02
    imbalance = 0.5

    if ob is not None:
        if ob.mid_price is not None:
            price_current = ob.mid_price
        elif closes_24h:
            price_current = closes_24h[-1]
        spread = max(0.0, ob.spread) if ob.spread is not None else 0.02
        imbalance = ob.imbalance()
    elif closes_24h:
        price_current = closes_24h[-1]

    price_current = _clamp(price_current, 0.001, 0.999)

    # ── 5. 수익률 계산 ────────────────────────────────────────────────────
    # 24h 수익률: 24h 전 첫 포인트 open → 현재 가격
    price_24h_return = 0.0
    if closes_24h:
        past_24h = history_24h[0].open if history_24h else price_current
        price_24h_return = _safe_return(price_current, past_24h)

    # 1h 수익률: 1h 전 첫 포인트 open → 현재 가격
    price_1h_return = 0.0
    if closes_1h:
        past_1h = history_1h[0].open if history_1h else price_current
        price_1h_return = _safe_return(price_current, past_1h)

    # ── 6. 변동성 ─────────────────────────────────────────────────────────
    volatility_24h = _stdev(closes_24h)
    volatility_1h  = _stdev(closes_1h)

    # ── 7. 24h 거래량 ─────────────────────────────────────────────────────
    volume_24h = 0.0
    try:
        trades = get_trades(market_id, start_24h, now)
        volume_24h = sum(t.size for t in trades)
    except Exception as exc:
        print(f"[features.structured] 거래량 오류 {market_id}: {exc}")

    # ── 8. 카테고리별 추가 피처 ───────────────────────────────────────────
    extra: dict = {}
    if category == "politics":
        extra = _politics_features(market_id, seconds_to_close)
    elif category == "crypto":
        extra = _crypto_features()
    elif category == "sports":
        extra = _sports_features(seconds_to_close)

    return StructuredFeatures(
        market_id=market_id,
        timestamp=now,
        category=category,
        seconds_to_close=seconds_to_close,
        price_current=price_current,
        price_1h_return=price_1h_return,
        price_24h_return=price_24h_return,
        volatility_1h=volatility_1h,
        volatility_24h=volatility_24h,
        volume_24h=volume_24h,
        spread=spread,
        orderbook_imbalance=imbalance,
        **extra,
    )
