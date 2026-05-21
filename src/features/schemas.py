"""
피처 엔지니어링 레이어 공통 스키마.

StructuredFeatures: 수치·범주형 피처 (가격, 유동성, 마켓 메타)
TextFeatures:       LLM 기반 텍스트 피처 (감성, 불확실성, 직관 점수)
FeatureVector:      두 피처 세트의 결합 — 예측 모델의 최종 입력
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StructuredFeatures(BaseModel):
    """
    수치·범주형 피처 벡터.

    price_current = YES 토큰 오더북 mid 가격 = 시장 암묵 확률 P_M 기반값.
    return / volatility는 해당 구간의 close 가격 시계열로 계산한다.

    예시:
    {
        "market_id": "0xabc",
        "timestamp": "2024-01-15T12:00:00Z",
        "category": "politics",
        "seconds_to_close": 86400.0,
        "price_current": 0.65,
        "price_1h_return": 0.015,
        "price_24h_return": -0.03,
        "volatility_1h": 0.008,
        "volatility_24h": 0.021,
        "volume_24h": 5200.0,
        "spread": 0.02,
        "orderbook_imbalance": 0.55
    }
    """
    market_id: str
    timestamp: datetime

    # ── 마켓 메타 ─────────────────────────────────────────────────────────
    category: str = Field(..., description="politics | crypto | sports | other")
    seconds_to_close: float = Field(..., description="마감까지 남은 초, 마감 후 음수")

    # ── 가격 / 유동성 ─────────────────────────────────────────────────────
    price_current: float = Field(
        ..., ge=0.0, le=1.0,
        description="현재 YES 토큰 mid 가격 (≈ P_M)",
    )
    price_1h_return: float = Field(
        default=0.0,
        description="최근 1h 수익률 = (current - price_1h_ago) / price_1h_ago",
    )
    price_24h_return: float = Field(
        default=0.0,
        description="최근 24h 수익률",
    )
    volatility_1h: float = Field(
        default=0.0, ge=0.0,
        description="최근 1h 1분봉 close 가격의 표준편차",
    )
    volatility_24h: float = Field(
        default=0.0, ge=0.0,
        description="최근 24h 1시간봉 close 가격의 표준편차",
    )
    volume_24h: float = Field(
        default=0.0, ge=0.0,
        description="최근 24h 체결 거래량 (USDC)",
    )
    spread: float = Field(
        default=0.02, ge=0.0,
        description="오더북 최우선 ask - bid",
    )
    orderbook_imbalance: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="bid잔량 / (bid+ask잔량), 상위 10레벨 합산",
    )

    # ── 카테고리별 선택 피처 ──────────────────────────────────────────────
    polls_yes_pct: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="[politics] 최근 여론조사 YES 응답 비율 평균",
    )
    days_to_election: Optional[float] = Field(
        default=None,
        description="[politics] 선거일까지 남은 일수",
    )
    crypto_underlying_volatility: Optional[float] = Field(
        default=None, ge=0.0,
        description="[crypto] 기초자산 24h 변동성",
    )
    seconds_to_game: Optional[float] = Field(
        default=None,
        description="[sports] 경기 시작까지 남은 초",
    )


class TextFeatures(BaseModel):
    """
    LLM(Claude) 기반 텍스트 분석 피처.

    llm_intuition_score 는 Claude의 직관적 YES 가능성 추정값이다.
    이 값은 구조화 모델의 보조 피처(입력)로만 사용하며,
    최종 확률 P_R 로 직접 사용해서는 안 된다.

    예시:
    {
        "market_id": "0xabc",
        "timestamp": "2024-01-15T12:00:00Z",
        "sentiment_score": 0.35,
        "uncertainty_score": 0.6,
        "negative_risk_count": 2,
        "llm_intuition_score": 0.62,
        "summary": "여론조사 박빙, 일부 긍정 신호 있으나 불확실성 높음"
    }
    """
    market_id: str
    timestamp: datetime

    sentiment_score: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="전반 감성: -1(부정) ~ +1(긍정), YES 방향이면 양수",
    )
    uncertainty_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="예측 불확실성: 0(확실) ~ 1(매우 불확실)",
    )
    negative_risk_count: int = Field(
        default=0, ge=0,
        description="이벤트가 NO로 끝날 수 있는 부정적 리스크 키워드 수",
    )
    llm_intuition_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Claude의 직관적 YES 가능성 추정 (보조 피처 전용)",
    )
    summary: str = Field(
        default="",
        description="한 줄 요약 (모니터링 및 로그용)",
    )
    source_count: int = Field(
        default=0, ge=0,
        description="분석에 사용된 뉴스+트윗 수",
    )


class FeatureVector(BaseModel):
    """
    예측 모델의 최종 입력 벡터.

    structured + text 두 세트를 하나로 묶는다.
    예측 모듈(prediction layer)은 이 타입을 입력으로 받는다.

    직렬화 예시:
        fv.model_dump_json()  → JSON 문자열
        FeatureVector.model_validate_json(json_str)  → 역직렬화
    """
    structured: StructuredFeatures
    text: TextFeatures

    @property
    def market_id(self) -> str:
        return self.structured.market_id

    @property
    def timestamp(self) -> datetime:
        return self.structured.timestamp

    @property
    def category(self) -> str:
        return self.structured.category

    def to_flat_dict(self) -> dict:
        """
        모델 학습용 플랫 딕셔너리.
        선택 피처(None)는 카테고리별 중앙값(0.5 등)으로 대체한다.
        """
        s = self.structured
        t = self.text
        return {
            # 메타
            "category_politics": 1 if s.category == "politics" else 0,
            "category_crypto":   1 if s.category == "crypto"   else 0,
            "category_sports":   1 if s.category == "sports"   else 0,
            "seconds_to_close":  s.seconds_to_close,
            # 가격·유동성
            "price_current":         s.price_current,
            "price_1h_return":       s.price_1h_return,
            "price_24h_return":      s.price_24h_return,
            "volatility_1h":         s.volatility_1h,
            "volatility_24h":        s.volatility_24h,
            "volume_24h":            s.volume_24h,
            "spread":                s.spread,
            "orderbook_imbalance":   s.orderbook_imbalance,
            # 카테고리별 (없으면 중립값)
            "polls_yes_pct":                    s.polls_yes_pct if s.polls_yes_pct is not None else 0.5,
            "crypto_underlying_volatility":     s.crypto_underlying_volatility if s.crypto_underlying_volatility is not None else 0.0,
            # 텍스트
            "sentiment_score":       t.sentiment_score,
            "uncertainty_score":     t.uncertainty_score,
            "negative_risk_count":   float(t.negative_risk_count),
            "llm_intuition_score":   t.llm_intuition_score,
        }
