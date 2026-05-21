"""
예측 모듈 공통 스키마.
"""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


# 모든 모델이 공유하는 피처 이름 목록 (순서 고정)
FEATURE_NAMES: list[str] = [
    "price_current",
    "price_1h_return",
    "price_24h_return",
    "volatility_1h",
    "volatility_24h",
    "volume_24h",
    "spread",
    "orderbook_imbalance",
    "seconds_to_close",          # to_flat_dict 키와 다름 → 변환 필요
    "sentiment_score",
    "uncertainty_score",
    "negative_risk_count",
    "llm_intuition_score",
    "polls_yes_pct",
    "category_politics",
    "category_crypto",
    "category_sports",
]


class PredictionResult(BaseModel):
    """
    단일 마켓에 대한 예측 결과.

    P_M: 시장 암묵 확률 (YES 토큰 현재 가격)
    P_R: 모델 추정 실제 확률
    alpha = P_R - P_M 은 시그널 모듈이 계산한다.

    예시:
    {
        "market_id": "0xabc",
        "timestamp": "2024-01-15T12:00:00Z",
        "P_M": 0.62,
        "P_R": 0.71,
        "model_breakdown": {"baseline": 0.69, "tree": 0.73, "llm_text": 0.70, "ensemble": 0.71},
        "confidence": 0.82
    }
    """
    market_id: str
    timestamp: datetime

    P_M: float = Field(..., ge=0.0, le=1.0, description="시장 암묵 확률 (현재 가격)")
    P_R: float = Field(..., ge=0.0, le=1.0, description="모델 추정 실제 확률")

    model_breakdown: dict = Field(
        default_factory=dict,
        description="모델별 예측값: baseline / tree / llm_text / ensemble",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="예측 신뢰도 (1 - 정규화된 분산). 모델 간 일치도가 높을수록 높다.",
    )
    is_trained: bool = Field(
        default=False,
        description="학습된 모델 사용 여부. False면 naive prior로 추정한 값.",
    )
