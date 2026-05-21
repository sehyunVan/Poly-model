"""
features 패키지 공개 인터페이스.

예측 모듈은 이 패키지를 통해 피처 벡터를 생성한다:

    from features import build_feature_vector, FeatureVector
    fv = build_feature_vector("0xabc123")
"""

from .schemas import StructuredFeatures, TextFeatures, FeatureVector
from .structured import build_structured_features
from .text import build_text_features


def build_feature_vector(
    market_id: str,
    articles=None,
    tweets=None,
) -> FeatureVector:
    """
    단일 마켓에 대한 완전한 FeatureVector를 생성한다.

    내부적으로 build_structured_features → build_text_features 순서로 호출한다.
    articles/tweets가 None이면 빈 목록으로 처리한다 (텍스트 피처는 기본값).

    Args:
        market_id: 마켓 condition_id
        articles:  관련 뉴스 기사 (list[Article] 또는 None)
        tweets:    관련 트윗 (list[Tweet] 또는 None)

    Returns:
        FeatureVector (structured + text)
    """
    from data import get_market

    if articles is None:
        articles = []
    if tweets is None:
        tweets = []

    # 구조화 피처 생성
    structured = build_structured_features(market_id)

    # 마켓 텍스트 정보 조회
    market = get_market(market_id)
    title = market.title if market else market_id
    description = market.description if market else ""

    # 텍스트 피처 생성
    text = build_text_features(
        market_id=market_id,
        market_title=title,
        market_description=description,
        articles=articles,
        tweets=tweets,
        category=structured.category,
        seconds_to_close=structured.seconds_to_close,
    )

    return FeatureVector(structured=structured, text=text)


__all__ = [
    "StructuredFeatures",
    "TextFeatures",
    "FeatureVector",
    "build_structured_features",
    "build_text_features",
    "build_feature_vector",
]
