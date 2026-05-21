"""
LLM 기반 텍스트 피처 생성 모듈.

Claude API(claude-sonnet-4-6)를 호출해 뉴스·트윗을 분석하고
TextFeatures(감성 점수, 불확실성, 직관 확률 등)를 생성한다.

중요:
    llm_intuition_score 는 Claude의 직관적 YES 가능성 추정이다.
    이 값은 반드시 예측 모델의 보조 피처로만 사용해야 하며,
    최종 확률 P_R 로 직접 사용해서는 안 된다.

환경 변수:
    ANTHROPIC_API_KEY - Claude API 키 (필수)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from data.schemas import Article, Tweet
from .schemas import TextFeatures

# ── 프롬프트 템플릿 ───────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
당신은 예측 마켓 분석 전문가다.
아래 예측 마켓 이벤트와 관련 텍스트를 분석하고, 지정된 JSON 형식으로만 응답하라.

─────────────────────────────────────
[마켓 정보]
제목: {title}
설명: {description}
카테고리: {category}
마감까지: {hours_to_close:.1f}시간
─────────────────────────────────────
[관련 뉴스 ({article_count}건)]
{articles_text}
─────────────────────────────────────
[관련 트윗 ({tweet_count}건)]
{tweets_text}
─────────────────────────────────────

[분석 지침]
1. YES 방향 근거와 NO 방향 근거를 각각 명확히 파악한다.
2. sentiment_score:
   - 뉴스·트윗의 전반적 감성을 YES 방향 기준으로 평가한다.
   - +1에 가까울수록 "YES 결과를 지지하는 정보가 많다".
   - -1에 가까울수록 "NO 결과를 지지하는 정보가 많다".
3. uncertainty_score:
   - 정보 간 의견 충돌이 크거나, 데이터가 부족하거나, 예측이 어려울수록 높다.
   - 정보가 없으면(뉴스·트윗 0건) 0.7 이상으로 설정한다.
4. negative_risk_count:
   - 이벤트가 NO로 끝날 수 있는 구체적 위험 요인의 수를 센다.
   - 예: "경쟁 후보 지지율 상승", "규제 불확실성", "부상 가능성" 등 각각 +1.
5. llm_intuition_score:
   - "이 이벤트가 YES로 끝날 가능성은 얼마나 되는가?"에 대한 직관적 추정.
   - 0.5 = 중립 / 0.7 = YES 가능성 높음 / 0.3 = NO 가능성 높음.
   - !! 이 값은 보조 피처로만 사용된다. 최종 확률 P_R에 직접 사용하지 않는다.
6. summary: 핵심 근거를 포함한 한 줄 요약 (한국어, 50자 이내).

[응답 형식 — JSON만 출력, 다른 텍스트 없이]
{{
  "sentiment_score": <float, -1.0 ~ 1.0>,
  "uncertainty_score": <float, 0.0 ~ 1.0>,
  "negative_risk_count": <int, 0 이상>,
  "llm_intuition_score": <float, 0.0 ~ 1.0>,
  "summary": "<string>"
}}
"""

# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _format_articles(articles: list[Article], max_items: int = 5) -> str:
    """기사 목록을 프롬프트용 텍스트로 변환한다."""
    if not articles:
        return "(관련 뉴스 없음)"
    lines = []
    for i, a in enumerate(articles[:max_items], 1):
        summary = a.summary[:150] if a.summary else ""
        lines.append(f"{i}. [{a.source}] {a.title}")
        if summary:
            lines.append(f"   → {summary}")
    return "\n".join(lines)


def _format_tweets(tweets: list[Tweet], max_items: int = 8) -> str:
    """트윗 목록을 프롬프트용 텍스트로 변환한다."""
    if not tweets:
        return "(관련 트윗 없음)"
    lines = []
    for i, t in enumerate(tweets[:max_items], 1):
        text = t.text[:200]
        lines.append(f"{i}. {t.author}: {text}")
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    """
    LLM 응답에서 JSON 객체를 추출한다.
    ```json ... ``` 마크다운 블록이 있어도 처리한다.
    """
    # 마크다운 코드 블록 제거
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # 중괄호로 시작하는 첫 번째 JSON 블록 추출
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 블록을 찾을 수 없음: {raw[:200]}")
    return json.loads(match.group())


def _default_text_features(market_id: str) -> TextFeatures:
    """API 실패 시 반환할 중립 기본값."""
    return TextFeatures(
        market_id=market_id,
        timestamp=datetime.now(timezone.utc),
        sentiment_score=0.0,
        uncertainty_score=0.7,   # 데이터 없으면 불확실성 높게
        negative_risk_count=0,
        llm_intuition_score=0.5,
        summary="텍스트 데이터 없음 또는 분석 실패",
        source_count=0,
    )


# ── 공개 API ──────────────────────────────────────────────────────────────────

def build_text_features(
    market_id: str,
    market_title: str,
    market_description: str,
    articles: list[Article],
    tweets: list[Tweet],
    category: str = "other",
    seconds_to_close: float = 0.0,
    model: str = "claude-sonnet-4-6",
) -> TextFeatures:
    """
    Claude API를 사용해 텍스트 기반 피처를 생성한다.

    뉴스·트윗이 없으면 마켓 제목과 설명만으로 분석한다.
    ANTHROPIC_API_KEY가 없거나 API 호출에 실패하면
    중립 기본값(uncertainty_score=0.7)을 반환한다.

    Args:
        market_id:          마켓 condition_id
        market_title:       마켓 제목 (예측 질문)
        market_description: 마켓 상세 설명
        articles:           관련 뉴스 기사 목록
        tweets:             관련 트윗 목록
        category:           마켓 카테고리 (프롬프트 컨텍스트용)
        seconds_to_close:   마감까지 남은 초
        model:              사용할 Claude 모델 ID

    Returns:
        TextFeatures 객체.

    예시:
        >>> from data import get_news
        >>> from datetime import datetime, timezone, timedelta
        >>> articles = get_news("US election 2024", datetime.now(timezone.utc) - timedelta(days=1))
        >>> feats = build_text_features(
        ...     "0xabc", "Will Biden win?", "Resolves YES if...",
        ...     articles, tweets=[]
        ... )
        >>> print(feats.sentiment_score, feats.summary)
    """
    import anthropic  # 지연 임포트: API 키 없어도 모듈 임포트 가능

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[features.text] ANTHROPIC_API_KEY 없음 → 기본값 사용")
        return _default_text_features(market_id)

    # 프롬프트 조립
    prompt = _ANALYSIS_PROMPT.format(
        title=market_title,
        description=market_description[:500] if market_description else "(설명 없음)",
        category=category,
        hours_to_close=max(0.0, seconds_to_close / 3600),
        article_count=len(articles),
        articles_text=_format_articles(articles),
        tweet_count=len(tweets),
        tweets_text=_format_tweets(tweets),
    )

    source_count = len(articles) + len(tweets)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.2,    # 일관된 출력을 위해 낮은 temperature
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
    except Exception as exc:
        print(f"[features.text] Claude API 호출 실패 {market_id}: {exc}")
        return _default_text_features(market_id)

    # JSON 파싱
    try:
        data = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[features.text] JSON 파싱 실패 {market_id}: {exc}\n응답: {raw_text[:300]}")
        return _default_text_features(market_id)

    # 값 범위 클램핑 및 타입 변환
    def _f(key: str, default: float, lo: float = -1.0, hi: float = 1.0) -> float:
        try:
            return max(lo, min(hi, float(data.get(key, default))))
        except (TypeError, ValueError):
            return default

    return TextFeatures(
        market_id=market_id,
        timestamp=datetime.now(timezone.utc),
        sentiment_score=_f("sentiment_score", 0.0, -1.0, 1.0),
        uncertainty_score=_f("uncertainty_score", 0.5, 0.0, 1.0),
        negative_risk_count=max(0, int(data.get("negative_risk_count", 0))),
        llm_intuition_score=_f("llm_intuition_score", 0.5, 0.0, 1.0),
        summary=str(data.get("summary", ""))[:200],
        source_count=source_count,
    )
