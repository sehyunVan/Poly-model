"""
앙상블 예측 모듈 — predict_probability() 공개 인터페이스.

카테고리별 가중치(config/model_weights.yaml)로
baseline / tree / llm_text 세 소스를 가중 평균해 최종 P_R을 계산한다.

학습된 모델이 없으면 naive prior(P_M + 텍스트 신호 보정)를 반환한다.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from features.schemas import FeatureVector
from .schemas import PredictionResult, FEATURE_NAMES
from .baseline import LogisticBaseline
from .tree_model import TreeModel

# ── 설정 로드 ─────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path("config/model_weights.yaml")

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # 기본값 하드코딩 fallback
        return {
            "category_weights": {
                "politics": {"baseline": 0.40, "tree": 0.40, "llm_text": 0.20},
                "crypto":   {"baseline": 0.30, "tree": 0.60, "llm_text": 0.10},
                "sports":   {"baseline": 0.50, "tree": 0.40, "llm_text": 0.10},
                "other":    {"baseline": 0.45, "tree": 0.45, "llm_text": 0.10},
            },
            "naive_prior": {"w_market": 0.55, "w_llm": 0.30, "w_sentiment": 0.15},
        }


# ── 싱글턴 모델 인스턴스 ──────────────────────────────────────────────────────
# 프로세스 내에서 한 번만 로드한다.

_baseline_model: Optional[LogisticBaseline] = None
_tree_model: Optional[TreeModel] = None
_models_loaded: bool = False

def _ensure_models_loaded(
    baseline_path: str | Path = "models/baseline.pkl",
    tree_path: str | Path = "models/tree_model.pkl",
) -> tuple[bool, bool]:
    """
    디스크에서 모델을 로드한다 (최초 1회).

    Returns:
        (baseline_loaded, tree_loaded)
    """
    global _baseline_model, _tree_model, _models_loaded

    if _models_loaded:
        return (
            _baseline_model is not None and _baseline_model.is_fitted,
            _tree_model is not None and _tree_model.is_fitted,
        )

    _baseline_model = LogisticBaseline()
    _tree_model = TreeModel()

    b_ok = _baseline_model.load(baseline_path)
    t_ok = _tree_model.load(tree_path)
    _models_loaded = True

    if not b_ok:
        print("[prediction.ensemble] baseline 모델 없음 → naive prior 사용")
    if not t_ok:
        print("[prediction.ensemble] tree 모델 없음 → naive prior 사용")

    return b_ok, t_ok


def reload_models(
    baseline_path: str | Path = "models/baseline.pkl",
    tree_path: str | Path = "models/tree_model.pkl",
) -> None:
    """재학습 후 모델을 다시 로드한다 (training.py에서 호출)."""
    global _models_loaded
    _models_loaded = False
    _ensure_models_loaded(baseline_path, tree_path)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, v))


def _naive_prior(fv: FeatureVector, cfg: dict) -> float:
    """
    학습된 모델이 없을 때 사용하는 naive prior.

    P_R ≈ w_market * P_M + w_llm * llm_intuition + w_sentiment * (0.5 + sentiment/2)

    시장 가격(P_M)을 주축으로 텍스트 신호를 소폭 반영한다.
    """
    w = cfg.get("naive_prior", {"w_market": 0.55, "w_llm": 0.30, "w_sentiment": 0.15})
    p_m = fv.structured.price_current
    llm = fv.text.llm_intuition_score
    sent = fv.text.sentiment_score            # -1 ~ +1
    sent_prob = 0.5 + sent / 2               # → 0 ~ 1

    p_r = w["w_market"] * p_m + w["w_llm"] * llm + w["w_sentiment"] * sent_prob
    return _clamp(p_r)


def _confidence_from_predictions(preds: list[float]) -> float:
    """
    모델 예측값들의 일치도에서 신뢰도를 계산한다.

    신뢰도 = 1 - (표준편차 / 0.5)
    표준편차 0 → 신뢰도 1.0 (완전 일치)
    표준편차 0.5 (최대) → 신뢰도 0.0
    """
    if len(preds) < 2:
        return 0.5
    std = statistics.stdev(preds)
    return _clamp(1.0 - (std / 0.5), lo=0.0, hi=1.0)


# ── 공개 API ──────────────────────────────────────────────────────────────────

def predict_probability(
    fv: FeatureVector,
    baseline_path: str | Path = "models/baseline.pkl",
    tree_path: str | Path = "models/tree_model.pkl",
) -> PredictionResult:
    """
    FeatureVector를 입력으로 받아 최종 P_R을 추정한다.

    처리 흐름:
        1. 모델 로드 (최초 1회)
        2. 각 소스별 예측:
             baseline  → LogisticRegression 예측 확률
             tree      → XGBoost 예측 확률
             llm_text  → TextFeatures.llm_intuition_score (직접 사용)
        3. 카테고리별 가중치로 가중 평균 → P_R
        4. 모델 간 일치도 → confidence
        5. 미학습 시 naive prior 반환

    Args:
        fv:             FeatureVector (TASK-2 출력)
        baseline_path:  baseline 모델 pkl 경로
        tree_path:      tree 모델 pkl 경로

    Returns:
        PredictionResult

    예시:
        >>> result = predict_probability(feature_vector)
        >>> print(f"P_M={result.P_M:.3f}, P_R={result.P_R:.3f}, alpha={result.P_R - result.P_M:.3f}")
    """
    cfg = _load_config()
    b_ok, t_ok = _ensure_models_loaded(baseline_path, tree_path)

    p_m = _clamp(fv.structured.price_current)
    category = fv.category
    weights = cfg["category_weights"].get(category, cfg["category_weights"]["other"])

    now = datetime.now(timezone.utc)

    # ── 학습된 모델이 하나도 없으면 naive prior ──────────────────────────
    if not b_ok and not t_ok:
        p_r = _naive_prior(fv, cfg)
        llm_score = fv.text.llm_intuition_score
        return PredictionResult(
            market_id=fv.market_id,
            timestamp=now,
            P_M=p_m,
            P_R=p_r,
            model_breakdown={
                "baseline": p_r,
                "tree":     p_r,
                "llm_text": llm_score,
                "ensemble": p_r,
            },
            confidence=0.4,   # 미학습 상태이므로 낮은 신뢰도
            is_trained=False,
        )

    # ── 각 소스별 예측 수집 ──────────────────────────────────────────────
    llm_score = _clamp(fv.text.llm_intuition_score)

    p_baseline = _baseline_model.predict_proba(fv) if b_ok else _naive_prior(fv, cfg)
    p_tree     = _tree_model.predict_proba(fv)     if t_ok else _naive_prior(fv, cfg)

    # ── 카테고리 가중 평균 ────────────────────────────────────────────────
    w_b = weights["baseline"]
    w_t = weights["tree"]
    w_l = weights["llm_text"]

    # 사용 불가능한 모델의 가중치를 나머지에 재분배
    if not b_ok:
        w_t += w_b * (w_t / (w_t + w_l)) if (w_t + w_l) > 0 else w_b / 2
        w_l += w_b * (w_l / (w_t + w_l)) if (w_t + w_l) > 0 else w_b / 2
        w_b = 0.0
    if not t_ok:
        w_b += w_t * (w_b / (w_b + w_l)) if (w_b + w_l) > 0 else w_t / 2
        w_l += w_t * (w_l / (w_b + w_l)) if (w_b + w_l) > 0 else w_t / 2
        w_t = 0.0

    # 가중치 정규화
    total_w = w_b + w_t + w_l
    if total_w > 0:
        w_b /= total_w; w_t /= total_w; w_l /= total_w

    p_r = _clamp(w_b * p_baseline + w_t * p_tree + w_l * llm_score)

    # ── 신뢰도 ───────────────────────────────────────────────────────────
    contributing = [v for v, w in [(p_baseline, w_b), (p_tree, w_t), (llm_score, w_l)] if w > 0]
    model_agreement = _confidence_from_predictions(contributing)

    # Blend model agreement with LLM's own uncertainty estimate.
    # uncertainty_score=1.0 means Claude found conflicting/sparse information →
    # penalise confidence regardless of model agreement (which may be spurious).
    llm_certainty = _clamp(1.0 - fv.text.uncertainty_score, lo=0.0, hi=1.0)
    confidence = _clamp(0.5 * model_agreement + 0.5 * llm_certainty, lo=0.0, hi=1.0)

    return PredictionResult(
        market_id=fv.market_id,
        timestamp=now,
        P_M=p_m,
        P_R=p_r,
        model_breakdown={
            "baseline": round(p_baseline, 4),
            "tree":     round(p_tree, 4),
            "llm_text": round(llm_score, 4),
            "ensemble": round(p_r, 4),
        },
        confidence=round(confidence, 4),
        is_trained=b_ok or t_ok,
    )
