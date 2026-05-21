"""
트리 기반 앙상블 예측 모델.

XGBoost가 설치되어 있으면 XGBClassifier를 사용하고,
없으면 sklearn GradientBoostingClassifier로 자동 fallback한다.
인터페이스는 baseline.LogisticBaseline과 동일하다.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from features.schemas import FeatureVector
from .baseline import _extract_feature_array   # 피처 추출 공유
from .schemas import FEATURE_NAMES
from ._integrity import save_hash, verify_hash

_log = logging.getLogger("prediction.tree_model")

DEFAULT_MODEL_PATH = Path("models/tree_model.pkl")

# XGBoost 가용 여부 확인
try:
    from xgboost import XGBClassifier as _XGB
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def _make_classifier():
    """환경에 따라 최적의 트리 분류기를 반환한다."""
    if _XGBOOST_AVAILABLE:
        return _XGB(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        print("[prediction.tree_model] XGBoost 없음 → GradientBoostingClassifier 사용")
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )


class TreeModel:
    """
    XGBoost (또는 GradientBoosting) 기반 예측 모델.

    LogisticBaseline과 동일한 인터페이스를 제공한다.
    피처 스케일링이 불필요하므로 파이프라인 없이 직접 사용한다.

    사용 예시:
        model = TreeModel()
        model.fit(X, y)
        p = model.predict_proba(feature_vector)  # 0~1 float
    """

    def __init__(self) -> None:
        self._clf = None
        self.is_fitted: bool = False
        self._backend: str = "xgboost" if _XGBOOST_AVAILABLE else "gradient_boosting"

    @property
    def backend(self) -> str:
        return self._backend

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: (n_samples, len(FEATURE_NAMES)) 피처 행렬
            y: (n_samples,) 레이블 {0, 1}
        """
        if len(np.unique(y)) < 2:
            raise ValueError("학습 데이터에 두 클래스(0, 1)가 모두 있어야 합니다.")

        self._clf = _make_classifier()
        self._clf.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, fv: FeatureVector) -> float:
        """
        YES(=1) 클래스의 예측 확률을 반환한다.

        Returns:
            0~1 float. 모델 미학습 시 0.5 반환.
        """
        if not self.is_fitted or self._clf is None:
            return 0.5

        x = _extract_feature_array(fv).reshape(1, -1)
        proba = self._clf.predict_proba(x)[0]
        # classes_ 순서가 [0,1]임을 가정 (XGBoost, sklearn 모두 동일)
        return float(proba[1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """배치 예측."""
        if not self.is_fitted or self._clf is None:
            return np.full(len(X), 0.5)
        return self._clf.predict_proba(X)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        """
        피처 중요도를 반환한다 (학습 후에만 유효).

        Returns:
            {feature_name: importance_score} 딕셔너리 (합계 ≈ 1.0)
        """
        if not self.is_fitted or self._clf is None:
            return {}
        importances = self._clf.feature_importances_
        total = importances.sum()
        if total == 0:
            return {}
        return {name: float(imp / total) for name, imp in zip(FEATURE_NAMES, importances)}

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """학습된 모델을 pickle로 저장하고 SHA-256 사이드카를 기록한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self._clf,
                "is_fitted": self.is_fitted,
                "backend": self._backend,
            }, f)
        save_hash(path)

    def load(self, path: str | Path = DEFAULT_MODEL_PATH) -> bool:
        """
        저장된 모델을 로드한다. 로드 전 SHA-256 무결성 검증을 수행한다.

        Returns:
            성공 시 True, 파일 없거나 해시 불일치 또는 실패 시 False.
        """
        path = Path(path)
        if not path.exists():
            return False
        if not verify_hash(path):
            _log.error("tree 모델 로드 취소 — 무결성 검증 실패: %s", path)
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._clf = data["clf"]
            self.is_fitted = data["is_fitted"]
            self._backend = data.get("backend", self._backend)
            return True
        except Exception as exc:
            _log.error("tree 모델 로드 실패 %s: %s", path, exc)
            self.is_fitted = False
            return False
