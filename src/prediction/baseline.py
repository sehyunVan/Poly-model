"""
로지스틱 회귀 기반 베이스라인 예측 모델.

sklearn.linear_model.LogisticRegression을 사용한다.
피처 스케일링(StandardScaler)을 파이프라인으로 포함한다.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.schemas import FeatureVector
from .schemas import FEATURE_NAMES
from ._integrity import save_hash, verify_hash

_log = logging.getLogger("prediction.baseline")

# 모델 기본 저장 경로
DEFAULT_MODEL_PATH = Path("models/baseline.pkl")


def _extract_feature_array(fv: FeatureVector) -> np.ndarray:
    """
    FeatureVector → FEATURE_NAMES 순서의 1-D numpy 배열.

    to_flat_dict()의 키 이름과 FEATURE_NAMES이 다른 경우:
      - "seconds_to_close" → flat_dict의 "seconds_to_close" 키 사용
    """
    flat = fv.to_flat_dict()
    # to_flat_dict는 "seconds_to_close"를 그대로 포함하지 않으므로 structured에서 직접 가져온다
    flat["seconds_to_close"] = fv.structured.seconds_to_close

    row = [flat.get(name, 0.0) for name in FEATURE_NAMES]
    return np.array(row, dtype=float)


class LogisticBaseline:
    """
    로지스틱 회귀 기반 베이스라인 모델.

    파이프라인 구성:
        StandardScaler → LogisticRegression(C=1.0, max_iter=1000)

    학습/저장/로드:
        model = LogisticBaseline()
        model.fit(X, y)          # X: (n_samples, n_features), y: (n_samples,) {0,1}
        model.save("models/baseline.pkl")

        model2 = LogisticBaseline()
        model2.load("models/baseline.pkl")
        p = model2.predict_proba(feature_vector)  # 0~1 float
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        모델을 학습한다.

        Args:
            X: (n_samples, len(FEATURE_NAMES)) 피처 행렬
            y: (n_samples,) 레이블 배열 {0, 1}
        """
        if len(np.unique(y)) < 2:
            raise ValueError("학습 데이터에 두 클래스(0, 1)가 모두 있어야 합니다.")

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self._pipeline.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, fv: FeatureVector) -> float:
        """
        YES(=1) 클래스의 예측 확률을 반환한다.

        Args:
            fv: FeatureVector

        Returns:
            0~1 float. 모델 미학습 시 0.5 반환.
        """
        if not self.is_fitted or self._pipeline is None:
            return 0.5

        x = _extract_feature_array(fv).reshape(1, -1)
        proba = self._pipeline.predict_proba(x)[0]
        # predict_proba 결과: [P(NO), P(YES)], classes_=[0,1]
        yes_idx = list(self._pipeline.named_steps["lr"].classes_).index(1)
        return float(proba[yes_idx])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """
        배치 예측. 학습 루프에서 성능 평가 시 사용.

        Returns:
            (n_samples,) YES 확률 배열
        """
        if not self.is_fitted or self._pipeline is None:
            return np.full(len(X), 0.5)

        proba = self._pipeline.predict_proba(X)
        yes_idx = list(self._pipeline.named_steps["lr"].classes_).index(1)
        return proba[:, yes_idx]

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """학습된 모델을 pickle로 저장하고 SHA-256 사이드카를 기록한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self._pipeline, "is_fitted": self.is_fitted}, f)
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
            _log.error("baseline 모델 로드 취소 — 무결성 검증 실패: %s", path)
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._pipeline = data["pipeline"]
            self.is_fitted = data["is_fitted"]
            return True
        except Exception as exc:
            _log.error("baseline 모델 로드 실패 %s: %s", path, exc)
            self.is_fitted = False
            return False
