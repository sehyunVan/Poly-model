"""
Crypto-specific binary classifier: predicts Up (1) or Down (0) for a
Polymarket BTC/ETH up/down 5-minute market.

Features (all numeric, filled with neutral defaults if unavailable):
  rsi_14, momentum_1m, momentum_5m, momentum_15m, momentum_1h,
  bb_position_20, volume_surge, volatility_1h,
  fear_greed, llm_sentiment, news_count,
  poly_imbalance   ← Polymarket orderbook imbalance for this specific market

Model: LogisticRegression (sklearn) — fast to train, interpretable.
Falls back to LightGBM once we have ≥ 100 labeled samples.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("crypto.model")

FEATURE_NAMES = [
    "rsi_14",
    "momentum_1m",
    "momentum_5m",
    "momentum_15m",
    "momentum_1h",
    "bb_position_20",
    "volume_surge",
    "volatility_1h",
    "fear_greed",
    "llm_sentiment",
    "news_count",
    "poly_imbalance",
]

# Neutral defaults used when a feature cannot be computed
FEATURE_DEFAULTS = {
    "rsi_14":         50.0,
    "momentum_1m":    0.0,
    "momentum_5m":    0.0,
    "momentum_15m":   0.0,
    "momentum_1h":    0.0,
    "bb_position_20": 0.5,
    "volume_surge":   1.0,
    "volatility_1h":  0.01,
    "fear_greed":     0.5,
    "llm_sentiment":  0.0,
    "news_count":     0.0,
    "poly_imbalance": 0.5,
}

LGBM_THRESHOLD = 100  # switch to LightGBM after this many samples


@dataclass
class CryptoFeatureVector:
    features: dict[str, float]
    market_id: str
    symbol: str  # "BTC" or "ETH"

    def to_array(self) -> np.ndarray:
        return np.array([
            self.features.get(k, FEATURE_DEFAULTS[k])
            for k in FEATURE_NAMES
        ], dtype=float)


@dataclass
class CryptoPrediction:
    p_up: float       # probability of price going Up
    p_down: float     # = 1 - p_up
    alpha: float      # p_up - market_price_up  (our edge)
    direction: str    # "UP", "DOWN", or "NO_TRADE"
    confidence: float


class CryptoModel:
    """
    Wrapper around sklearn LogisticRegression (or LightGBM).
    Handles save/load and prediction with feature defaults.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._n_samples = 0
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                self._model = saved["model"]
                self._n_samples = saved.get("n_samples", 0)
                log.info("Crypto model loaded from %s  (n=%d)", self.model_path, self._n_samples)
            except Exception as exc:
                log.warning("Could not load crypto model: %s — will use neutral prior", exc)
                self._model = None
        else:
            log.info("No crypto model found at %s — using neutral prior", self.model_path)

    def predict(self, fv: CryptoFeatureVector) -> float:
        """Returns P(Up). Falls back to 0.5 if no model."""
        if self._model is None:
            return 0.5
        try:
            import pandas as pd
            import warnings
            X = pd.DataFrame([fv.features.get(k, FEATURE_DEFAULTS[k]) for k in FEATURE_NAMES],
                             index=FEATURE_NAMES).T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            up_idx = classes.index(1) if 1 in classes else 1
            return float(proba[up_idx])
        except Exception as exc:
            log.warning("Prediction failed: %s", exc)
            return 0.5

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Retrain on full labeled dataset.
        Uses LightGBM if ≥ LGBM_THRESHOLD samples, else LogisticRegression.
        """
        n = len(y)
        if n < 10:
            log.warning("Too few samples to train (%d). Skipping.", n)
            return

        n_up = int(y.sum())
        n_down = n - n_up
        if n_up == 0 or n_down == 0:
            log.warning("Single class only (up=%d, down=%d). Skipping.", n_up, n_down)
            return

        if n >= LGBM_THRESHOLD:
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=5,
                    verbose=-1,
                )
                model.fit(X, y)
                self._model = model
                self._n_samples = n
                log.info("CryptoModel retrained with LightGBM: %d samples (up=%d, down=%d)", n, n_up, n_down)
            except ImportError:
                self._train_logistic(X, y, n, n_up, n_down)
        else:
            self._train_logistic(X, y, n, n_up, n_down)

        self._save()

    def _train_logistic(self, X, y, n, n_up, n_down):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
        ])
        pipe.fit(X, y)
        self._model = pipe
        self._n_samples = n
        log.info("CryptoModel retrained with LogisticRegression: %d samples (up=%d, down=%d)", n, n_up, n_down)

    def _save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self._model, "n_samples": self._n_samples}, f)
        log.info("Crypto model saved to %s", self.model_path)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def is_trained(self) -> bool:
        return self._model is not None


def make_prediction(
    fv: CryptoFeatureVector,
    model: CryptoModel,
    market_price_up: float,
    alpha_threshold: float = 0.03,
) -> CryptoPrediction:
    """
    Given features and the current Polymarket price for 'Up',
    compute alpha and decide direction.
    """
    p_up = model.predict(fv)
    p_down = 1.0 - p_up
    alpha_up   = p_up   - market_price_up
    alpha_down = p_down - (1.0 - market_price_up)

    if abs(alpha_up) >= abs(alpha_down):
        best_alpha = alpha_up
        direction = "UP" if alpha_up >= alpha_threshold else "NO_TRADE"
    else:
        best_alpha = alpha_down
        direction = "DOWN" if alpha_down >= alpha_threshold else "NO_TRADE"

    confidence = abs(best_alpha)

    return CryptoPrediction(
        p_up=p_up,
        p_down=p_down,
        alpha=best_alpha,
        direction=direction,
        confidence=confidence,
    )
