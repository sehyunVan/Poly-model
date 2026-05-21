"""
Alpha computation and trade signal generation.

Core formula:
    alpha = P_R - P_M

Signal rules (checked in priority order):
    1. market close < min_seconds_to_close  → NO_TRADE  (event risk)
    2. confidence < min_confidence          → NO_TRADE  (uncertain prediction)
    3. |alpha| <= alpha_threshold           → NO_TRADE  (no edge)
    4. alpha > alpha_threshold              → BUY_YES
    5. alpha < -alpha_threshold             → BUY_NO

Size factor:
    base_size_factor = min(|alpha| / alpha_full_size, 1.0)
    Full size (1.0) is reached when |alpha| >= alpha_full_size (default 0.20).

All thresholds are loaded from config/signal_params.yaml.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from data.schemas import Market
from prediction.schemas import PredictionResult
from .schemas import TradeSignal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path("config/signal_params.yaml")


def _load_params() -> dict:
    """Load signal parameters from YAML; fall back to safe defaults."""
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "alpha_threshold": 0.05,
            "min_confidence": 0.40,
            "min_seconds_to_close": 3600,
            "alpha_full_size": 0.20,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_alpha(P_R: float, P_M: float) -> float:
    """
    Return the signed edge: P_R - P_M.

    Positive alpha  → model thinks YES is underpriced → BUY_YES edge.
    Negative alpha  → model thinks NO  is underpriced → BUY_NO  edge.

    Args:
        P_R: Model-estimated probability (0–1).
        P_M: Market-implied probability  (0–1), i.e. current YES token price.

    Returns:
        float in (-1, 1).
    """
    return P_R - P_M


def generate_signals(
    prediction: PredictionResult,
    market: Market,
    alpha_threshold: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_seconds_to_close: Optional[float] = None,
    alpha_full_size: Optional[float] = None,
) -> TradeSignal:
    """
    Convert a PredictionResult into an actionable TradeSignal.

    Priority order of NO_TRADE conditions:
        1. Market closes within min_seconds_to_close  (event-risk window)
        2. Prediction confidence below min_confidence (model too uncertain)
        3. |alpha| at or below alpha_threshold        (no exploitable edge)

    If none of the above trigger, direction is determined by the sign of alpha.

    base_size_factor scales linearly from 0 to 1 as |alpha| grows from
    alpha_threshold to alpha_full_size.  Beyond alpha_full_size the factor
    is capped at 1.0.

    Args:
        prediction:           Output from predict_probability().
        market:               Current Market metadata (for close time).
        alpha_threshold:      Override config value for tau_alpha.
        min_confidence:       Override config value.
        min_seconds_to_close: Override config value.
        alpha_full_size:      Override config value.

    Returns:
        TradeSignal with direction and base_size_factor set.

    Example:
        >>> signal = generate_signals(prediction, market)
        >>> if signal.direction != "NO_TRADE":
        ...     execute_trade(signal)
    """
    params = _load_params()

    tau        = alpha_threshold      if alpha_threshold      is not None else params["alpha_threshold"]
    min_conf   = min_confidence       if min_confidence       is not None else params["min_confidence"]
    min_secs   = min_seconds_to_close if min_seconds_to_close is not None else params["min_seconds_to_close"]
    full_size  = alpha_full_size      if alpha_full_size      is not None else params["alpha_full_size"]

    alpha = compute_alpha(prediction.P_R, prediction.P_M)
    now   = datetime.now(timezone.utc)

    def _no_trade(reason: str) -> TradeSignal:
        return TradeSignal(
            market_id=prediction.market_id,
            timestamp=now,
            direction="NO_TRADE",
            alpha=alpha,
            base_size_factor=0.0,
            reason=reason,
            P_M=prediction.P_M,
            P_R=prediction.P_R,
            confidence=prediction.confidence,
        )

    # ------------------------------------------------------------------
    # Rule 1 — event-risk window
    # ------------------------------------------------------------------
    secs_left = market.seconds_to_close
    if secs_left < min_secs:
        return _no_trade(
            f"market closes in {secs_left:.0f}s "
            f"(min {min_secs:.0f}s required)"
        )

    # ------------------------------------------------------------------
    # Rule 2 — prediction confidence gate
    # ------------------------------------------------------------------
    if prediction.confidence < min_conf:
        return _no_trade(
            f"confidence {prediction.confidence:.3f} below "
            f"minimum {min_conf:.3f}"
        )

    # ------------------------------------------------------------------
    # Rule 3 — alpha threshold (no edge)
    # ------------------------------------------------------------------
    abs_alpha = abs(alpha)
    if abs_alpha <= tau:
        return _no_trade(
            f"|alpha| {abs_alpha:.4f} at or below threshold {tau:.4f}"
        )

    # ------------------------------------------------------------------
    # Size factor — linear scale from threshold to full_size
    # ------------------------------------------------------------------
    # At |alpha| = tau       → factor = 0  (just crossed threshold)
    # At |alpha| = full_size → factor = 1
    range_above = max(full_size - tau, 1e-9)
    base_size_factor = min((abs_alpha - tau) / range_above, 1.0)

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------
    if alpha > tau:
        direction = "BUY_YES"
    else:
        direction = "BUY_NO"

    return TradeSignal(
        market_id=prediction.market_id,
        timestamp=now,
        direction=direction,
        alpha=alpha,
        base_size_factor=round(base_size_factor, 4),
        reason=(
            f"alpha={alpha:.4f} (threshold={tau:.4f}); "
            f"confidence={prediction.confidence:.3f}; "
            f"size_factor={base_size_factor:.4f}"
        ),
        P_M=prediction.P_M,
        P_R=prediction.P_R,
        confidence=prediction.confidence,
    )
