"""
Swarm consensus engine — WHALE-GATED strategy.

Flow:
  1. Whale signals fetched for all candidate markets (WS buffer or REST fallback)
  2. Markets WITHOUT a whale signal are dropped immediately — no AI cost, no trade
  3. Markets WITH a whale signal: AI swarm asked "should we follow?"
  4. Score = whale_strength × ai_agreement_fraction × avg_confidence
  5. Top-N returned sorted by score

Score formula:
    whale_score  = whale.strength                        [0.30 – 1.0]
    ai_agree_frac = models_that_voted_follow / total     [0 – 1]
    avg_conf      = average confidence of agreeing models [0 – 1]
    score = whale_score × 0.50 + ai_agree_frac × avg_conf × 0.50

Gate to execute:
  - score >= EXEC_SCORE_MIN
  - ai_agree_frac >= AI_AGREE_MIN  (majority must validate)
  - whale.strength >= WHALE_STRENGTH_MIN
  - avg_conf >= AVG_CONF_MIN  (NEW: confidence gate — strongest predictor)

Data analysis (108 post-RAG trades, 2026-04-07/08):
  avg_conf >= 65: 77.8% WR (72/108, 67% coverage)
  avg_conf >= 70: 82.9% WR (41/108, 38% coverage)   ← deployed 2026-04-15
  deepseek voted: 83.9% WR (31/108, 29% coverage)   ← quality oracle signal
  NO direction:   70.9% vs YES 58.5%                 ← logged for monitoring
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from bot.config import ACTIVE_MODELS, CONSENSUS_TOP_N
from bot.data.markets import MarketData
from bot.data.whale import WhaleSignal, WhaleFeed, get_whale_signals, whale_feed
from bot.swarm.models import Decision, ModelVerdict, ask_swarm, synthesize_verdicts

log = logging.getLogger(__name__)

# ── Gate thresholds ───────────────────────────────────────────────────────────
WHALE_STRENGTH_MIN = 0.30   # minimum whale flow imbalance to consider
AI_AGREE_MIN       = 0.50   # majority of all models must vote consensus direction
EXEC_SCORE_MIN     = 0.60   # ★ relaxed 0.70→0.60 (2026-05-14): paper-only NO in [0.60–0.70) now 88.6% WR / +$83 on 79 trades; 2026-04-18 tightening superseded by post-synthesis data
AVG_CONF_MIN       = 70.0   # strongest predictor (corr=+0.38): 82.9% WR at ≥70 vs 77.8% at ≥65
DEEPSEEK_NAME      = "deepseek"  # model name for quality-oracle logging


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MarketConsensus:
    market:           MarketData
    direction:        Decision
    score:            float           # composite whale × AI score
    whale_score:      float           # pure whale component
    ai_score:         float           # pure AI component
    yes_votes:        int
    no_votes:         int
    no_trade_votes:   int
    avg_confidence:   float
    ai_agree_frac:    float           # fraction of models that voted FOLLOW
    whale:                Optional[WhaleSignal] = None
    verdicts:             list[ModelVerdict] = field(default_factory=list)
    # Second-round synthesis fields (None when synthesis model is unavailable)
    synthesis_confidence: Optional[float] = None
    synthesis_reasoning:  Optional[str] = None
    # Sizing multiplier for directional disagreements (★ added 2026-04-27)
    direction_mismatch_penalty: float = 1.0  # 0.5 if model_direction ≠ whale_direction, 1.0 otherwise

    def __repr__(self) -> str:
        return (
            f"Consensus({self.direction.value}  score={self.score:.3f}  "
            f"whale={self.whale_score:.2f}  ai={self.ai_score:.2f}  "
            f"agree={self.ai_agree_frac:.0%}  "
            f"yes={self.yes_votes} no={self.no_votes} skip={self.no_trade_votes}  "
            f"{self.market.question[:55]!r})"
        )


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_verdicts(
    verdicts: list[ModelVerdict],
    whale_direction: str,
) -> tuple[float, int, int, int, float, float, str]:
    """
    Blind vote scoring: models voted YES/NO/NO_TRADE independently.

    Steps:
      1. Find model majority direction (most YES or NO votes, ignoring NO_TRADE).
      2. Check alignment: majority must match whale_direction.
      3. agree_frac = fraction of all models that voted the majority direction.

    Returns (ai_score, yes_v, no_v, skip_v, avg_conf, agree_frac, model_direction).
    model_direction is "YES", "NO", or "NONE" (no majority / all abstained).
    """
    n = len(verdicts)
    if n == 0:
        return 0.0, 0, 0, 0, 0.0, 0.0, "NONE"

    yes_v  = [v for v in verdicts if v.decision == Decision.YES]
    no_v   = [v for v in verdicts if v.decision == Decision.NO]
    skip_v = [v for v in verdicts if v.decision == Decision.NO_TRADE]

    # Model majority direction: whichever of YES/NO got more votes
    if len(yes_v) > len(no_v):
        model_direction = "YES"
        follow_v = yes_v
    elif len(no_v) > len(yes_v):
        model_direction = "NO"
        follow_v = no_v
    else:
        # Tie or all abstained — no clear model direction
        model_direction = "NONE"
        follow_v = []

    if not follow_v:
        return 0.0, len(yes_v), len(no_v), len(skip_v), 0.0, 0.0, model_direction

    agree_frac = len(follow_v) / n
    avg_conf   = sum(v.confidence for v in follow_v) / len(follow_v) / 100
    ai_score   = agree_frac * avg_conf

    return (
        round(ai_score, 4),
        len(yes_v), len(no_v), len(skip_v),
        round(avg_conf * 100, 1),
        round(agree_frac, 3),
        model_direction,
    )


def _composite_score(whale_strength: float, ai_score: float) -> float:
    return round(whale_strength * 0.50 + ai_score * 0.50, 4)


# ── Evaluate one market ───────────────────────────────────────────────────────

async def evaluate_market(
    market: MarketData,
    whale: WhaleSignal,
) -> Optional[MarketConsensus]:
    """
    Run AI swarm to validate whale signal.
    Returns None if whale is too weak or AI doesn't agree.
    """
    if not whale.has_signal:
        log.debug("Skip %r — no whale signal (strength=%.2f)", market.question[:40], whale.strength)
        return None

    verdicts, context = await ask_swarm(market, whale)
    ai_score, yes_v, no_v, skip_v, avg_conf, agree_frac, model_direction = _score_verdicts(
        verdicts, whale.direction
    )

    # ── Blind alignment check ──────────────────────────────────────────────────
    # Models voted without seeing the whale direction. Only proceed if their
    # independent majority matches the whale — both sources agree on direction.
    if model_direction == "NONE":
        log.info(
            "No-majority SKIP %r — models split (yes=%d no=%d skip=%d), no clear direction",
            market.question[:50], yes_v, no_v, skip_v,
        )
        return None

    # ★ MODIFIED 2026-04-27: Direction-mismatch from SKIP → SIZE 50%
    # Was: hard-block when model_direction ≠ whale_direction
    # Now: allow trade but mark for reduced sizing (0.5× multiplier)
    # Reason: whale might be wrong, models might be right — test if they're profitable
    # After 50+ mismatch trades: decide if filter should stay removed or reinstated
    direction_mismatch_penalty = 1.0
    if model_direction != whale.direction:
        log.info(
            "Direction-mismatch eval: models say %s, whale says %s "
            "→ sizing at 50%% (was skip) — %s",
            model_direction, whale.direction, market.question[:50]
        )
        direction_mismatch_penalty = 0.5  # Apply 50% sizing penalty, continue

    # ── Remaining quality gates ────────────────────────────────────────────────
    if agree_frac < AI_AGREE_MIN:
        log.info(
            "Low-agree SKIP %r — AI agree=%.0f%% (yes=%d no=%d skip=%d) < %.0f%% required",
            market.question[:50], agree_frac * 100, yes_v, no_v, skip_v, AI_AGREE_MIN * 100,
        )
        return None

    # Confidence gate: strongest single predictor (corr=+0.38 on 108 post-RAG trades).
    # avg_conf >= 65: 77.8% WR vs 64.8% baseline (+13pp).
    if avg_conf < AVG_CONF_MIN:
        log.info(
            "Low-confidence SKIP %r — avg_conf=%.0f%% < %.0f%% required "
            "(yes=%d no=%d skip=%d)",
            market.question[:50], avg_conf, AVG_CONF_MIN, yes_v, no_v, skip_v,
        )
        return None

    whale_score = whale.strength
    score       = _composite_score(whale_score, ai_score)

    # ★ Score ceiling REMOVED 2026-05-14: paper-only NO with score>=0.80 = 13 trades,
    # 100% WR, +$33.81 real PnL. Historic "0% WR n=6" claim that motivated the ceiling
    # is contradicted by post-synthesis data. Keeping no-op block for revert safety.
    if score < EXEC_SCORE_MIN:
        log.info(
            "Low-score SKIP %r — score=%.3f < %.2f hard floor",
            market.question[:50], score, EXEC_SCORE_MIN,
        )
        return None

    direction = Decision.YES if whale.direction == "YES" else Decision.NO

    # ── Round-1 deepseek veto ──────────────────────────────────────────────────
    # Deepseek abstains on 87% of losses vs 63% of wins — cheap pre-filter
    # before the more expensive synthesis call.
    # Important: only veto when deepseek explicitly voted NO_TRADE (no error).
    # If deepseek had an API error (402, timeout, etc.) treat as absent — the
    # other models still provide signal. A failed call ≠ a genuine abstain.
    deepseek_v = next(
        (v for v in verdicts if v.model_name == DEEPSEEK_NAME), None
    )
    if deepseek_v is None:
        # DeepSeek not in ACTIVE_MODELS — veto gate not applicable
        deepseek_voted = True
    elif deepseek_v.error is not None:
        # API error (402 out-of-credits, 500, timeout) — don't veto, log warning
        log.warning(
            "Deepseek API error — skipping veto gate for %r: %s",
            market.question[:40], deepseek_v.error[:80],
        )
        deepseek_voted = True
    else:
        # Normal response — veto if deepseek explicitly abstained
        deepseek_voted = deepseek_v.decision != Decision.NO_TRADE

    if not deepseek_voted:
        log.info(
            "Deepseek-veto SKIP %r — deepseek abstained in round 1 (yes=%d no=%d skip=%d score=%.3f)",
            market.question[:50], yes_v, no_v, skip_v, score,
        )
        return None

    # ── Round-2 synthesis (Design A) ───────────────────────────────────────────
    # Deepseek sees ALL panel reasoning (anonymised) and produces a calibrated
    # final verdict. This is a second deepseek call with richer context than round 1.
    # Gate: if synthesis returns NO_TRADE → skip. Otherwise, synthesis_confidence
    # is stored and used by execute_pick() for bet sizing (blended with score-norm).
    # If synthesis fails/times out, proceed with original avg_confidence (round-1
    # deepseek veto already guards quality).
    # ★ DISABLED 2026-05-13: Deepseek API credit exhausted. Reverting to round-1
    # veto only (deepseek_voted gate at line 244). Synthesis provided confidence
    # calibration but round-1 veto already provides quality gating.
    # Re-enable when deepseek credits are topped up: uncomment the call below.
    synthesis = None  # await synthesize_verdicts(
        # verdicts, market, whale, context, model_direction
    # )

    synthesis_confidence: Optional[float] = None
    synthesis_reasoning:  Optional[str] = None

    if synthesis is not None:
        if synthesis.decision == Decision.NO_TRADE:
            log.info(
                "Synthesis-veto SKIP %r — deepseek synthesis NO_TRADE (conf=%d): %s",
                market.question[:50], synthesis.confidence, synthesis.reasoning,
            )
            return None
        if synthesis.decision.value != model_direction:
            log.info(
                "Synthesis-direction SKIP %r — synthesis says %s but panel majority is %s",
                market.question[:50], synthesis.decision.value, model_direction,
            )
            return None
        synthesis_confidence = float(synthesis.confidence)
        synthesis_reasoning  = synthesis.reasoning

    log.info(
        "%r  whale=%s(%.2f)  models=%s  ai_agree=%.0f%%  conf=%.0f%%  score=%.3f  "
        "synthesis=%s(%.0f)",
        market.question[:40], whale.direction, whale_score,
        model_direction, agree_frac * 100, avg_conf, score,
        synthesis.decision.value if synthesis else "n/a",
        synthesis_confidence or 0.0,
    )

    return MarketConsensus(
        market=market,
        direction=direction,
        score=score,
        whale_score=whale_score,
        ai_score=ai_score,
        yes_votes=yes_v,
        no_votes=no_v,
        no_trade_votes=skip_v,
        avg_confidence=avg_conf,
        ai_agree_frac=agree_frac,
        whale=whale,
        verdicts=verdicts,
        synthesis_confidence=synthesis_confidence,
        synthesis_reasoning=synthesis_reasoning,
        direction_mismatch_penalty=direction_mismatch_penalty,
    )


# ── Evaluate all candidates ───────────────────────────────────────────────────

async def run_swarm(
    markets: list[MarketData],
    feed: WhaleFeed = whale_feed,
) -> list[MarketConsensus]:
    """
    1. Subscribe new market tokens to the WhaleFeed
    2. Fetch whale signals for all candidates (WS buffer + REST fallback)
    3. Drop markets without a whale signal — no AI calls wasted
    4. Run AI swarm on whale-triggered markets
    5. Return top-N by composite score
    """
    if not ACTIVE_MODELS:
        log.error("No active models — add API keys to .env")
        return []
    if not markets:
        log.info("No candidate markets to evaluate.")
        return []

    # Subscribe new tokens to the live WebSocket feed
    token_ids = []
    for m in markets:
        if m.yes_token_id:
            token_ids.append(m.yes_token_id)
        if m.no_token_id:
            token_ids.append(m.no_token_id)
    if token_ids:
        feed.subscribe(token_ids)

    # Fetch whale signals (WS buffer first, REST fallback)
    whale_map = await get_whale_signals(markets, feed)

    # Filter to only markets with a whale signal
    whale_markets = [
        (m, whale_map[m.id])
        for m in markets
        if whale_map.get(m.id) and whale_map[m.id].has_signal
    ]

    log.info(
        "Swarm: %d candidates  %d have whale signals  (%d models each)",
        len(markets), len(whale_markets), len(ACTIVE_MODELS),
    )

    if not whale_markets:
        log.info("No whale signals this cycle — no AI calls made.")
        return []

    # Run AI on whale-triggered markets sequentially (rate limit protection)
    results = []
    for market, whale in whale_markets:
        log.info("Evaluating whale: %s  %s", whale.summary(), market.question[:50])
        result = await evaluate_market(market, whale)
        if result is not None:
            results.append(result)
        await asyncio.sleep(0.5)

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[:CONSENSUS_TOP_N]

    log.info(
        "Swarm done — %d passed AI gate, top %d returned",
        len(results), len(top),
    )
    return top
