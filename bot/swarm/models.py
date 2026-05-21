"""
AI model adapters for the swarm.

Each adapter implements async predict(market) -> ModelVerdict.
Provider "anthropic" uses the Anthropic SDK directly.
Provider "openai-compat" uses the OpenAI SDK pointed at any compatible base_url
(works for OpenAI, DeepSeek, Grok/xAI, Qwen, etc.).

Adding a new provider = add one entry to ALL_MODELS in config.py.
No code changes needed here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from bot.config import ACTIVE_MODELS, MODEL_TIMEOUT_S
from bot.data.markets import MarketData
from bot.data.whale import WhaleSignal
from bot.swarm.search import fetch_market_context

log = logging.getLogger(__name__)

# Model used as the second-round synthesis judge.
# Must match a name in ACTIVE_MODELS — deepseek chosen because it has the
# highest accuracy (67.2%) and abstains on 87% of losses (quality oracle).
SYNTHESIS_MODEL_NAME = "deepseek"


# ── Verdict ───────────────────────────────────────────────────────────────────

class Decision(str, Enum):
    YES      = "YES"
    NO       = "NO"
    NO_TRADE = "NO_TRADE"


@dataclass
class ModelVerdict:
    model_name: str
    decision:   Decision
    confidence: int          # 0–100
    reasoning:  str
    error:      Optional[str] = None   # set if the call failed

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.decision != Decision.NO_TRADE


# ── Prompt ────────────────────────────────────────────────────────────────────
# Blind design: models do NOT see the whale direction or flow amounts.
# They form an independent YES/NO view from RAG + market odds alone.
# consensus.py then checks if model majority matches the whale direction —
# only executing when both agree independently.
# Whale data is used for market selection (trigger) but not shown to models.

_PROMPT = """\
You are a superforecaster analyzing a prediction market. \
Your job: use all available information to estimate the true probability, \
then decide whether to bet YES or NO.

Market: {question}

MARKET DATA:
- Current market odds: YES={yes_price:.3f} ({yes_pct:.1f}%)  NO={no_price:.3f} ({no_pct:.1f}%)
- 24h volume: ${volume_24h:,.0f}
- Closes in: {hours:.1f} hours

RECENT INFORMATION (web search results):
{context}

TASK — Think step by step:
1. What does the recent information say about the likely outcome?
2. What is your best estimate of the true probability of YES?
3. Is there enough edge vs the current market odds to bet?

BREAKEVEN:
- Betting YES requires true P(YES) > {yes_pct:.0f}% to be profitable
- Betting NO requires true P(YES) < {no_pct:.0f}% to be profitable

CALIBRATION: If evidence clearly supports one side, commit to it. \
Only use NO_TRADE when evidence is absent or genuinely ambiguous.

Respond in EXACTLY this format (no other text):
DECISION: YES|NO|NO_TRADE
CONFIDENCE: <integer 0-100>
REASONING: <one sentence citing the key evidence>"""


_ABOVE_BELOW_PATTERNS_M = frozenset(["above $", "below $", "above or below $"])
_CRYPTO_ASSETS_M = frozenset(["btc", "bitcoin", "eth", "ethereum", "sol", "solana", "bnb", "xrp"])


def _is_above_below_question(question: str) -> bool:
    q = question.lower()
    return (any(p in q for p in _ABOVE_BELOW_PATTERNS_M)
            and any(a in q for a in _CRYPTO_ASSETS_M))


def _build_prompt(market: MarketData, whale: "WhaleSignal", context: str) -> str:
    """Build the blind superforecaster prompt with RAG context.

    Whale direction is intentionally excluded so models form an independent
    view. consensus.py checks alignment with the whale after all votes are in.
    """
    context_block = context.strip() if context.strip() else "(no search results found)"

    if _is_above_below_question(market.question):
        context_block = (
            "[PRICE-LEVEL MARKET: This asks whether a crypto asset will reach a specific "
            "dollar target. Key factors: current spot price vs strike level, time to expiry, "
            "recent trend/momentum. YES = price reaches/exceeds strike; NO = it does not.]\n\n"
            + context_block
        )

    return _PROMPT.format(
        question=market.question,
        yes_price=market.yes_price,
        yes_pct=market.yes_price * 100,
        no_price=market.no_price,
        no_pct=market.no_price * 100,
        volume_24h=market.volume_24h,
        hours=market.hours_to_close,
        context=context_block,
    )


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(text: str, model_name: str, whale_direction: str) -> tuple[Decision, int, str]:
    """
    Parse model response. Models now vote YES/NO/NO_TRADE directly.
    whale_direction is unused here — alignment check happens in consensus.py.
    """
    decision   = Decision.NO_TRADE
    confidence = 0
    reasoning  = ""

    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("YES"):
                decision = Decision.YES
            elif val.startswith("NO_TRADE") or "NO TRADE" in val:
                decision = Decision.NO_TRADE
            elif val.startswith("NO"):
                decision = Decision.NO
            # legacy FOLLOW still accepted — map to whale direction
            elif "FOLLOW" in val:
                decision = Decision.YES if whale_direction == "YES" else Decision.NO
        elif line.upper().startswith("CONFIDENCE:"):
            m = re.search(r"\d+", line)
            if m:
                confidence = min(100, max(0, int(m.group())))
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return decision, confidence, reasoning


# ── Anthropic adapter ─────────────────────────────────────────────────────────

async def _call_anthropic(cfg: dict, prompt: str) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=cfg["api_key"])
    msg = await client.messages.create(
        model=cfg["model"],
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── OpenAI-compatible adapter ─────────────────────────────────────────────────

async def _call_openai_compat(cfg: dict, prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
    )
    resp = await client.chat.completions.create(
        model=cfg["model"],
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ── Dispatcher ────────────────────────────────────────────────────────────────

async def _call_model(cfg: dict, prompt: str) -> str:
    provider = cfg["provider"]
    if provider == "anthropic":
        return await _call_anthropic(cfg, prompt)
    elif provider == "openai-compat":
        return await _call_openai_compat(cfg, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Public: run one model on one market ──────────────────────────────────────

async def ask_model(cfg: dict, market: MarketData, whale: "WhaleSignal", context: str) -> ModelVerdict:
    """Call one model and return its verdict. Never raises — errors go into verdict.error."""
    name   = cfg["name"]
    prompt = _build_prompt(market, whale, context)
    try:
        raw = await asyncio.wait_for(_call_model(cfg, prompt), timeout=MODEL_TIMEOUT_S)
        decision, confidence, reasoning = _parse_response(raw, name, whale.direction)
        log.debug("%s → %s (%d) %s", name, decision.value, confidence, reasoning[:60])
        return ModelVerdict(
            model_name=name,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
        )
    except asyncio.TimeoutError:
        log.warning("%s timed out on market %r", name, market.question[:40])
        return ModelVerdict(name, Decision.NO_TRADE, 0, "", error="timeout")
    except Exception as e:
        log.warning("%s failed on market %r: %s", name, market.question[:40], e)
        return ModelVerdict(name, Decision.NO_TRADE, 0, "", error=str(e))


# ── Public: run all active models on one market ───────────────────────────────

async def ask_swarm(
    market: MarketData, whale: "WhaleSignal"
) -> tuple[list[ModelVerdict], str]:
    """
    Fetch RAG context once, then run all ACTIVE_MODELS concurrently on a single market.
    Context is shared across all models — one search per market, not one per model.
    Returns (verdicts, context) — context is threaded back to the caller so the
    synthesis round can reuse it without a second search.
    """
    context = fetch_market_context(market.question, market.hours_to_close)
    if context:
        log.info("RAG context fetched for %r (%d chars)", market.question[:50], len(context))
    else:
        log.info("RAG: no context found for %r", market.question[:50])

    tasks = [ask_model(cfg, market, whale, context) for cfg in ACTIVE_MODELS]
    verdicts = list(await asyncio.gather(*tasks))
    return verdicts, context


# ── Synthesis prompt ──────────────────────────────────────────────────────────
# Round-2 synthesis: deepseek sees all panel reasoning (anonymised — no model names,
# no vote counts) and outputs a calibrated YES/NO/NO_TRADE with confidence.
# Design goals:
#   • Show decision + confidence + reasoning per analyst so deepseek can weigh arguments
#   • Do NOT reveal how many analysts voted each way — prevent pure-majority anchoring
#   • Ask deepseek to specifically look for valid minority/abstain counterarguments
#   • Confidence here is used for bet sizing, not as a gate (gate is NO_TRADE check)

_SYNTHESIS_PROMPT = """\
You are a senior analyst providing a final synthesis of a prediction market panel.

Market: {question}
Odds: YES={yes_pct:.0f}%  NO={no_pct:.0f}%
Closes in: {hours:.1f}h

PANEL ANALYSES (anonymised — focus on arguments, not who said what):
{analyst_block}

BACKGROUND CONTEXT:
{context_snippet}

SYNTHESIS TASK — The panel majority voted {majority_direction}. Your job:
1. Weigh the strongest argument on each side
2. Ask: does the minority or abstaining view raise a valid concern the majority ignored?
3. Is there enough evidence for {majority_direction} to profit vs the \
{bet_side_pct:.0f}% market price?

Rules:
- High confidence (≥75): majority reasoning is clearly supported, minority \
counterarguments are weak or already addressed
- Medium confidence (55-74): majority is probably right but uncertainty remains
- Low confidence (<55): genuine ambiguity — prefer NO_TRADE
- NO_TRADE: evidence is missing, contradictory, or the edge over {bet_side_pct:.0f}% \
is unclear

Respond in EXACTLY this format:
DECISION: YES|NO|NO_TRADE
CONFIDENCE: <integer 0-100>
SYNTHESIS: <one sentence — the single most decisive factor>"""


def _build_synthesis_prompt(
    market: MarketData,
    verdicts: list[ModelVerdict],
    context: str,
    majority_direction: str,
) -> str:
    lines = []
    for v in verdicts:
        reasoning = v.reasoning.strip() if v.reasoning else "no reasoning provided"
        if v.decision == Decision.NO_TRADE:
            lines.append(f"- ABSTAIN: {reasoning}")
        else:
            lines.append(f"- {v.decision.value} (conf={v.confidence}): {reasoning}")
    analyst_block = "\n".join(lines)

    context_snippet = context[:600].strip() if context.strip() else "(no web context available)"
    bet_side_pct = (
        market.yes_price * 100 if majority_direction == "YES" else market.no_price * 100
    )

    return _SYNTHESIS_PROMPT.format(
        question=market.question,
        yes_pct=market.yes_price * 100,
        no_pct=market.no_price * 100,
        hours=market.hours_to_close,
        analyst_block=analyst_block,
        context_snippet=context_snippet,
        majority_direction=majority_direction,
        bet_side_pct=bet_side_pct,
    )


async def synthesize_verdicts(
    verdicts: list[ModelVerdict],
    market: MarketData,
    whale: "WhaleSignal",
    context: str,
    majority_direction: str,
) -> Optional[ModelVerdict]:
    """
    Second-round synthesis: deepseek sees all panel reasoning and produces a
    calibrated final verdict with confidence.

    Returns None if the synthesis model is unavailable or times out — callers
    should proceed with the original avg_confidence in that case rather than
    blocking the trade (round-1 deepseek veto already guards quality).
    """
    synth_cfg = next(
        (m for m in ACTIVE_MODELS if m["name"] == SYNTHESIS_MODEL_NAME), None
    )
    if synth_cfg is None:
        log.debug("Synthesis model %r not in ACTIVE_MODELS — skipping synthesis", SYNTHESIS_MODEL_NAME)
        return None

    prompt = _build_synthesis_prompt(market, verdicts, context, majority_direction)

    try:
        raw = await asyncio.wait_for(
            _call_model(synth_cfg, prompt), timeout=MODEL_TIMEOUT_S
        )
        decision, confidence, reasoning = _parse_response(raw, SYNTHESIS_MODEL_NAME, whale.direction)
        log.info(
            "Synthesis %s conf=%d: %s",
            decision.value, confidence, reasoning[:80],
        )
        return ModelVerdict(
            model_name=f"{SYNTHESIS_MODEL_NAME}-synthesis",
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
        )
    except asyncio.TimeoutError:
        log.warning("Synthesis timed out for %r", market.question[:40])
        return None
    except Exception as e:
        log.warning("Synthesis failed for %r: %s", market.question[:40], e)
        return None
