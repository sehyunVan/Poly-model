"""
src/betfair/signal.py — Swarm AI signal for Betfair football MATCH_ODDS markets.

Pivoted from horse racing 2026-06-30. Tests whether the price-orthogonal swarm edge
proven on Polymarket (component B: the LLM reads news better than the crowd) transfers
to news-driven football outcomes.

Each model votes HOME / DRAW / AWAY / NO_TRADE *blind* — it sees the RAG team-news
context and the market's implied probabilities, but NOT the other models' votes. The
caller backs the plurality outcome if the agreement + confidence gates pass.

The vote `margin` (plurality minus runner-up vote count) is logged per bet: on
Polymarket it was the strongest price-orthogonal predictor of win rate, so we capture
it here to test the same hypothesis on football.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)

MODEL_TIMEOUT_S = 20  # seconds per LLM call


# ── Verdict types ─────────────────────────────────────────────────────────────

class MatchOutcome(str, Enum):
    HOME     = "HOME"
    DRAW     = "DRAW"
    AWAY     = "AWAY"
    NO_TRADE = "NO_TRADE"


@dataclass
class MatchVerdict:
    model_name: str
    outcome:    MatchOutcome
    confidence: int            # 0–100
    reasoning:  str
    error:      Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.outcome != MatchOutcome.NO_TRADE


@dataclass
class MatchConsensus:
    outcome:    Optional[MatchOutcome]   # None = no actionable consensus
    agree_frac: float
    avg_conf:   float
    votes:      dict                     # {"HOME":n,"DRAW":n,"AWAY":n,"NO_TRADE":n}
    margin:     int                      # plurality_votes - runner_up_votes
    n_models:   int


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT = """\
You are a professional football (soccer) analyst and superforecaster.

MATCH: {home} (home) vs {away} (away)
COMPETITION: {competition}
Kickoff in ~{mins:.0f} minutes.

Current Betfair market odds (decimal) and implied probabilities:
  {home} win: {oh:.2f}  ({ph:.0f}%)
  Draw:       {od:.2f}  ({pd:.0f}%)
  {away} win: {oa:.2f}  ({pa:.0f}%)
(Implied probabilities sum to >100% because of the bookmaker margin.)

RECENT TEAM NEWS (web search — confirmed lineups, injuries, suspensions, form, motivation):
{context}

TASK — think step by step:
1. Using the team news (especially CONFIRMED lineups and key absences), estimate the TRUE
   probability of each outcome.
2. Does any SINGLE outcome look underpriced — your true probability meaningfully ABOVE the
   market's implied probability, by enough to beat Betfair's 5% commission on winnings?
3. If no outcome offers a clear edge, vote NO_TRADE. Be selective — most matches are fairly
   priced, and backing a fair price loses to commission over time.

Respond in EXACTLY this format:
PICK: HOME|DRAW|AWAY|NO_TRADE
CONFIDENCE: <integer 0-100>
REASONING: <one sentence citing the key evidence>"""


def _build_prompt(home, away, competition, oh, od, oa, mins, context) -> str:
    ph = 100.0 / max(oh, 1.01)
    pd = 100.0 / max(od, 1.01)
    pa = 100.0 / max(oa, 1.01)
    ctx = context.strip() if context and context.strip() else "(no search results found)"
    return _PROMPT.format(
        home=home, away=away, competition=competition,
        oh=oh, od=od, oa=oa, ph=ph, pd=pd, pa=pa, mins=mins, context=ctx,
    )


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[MatchOutcome, int, str]:
    outcome = MatchOutcome.NO_TRADE
    confidence = 50
    reasoning = ""
    for line in text.splitlines():
        s = line.strip()
        u = s.upper()
        if u.startswith("PICK:"):
            v = u.split(":", 1)[1].strip()
            if "HOME" in v:
                outcome = MatchOutcome.HOME
            elif "AWAY" in v:
                outcome = MatchOutcome.AWAY
            elif "DRAW" in v:
                outcome = MatchOutcome.DRAW
            else:
                outcome = MatchOutcome.NO_TRADE
        elif u.startswith("CONFIDENCE:"):
            m = re.search(r"\d+", s.split(":", 1)[1])
            if m:
                confidence = max(0, min(100, int(m.group())))
        elif u.startswith("REASONING:"):
            reasoning = s.split(":", 1)[1].strip()
    return outcome, confidence, reasoning


# ── Per-model LLM call ────────────────────────────────────────────────────────

async def _call_model(model: dict, prompt: str) -> MatchVerdict:
    name = model["name"]
    provider = model.get("provider", "openai-compat")
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=model["api_key"])
            msg = await asyncio.wait_for(
                client.messages.create(
                    model=model["model"],
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=MODEL_TIMEOUT_S,
            )
            text = msg.content[0].text
        else:  # openai-compat
            import openai
            client = openai.AsyncOpenAI(
                api_key=model["api_key"],
                base_url=model.get("base_url"),
            )
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model["model"],
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=MODEL_TIMEOUT_S,
            )
            text = resp.choices[0].message.content or ""
        outcome, confidence, reasoning = _parse_response(text)
        return MatchVerdict(name, outcome, confidence, reasoning)
    except Exception as e:
        log.warning("Model %s error: %s", name, e)
        return MatchVerdict(name, MatchOutcome.NO_TRADE, 0, "", str(e))


# ── Public API ────────────────────────────────────────────────────────────────

async def ask_match_swarm(
    home: str,
    away: str,
    competition: str,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    mins_to_ko: float,
    rag_context: str = "",
) -> list[MatchVerdict]:
    """Call all active models in parallel; each votes HOME/DRAW/AWAY/NO_TRADE blind."""
    from bot.config import ACTIVE_MODELS  # reuse the existing swarm model roster

    prompt = _build_prompt(home, away, competition, odds_home, odds_draw, odds_away,
                           mins_to_ko, rag_context)
    verdicts = await asyncio.gather(*[_call_model(m, prompt) for m in ACTIVE_MODELS])
    log.debug(
        "Match swarm %s v %s — %s",
        home, away,
        [(v.model_name, v.outcome.value, v.confidence) for v in verdicts],
    )
    return list(verdicts)


def compute_consensus(
    verdicts: list[MatchVerdict],
    min_agree_frac: float = 0.55,
    min_confidence: int = 60,
) -> MatchConsensus:
    """
    Pick the plurality directional outcome among non-erroring models and apply gates.
    Returns MatchConsensus with outcome=None when there is no actionable consensus.
    """
    valid = [v for v in verdicts if v.error is None]
    votes = {"HOME": 0, "DRAW": 0, "AWAY": 0, "NO_TRADE": 0}
    for v in valid:
        votes[v.outcome.value] += 1

    n = len(valid)
    if n == 0:
        return MatchConsensus(None, 0.0, 0.0, votes, 0, 0)

    directional = [("HOME", votes["HOME"]), ("DRAW", votes["DRAW"]), ("AWAY", votes["AWAY"])]
    directional.sort(key=lambda x: -x[1])
    top_name, top_n = directional[0]
    runner_up_n = directional[1][1]
    margin = top_n - runner_up_n

    if top_n == 0:
        return MatchConsensus(None, 0.0, 0.0, votes, 0, n)   # everyone abstained
    if margin == 0:
        return MatchConsensus(None, top_n / n, 0.0, votes, 0, n)  # directional tie

    outcome = MatchOutcome(top_name)
    group = [v for v in valid if v.outcome == outcome]
    agree_frac = len(group) / n
    avg_conf = sum(v.confidence for v in group) / len(group)

    if agree_frac < min_agree_frac or avg_conf < min_confidence:
        return MatchConsensus(None, agree_frac, avg_conf, votes, margin, n)

    return MatchConsensus(outcome, agree_frac, avg_conf, votes, margin, n)
