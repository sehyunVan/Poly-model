"""
infra/types.py — Typed ID wrappers for Polymarket identifiers.

Motivation
----------
The Polymarket API uses three distinct string types that are NOT interchangeable:

  ConditionId  — hex condition hash (e.g. "0xabc123...").  Used in CTF contract
                 calls (redeemPositions), settlement lookups, and the data API.

  ClobTokenId  — numeric string token identifier for a specific outcome token
                 (e.g. "111222333...").  Used in CLOB order placement, orderbook
                 fetch, and fill polling.  clobTokenIds[0] = UP/YES token;
                 clobTokenIds[1] = DOWN/NO token.

  MarketSlug   — human-readable derived slug (e.g. "btc-updown-5m-1773967500").
                 Used for Gamma API lookups and settlement queries.

Root cause of past bugs:
  pos.market_id stored a ConditionId but was passed where a ClobTokenId was
  expected.  "invalid" AR redemptions fired because conditionId and token_id
  strings were silently interchangeable at call sites (both are str).

Usage
-----
    from infra.types import ConditionId, ClobTokenId, MarketSlug

    def place_order(token_id: ClobTokenId, ...) -> ...:
        ...

    def redeem(condition_id: ConditionId, ...) -> ...:
        ...

NewType is zero runtime cost — no wrapping, no isinstance checks needed.
Type errors are caught by mypy / Pyright, not at runtime.

If you pass a ClobTokenId where a ConditionId is expected, mypy will flag it.
Existing str values can be cast: ConditionId("0xabc...")
"""
from __future__ import annotations

from typing import NewType

# A Polymarket condition identifier — hex string prefixed with "0x".
# Used in: CTF redeemPositions(), data API /positions?user=..., settlement.
ConditionId = NewType("ConditionId", str)

# A Polymarket CLOB token identifier — numeric string, no "0x" prefix.
# Used in: CLOB /book, order placement, fill polling.
# clobTokenIds[0] = UP/YES token, clobTokenIds[1] = DOWN/NO token.
ClobTokenId = NewType("ClobTokenId", str)

# A human-readable market slug derived from symbol + timestamp.
# Format: "{symbol}-updown-5m-{window_start_ts}" for crypto markets.
# Used in: Gamma API /markets?slug=..., settlement by slug.
MarketSlug = NewType("MarketSlug", str)
