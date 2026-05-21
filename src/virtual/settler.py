"""
Virtual position settler.

settle_resolved_positions() closes virtual positions for markets that have
a known outcome.  Outcomes are looked up in two steps:

  1. Feature cache (Parquet) — fast, works for markets already in cache.
  2. Polymarket CLOB API     — direct lookup by market_id for any positions
                               not covered by the cache.

PnL formulas (prediction-market convention):
    YES bet at fill_price p, size s:
        YES wins  → pnl = s * (1 - p) / p
        NO  wins  → pnl = -s

    NO bet at fill_price p (price of YES token), size s:
        NO  wins  → pnl = s * p / (1 - p)
        YES wins  → pnl = -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from virtual.portfolio import VirtualPortfolio, VirtualPosition  # type: ignore

_DEFAULT_CACHE_DIR = Path("data/features_cache")
_CLOB_HOST = os.getenv("CLOB_HOST", "https://clob.polymarket.com")


# ── Outcome helpers ────────────────────────────────────────────────────────────

def _load_outcomes_from_cache(cache_dir: Path) -> dict[str, int]:
    """
    Scan all Parquet files in cache_dir and return a mapping of
    market_id → outcome (1 or 0).  Only includes rows with outcome in {0, 1}.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}

    outcomes: dict[str, int] = {}
    if not cache_dir.exists():
        return outcomes

    for parquet_file in sorted(cache_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_file, columns=["market_id", "outcome"])
            for _, row in df.iterrows():
                mid = str(row["market_id"])
                try:
                    oc = int(row["outcome"])
                    if oc in (0, 1):
                        outcomes[mid] = oc
                except (ValueError, TypeError):
                    continue
        except Exception:
            continue

    return outcomes


def _fetch_outcomes_from_api(market_ids: list[str]) -> dict[str, int]:
    """
    Query the Polymarket CLOB API for each market_id and return a mapping of
    market_id → outcome (1=YES / 0=NO) for any market that is closed and
    has a declared winner.

    Markets still open or not yet resolved are omitted from the result.
    """
    try:
        import httpx
    except ImportError:
        return {}

    outcomes: dict[str, int] = {}

    with httpx.Client(timeout=15.0) as client:
        for mid in market_ids:
            try:
                r = client.get(f"{_CLOB_HOST}/markets/{mid}")
                if r.status_code != 200:
                    continue
                data = r.json()

                # Must be closed to have a definitive outcome
                if data.get("active", True) and not data.get("closed", False):
                    continue

                # Find the winning token
                tokens = data.get("tokens", [])
                for token in tokens:
                    if token.get("winner", False):
                        outcome_str = str(token.get("outcome", "")).strip().upper()
                        if outcome_str == "YES":
                            outcomes[mid] = 1
                        elif outcome_str == "NO":
                            outcomes[mid] = 0
                        break
            except Exception:
                continue

    return outcomes


# ── PnL calculation ────────────────────────────────────────────────────────────

def _compute_pnl(position: VirtualPosition, outcome: int) -> float:
    """
    Compute realized PnL for a single position given its outcome.
    Clamps to realistic bounds to avoid infinite values at prices near 0/1.
    """
    s = position.size_usdc
    p = max(0.001, min(0.999, position.fill_price))   # YES token fill price

    if position.direction == "YES":
        if outcome == 1:
            return round(s * (1.0 - p) / p, 6)
        else:
            return round(-s, 6)
    else:  # "NO"
        if outcome == 0:
            return round(s * p / (1.0 - p), 6)
        else:
            return round(-s, 6)


# ── Main settler ───────────────────────────────────────────────────────────────

def settle_resolved_positions(
    vp: VirtualPortfolio,
    cache_dir: Union[str, Path] = _DEFAULT_CACHE_DIR,
) -> dict:
    """
    Close virtual positions that have a known outcome.

    Lookup order:
      1. Feature cache (fast, covers markets built through _process_market)
      2. Polymarket CLOB API (covers all markets including those not in cache)

    For each settled position:
    - realized_pnl is computed and written
    - available_usdc is updated: += (size_usdc + realized_pnl)
    - position is moved from vp.positions to vp.closed_positions

    Returns:
        {"settled": int, "still_open": int}
    """
    if not vp.positions:
        return {"settled": 0, "still_open": 0}

    cache_dir = Path(cache_dir)

    # Step 1: outcomes from feature cache
    outcomes = _load_outcomes_from_cache(cache_dir)

    # Step 2: for positions not covered by cache, ask the API directly
    missing_ids = [
        pos.market_id for pos in vp.positions
        if pos.market_id not in outcomes
    ]
    if missing_ids:
        api_outcomes = _fetch_outcomes_from_api(list(set(missing_ids)))
        outcomes.update(api_outcomes)

    if not outcomes:
        return {"settled": 0, "still_open": len(vp.positions)}

    still_open: list[VirtualPosition] = []
    settled_count = 0

    for pos in vp.positions:
        outcome = outcomes.get(pos.market_id)
        if outcome is None:
            still_open.append(pos)
            continue

        pnl = _compute_pnl(pos, outcome)
        pos.outcome      = outcome
        pos.realized_pnl = pnl

        # Return cost basis + PnL to available balance
        vp.available_usdc += pos.size_usdc + pnl
        vp.available_usdc  = round(vp.available_usdc, 6)

        vp.closed_positions.append(pos)
        settled_count += 1

    vp.positions = still_open
    vp.mark_updated()

    return {"settled": settled_count, "still_open": len(still_open)}
