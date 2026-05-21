"""
bot/swarm/search.py — Web search for RAG context in AI swarm evaluation.

Called once per market (shared across all 5 models) before building the prompt.
Uses DuckDuckGo (free, no API key) via duckduckgo-search package.
Falls back to empty string on any failure — swarm still runs, just without context.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import date
from typing import Optional

log = logging.getLogger(__name__)

# Max snippets to include in the prompt context
_MAX_SNIPPETS = 5
# Max chars per snippet body
_SNIPPET_CHARS = 220
# Subprocess timeout per query — primp (Rust HTTP client inside ddgs) can segfault
# and would kill the main process if called in-process. Running in a subprocess
# isolates the crash: the subprocess dies, we get [], main bot survives.
_SEARCH_SUBPROCESS_TIMEOUT = 15  # seconds

# Minimal script run inside the subprocess — no imports from bot/ needed
_SUBPROCESS_SCRIPT = (
    "import sys, json; "
    "from ddgs import DDGS; "
    "r = list(DDGS().text(sys.argv[1], max_results=int(sys.argv[2]))); "
    "print(json.dumps(r))"
)


def _ddg_search(query: str, max_results: int = 4) -> list[dict]:
    """Run a DuckDuckGo text search in an isolated subprocess.

    ddgs uses the primp Rust extension which can segfault. Running it in a
    subprocess means a crash returns [] without killing the swarm bot.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, query, str(max_results)],
            capture_output=True,
            text=True,
            timeout=_SEARCH_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        if proc.returncode != 0:
            log.debug("DDG subprocess exited %d for %r", proc.returncode, query[:60])
    except subprocess.TimeoutExpired:
        log.debug("DDG subprocess timed out for %r", query[:60])
    except Exception as e:
        log.debug("DDG search failed for %r: %s", query[:60], e)
    return []


def _build_queries(question: str, hours_to_close: float) -> list[str]:
    """Generate 2-3 targeted search queries for a market question."""
    today = date.today().isoformat()
    q = question.strip()

    queries = [f"{q} {today}"]

    q_lower = q.lower()

    # Sports-specific: look for injury/lineup intel
    if any(kw in q_lower for kw in [
        "spread", "win", "score", "vs", "game", "match", "tournament",
        "championship", "series", "playoff", "final", "cup", "bowl",
    ]):
        queries.append(f"{q} injury report lineup odds")

    # Politics: polls and news
    elif any(kw in q_lower for kw in [
        "president", "prime minister", "election", "vote", "senate",
        "congress", "chancellor", "minister", "candidate", "party",
    ]):
        queries.append(f"{q} polls prediction expert")

    # Economic / financial
    elif any(kw in q_lower for kw in [
        "rate", "gdp", "inflation", "fed", "interest", "bank",
        "stock", "market", "price", "tariff", "trade",
    ]):
        queries.append(f"{q} forecast analysis")

    # Generic second query
    else:
        queries.append(f"{q} prediction analysis latest")

    return queries[:2]  # max 2 queries to keep latency low


def fetch_market_context(question: str, hours_to_close: float) -> str:
    """
    Fetch web search snippets for a market question.
    Returns formatted string of up to _MAX_SNIPPETS snippets, or "" on failure.
    Called once per market before building the AI prompt.
    """
    queries = _build_queries(question, hours_to_close)

    snippets: list[str] = []
    seen: set[str] = set()

    for query in queries:
        results = _ddg_search(query, max_results=4)
        for r in results:
            url = r.get("href", "")
            if url in seen:
                continue
            seen.add(url)
            title = r.get("title", "").strip()
            body  = r.get("body", "").strip()
            if body:
                text = f"[{title}] {body[:_SNIPPET_CHARS]}"
                snippets.append(text)
            if len(snippets) >= _MAX_SNIPPETS:
                break
        if len(snippets) >= _MAX_SNIPPETS:
            break

    if not snippets:
        return ""

    return "\n".join(f"• {s}" for s in snippets)
