"""
Limitless × Polymarket ETH 15-min cross-venue arb scanner (logging mode).

Runs headless on the server. Every time it gets fresh quotes from both venues
it writes a snapshot to data/limitless_arb_log.jsonl and prints a summary line.

Both venues resolve against the same Chainlink ETH/USD stream, so
  YES_LIM ask + DOWN_POLY ask < 1.00  →  riskless locked profit

No credentials required — read-only public data.

Usage:
    python src/limitless/arb_scanner.py
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import socketio
import websockets

LOG_PATH = Path("data/limitless_arb_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [LIM_ARB] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("lim_arb")

LIMITLESS_API = "https://api.limitless.exchange"
LIMITLESS_WSS = "https://ws.limitless.exchange"
POLY_GAMMA    = "https://gamma-api.polymarket.com/markets"
POLY_CLOB     = "https://clob.polymarket.com/book"
POLY_WSS      = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# cost threshold to log as "arb opportunity" (covers ~1% fee per venue)
ARB_THRESHOLD = 0.98
ROTATE_GRACE  = 5   # seconds past expiry before rediscovering
REDISCOVER_RETRY = 3


@dataclass
class VenueBook:
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid:  float | None = None
    no_ask:  float | None = None
    last_update: float = 0.0


@dataclass
class State:
    lim:  VenueBook = field(default_factory=VenueBook)
    poly: VenueBook = field(default_factory=VenueBook)
    eth_price:     float | None = None
    strike:        float | None = None
    market_end_ts: int   | None = None
    lim_slug:  str = ""
    poly_slug: str = ""
    title:     str = ""
    snapshots: int = 0
    arb_hits:  int = 0


state = State()


# ── REST helpers ──────────────────────────────────────────────────────────────

async def fetch_limitless_eth_15min(client: httpx.AsyncClient) -> dict | None:
    now_ms = int(time.time() * 1000)
    candidates = []
    for page in range(1, 6):
        r = await client.get(f"{LIMITLESS_API}/markets/active", params={"page": page, "limit": 25})
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        for m in data:
            if m.get("stableSlug") == "eth-15min-price" and m.get("expirationTimestamp", 0) > now_ms:
                candidates.append(m)
    if not candidates:
        return None
    candidates.sort(key=lambda m: m["expirationTimestamp"])
    return candidates[0]


async def fetch_polymarket_eth_15min(client: httpx.AsyncClient, market_end_ts: int) -> dict | None:
    slug = f"eth-updown-15m-{market_end_ts - 900}"
    r = await client.get(POLY_GAMMA, params={"slug": slug, "closed": "false"})
    r.raise_for_status()
    arr = r.json()
    return arr[0] if arr else None


async def fetch_lim_book(client: httpx.AsyncClient, slug: str) -> dict:
    r = await client.get(f"{LIMITLESS_API}/markets/{slug}/orderbook")
    r.raise_for_status()
    return r.json()


async def fetch_poly_book(client: httpx.AsyncClient, token_id: str) -> dict:
    r = await client.get(POLY_CLOB, params={"token_id": token_id})
    r.raise_for_status()
    return r.json()


def _best_lim(book: dict) -> tuple[float | None, float | None]:
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    return (float(bids[0]["price"]) if bids else None,
            float(asks[0]["price"]) if asks else None)


def _best_poly(book: dict) -> tuple[float | None, float | None]:
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    return (float(bids[-1]["price"]) if bids else None,
            float(asks[-1]["price"]) if asks else None)


# ── WebSocket listeners ───────────────────────────────────────────────────────

async def limitless_wss_loop(slug: str):
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    @sio.on("orderbookUpdate", namespace="/markets")
    async def on_book(payload):
        if payload.get("marketSlug") != slug:
            return
        book = payload.get("orderbook", {})
        bb, ba = _best_lim(book)
        state.lim.yes_bid, state.lim.yes_ask = bb, ba
        if bb is not None: state.lim.no_ask = round(1 - bb, 4)
        if ba is not None: state.lim.no_bid = round(1 - ba, 4)
        state.lim.last_update = time.time()
        maybe_snapshot()

    @sio.on("oraclePriceData", namespace="/markets")
    async def on_oracle(payload):
        v = payload.get("value")
        if v:
            state.eth_price = float(v)

    try:
        await sio.connect(LIMITLESS_WSS, namespaces=["/markets"],
                          transports=["websocket"], wait_timeout=15)
        await sio.emit("subscribe_market_prices", {"marketSlugs": [slug]},
                       namespace="/markets")
        await sio.wait()
    except Exception as exc:
        log.warning("Limitless WS error: %s", exc)


async def polymarket_wss_loop(yes_token: str, no_token: str):
    sub_msg = json.dumps({"assets_ids": [yes_token, no_token], "type": "market"})
    while True:
        try:
            async with websockets.connect(POLY_WSS, ping_interval=30, ping_timeout=10) as ws:
                await ws.send(sub_msg)
                async for raw in ws:
                    try:
                        events = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(events, list):
                        events = [events]
                    for ev in events:
                        if ev.get("event_type") not in ("book", "price_change"):
                            continue
                        tid = ev.get("asset_id", "")
                        bb, ba = _best_poly(ev)
                        if tid == yes_token:
                            state.poly.yes_bid, state.poly.yes_ask = bb, ba
                        elif tid == no_token:
                            state.poly.no_bid, state.poly.no_ask = bb, ba
                        state.poly.last_update = time.time()
                        maybe_snapshot()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("Poly WS reconnect: %s", exc)
            await asyncio.sleep(2)


# ── Snapshot + logging ────────────────────────────────────────────────────────

def maybe_snapshot():
    L, P = state.lim, state.poly
    # need quotes from both venues to be worth logging
    if None in (L.yes_ask, L.no_ask, P.yes_ask, P.no_ask):
        return

    now = time.time()
    # don't spam — max one snapshot per second
    if now - getattr(maybe_snapshot, "_last", 0) < 1.0:
        return
    maybe_snapshot._last = now

    # four arb pairs
    edges = {
        "YES_LIM+DOWN_POLY": round(L.yes_ask + P.no_ask, 5),
        "NO_LIM+UP_POLY":    round(L.no_ask  + P.yes_ask, 5),
        "YES+NO_LIM":        round(L.yes_ask + L.no_ask, 5),
        "UP+DOWN_POLY":      round(P.yes_ask + P.no_ask, 5),
    }
    best_pair  = min(edges, key=edges.get)
    best_total = edges[best_pair]
    best_edge  = round(1.0 - best_total, 5)

    state.snapshots += 1
    is_arb = best_total < ARB_THRESHOLD
    if is_arb:
        state.arb_hits += 1

    row = {
        "ts":            datetime.now(timezone.utc).isoformat(),
        "lim_slug":      state.lim_slug,
        "poly_slug":     state.poly_slug,
        "strike":        state.strike,
        "eth_price":     state.eth_price,
        "lim_yes_bid":   L.yes_bid, "lim_yes_ask": L.yes_ask,
        "lim_no_bid":    L.no_bid,  "lim_no_ask":  L.no_ask,
        "poly_yes_bid":  P.yes_bid, "poly_yes_ask": P.yes_ask,
        "poly_no_bid":   P.no_bid,  "poly_no_ask":  P.no_ask,
        "edges":         edges,
        "best_pair":     best_pair,
        "best_total":    best_total,
        "best_edge":     best_edge,
        "is_arb":        is_arb,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")

    # console summary line every 30 snapshots OR on any arb hit
    if is_arb or state.snapshots % 30 == 0:
        eth_str = f"ETH=${state.eth_price:,.2f}" if state.eth_price else "ETH=?"
        marker  = "  *** ARB HIT ***" if is_arb else ""
        log.info(
            "%s | strike=%.2f | %s | best: %s=%.4f (edge %+.4f) | hits=%d/total=%d%s",
            state.title[:45], state.strike or 0, eth_str,
            best_pair, best_total, best_edge,
            state.arb_hits, state.snapshots, marker,
        )


# ── Discovery + main ─────────────────────────────────────────────────────────

async def discover(client: httpx.AsyncClient) -> tuple[str | None, str | None, str | None]:
    lm = await fetch_limitless_eth_15min(client)
    if not lm:
        return None, None, None

    slug = lm["slug"]
    state.lim = VenueBook()
    state.lim_slug = slug
    state.title = lm["title"]
    state.market_end_ts = lm["expirationTimestamp"] // 1000
    try:
        state.strike = float(lm["title"].split("$")[1].split(" ")[0].replace(",", ""))
    except Exception:
        state.strike = None

    book = await fetch_lim_book(client, slug)
    bb, ba = _best_lim(book)
    state.lim.yes_bid, state.lim.yes_ask = bb, ba
    if bb is not None: state.lim.no_ask = round(1 - bb, 4)
    if ba is not None: state.lim.no_bid = round(1 - ba, 4)

    state.poly = VenueBook()
    state.poly_slug = ""
    pm = await fetch_polymarket_eth_15min(client, state.market_end_ts)
    if not pm:
        log.warning("No matching Polymarket market for end_ts=%d", state.market_end_ts)
        return slug, None, None

    tokens = pm["clobTokenIds"]
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    py_yes, py_no = tokens[0], tokens[1]
    state.poly_slug = pm["slug"]

    yb = await fetch_poly_book(client, py_yes)
    nb = await fetch_poly_book(client, py_no)
    state.poly.yes_bid, state.poly.yes_ask = _best_poly(yb)
    state.poly.no_bid,  state.poly.no_ask  = _best_poly(nb)

    return slug, py_yes, py_no


async def main():
    log.info("Starting Limitless × Polymarket ETH-15m arb scanner")
    log.info("Logging to %s | arb threshold < %.2f", LOG_PATH, ARB_THRESHOLD)

    async with httpx.AsyncClient(timeout=15) as client:
        slug, py_yes, py_no = await discover(client)
        if not slug:
            log.error("No active ETH-15m market on Limitless — exiting")
            return
        log.info("Limitless: %s", slug)
        log.info("Polymarket: %s", state.poly_slug or "no match this cycle")

        while True:
            wss_tasks = [asyncio.create_task(limitless_wss_loop(slug))]
            if py_yes and py_no:
                wss_tasks.append(asyncio.create_task(polymarket_wss_loop(py_yes, py_no)))

            while True:
                if state.market_end_ts and time.time() >= state.market_end_ts + ROTATE_GRACE:
                    break
                await asyncio.sleep(1)

            log.info("Market expired — rotating. snapshots=%d arb_hits=%d",
                     state.snapshots, state.arb_hits)
            for t in wss_tasks:
                t.cancel()
            await asyncio.gather(*wss_tasks, return_exceptions=True)

            slug = py_yes = py_no = None
            for _ in range(10):
                slug, py_yes, py_no = await discover(client)
                if slug:
                    break
                await asyncio.sleep(REDISCOVER_RETRY)
            if not slug:
                log.error("Could not find next Limitless market — exiting")
                break
            log.info("Rotated to: %s | poly: %s", slug, state.poly_slug or "no match")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped. Total snapshots=%d arb_hits=%d", state.snapshots, state.arb_hits)
