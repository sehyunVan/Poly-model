"""
Polymarket MM reconnaissance.

Goal: measure the actual microstructure on a set of representative markets
before deciding whether MM is viable for us.

Picks ~8 active markets across categories (sports / politics / event /
crypto-5m), polls the CLOB order book every 5 seconds for N minutes, then
prints summary stats: top-of-book spread distribution, depth, mid-price
volatility, BBO turnover rate (a proxy for adverse-selection danger).

Usage on server:
    cd ~/poly-model && source .venv/bin/activate \
        && python /tmp/mm_recon.py --duration 1800 --interval 5

Outputs:
    Streams progress every 30 sec. Final summary printed to stdout.
    Raw snapshots written to /tmp/mm_recon_data.jsonl for re-analysis.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK = "https://clob.polymarket.com/book"
DATA_OUT  = Path("/tmp/mm_recon_data.jsonl")


def pick_markets(target_count: int = 8) -> list[dict]:
    """Pick markets that match what the swarm actually trades: active sports
    and event markets closing soon (next 24h) with meaningful volume AND
    where the price is in the swarm's playable band (yes_price 0.10–0.40
    so NO ask is 0.60–0.90 — the relevant zone for our actual edge)."""
    out: list[dict] = []
    # Closing in next 24h, active, has volume — what swarm targets
    from datetime import datetime, timezone, timedelta
    now = datetime.now(tz=timezone.utc)
    end_max = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    url = (
        f"{GAMMA_API}?active=true&closed=false"
        f"&end_date_max={end_max}"
        f"&volume_num_min=5000&limit=200"
        f"&order=volumeNum&ascending=false"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    mkts = resp.json()
    if not mkts:
        return []
    seen_events = set()
    for m in mkts:
        slug = m.get("eventSlug") or m.get("slug", "")
        if slug in seen_events:
            continue
        ctids = m.get("clobTokenIds")
        if not ctids:
            continue
        if isinstance(ctids, str):
            try:
                ctids = json.loads(ctids)
            except Exception:
                continue
        if not ctids or len(ctids) < 2:
            continue
        # Filter to swarm-playable band: yes_price in 0.10-0.50
        yes_price = float(m.get("lastTradePrice") or m.get("outcomePrices") or [0])[0] if isinstance(m.get("outcomePrices"), list) else None
        if yes_price is None:
            # fallback: try the field directly
            try:
                yes_price = float(m.get("yesPrice", 0)) or float(m.get("lastTradePrice", 0.5))
            except Exception:
                yes_price = 0.5
        if not (0.05 <= yes_price <= 0.55):
            continue
        seen_events.add(slug)
        out.append({
            "id":         str(m.get("id", "")),
            "question":   (m.get("question") or "")[:80],
            "slug":       slug,
            "yes_token":  str(ctids[0]),
            "no_token":   str(ctids[1]),
            "volume":     float(m.get("volumeNum", 0)),
            "end_date":   m.get("endDate", ""),
            "yes_price":  yes_price,
        })
        if len(out) >= target_count:
            break
    return out


def fetch_book(token_id: str) -> dict:
    """Single REST book fetch. Returns {} on failure."""
    try:
        r = requests.get(CLOB_BOOK, params={"token_id": token_id}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_err": str(exc)}


def parse_book(book: dict) -> dict | None:
    """Extract top-of-book + depth from a CLOB book response."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return None
    # bids descending, asks ascending in the API; but CLOB sometimes returns asks descending.
    try:
        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)
    except Exception:
        return None
    bid_at_top = sum(float(b["size"]) for b in bids if abs(float(b["price"]) - best_bid) < 1e-9)
    ask_at_top = sum(float(a["size"]) for a in asks if abs(float(a["price"]) - best_ask) < 1e-9)
    # 5-level depth as USD notional (size × price)
    depth_5_bid = sum(float(b["size"]) * float(b["price"]) for b in bids[:5])
    depth_5_ask = sum(float(a["size"]) * float(a["price"]) for a in asks[:5])
    return {
        "best_bid":   best_bid,
        "best_ask":   best_ask,
        "spread":     best_ask - best_bid,
        "mid":        (best_bid + best_ask) / 2,
        "size_bid":   bid_at_top,
        "size_ask":   ask_at_top,
        "depth_5b":   depth_5_bid,
        "depth_5a":   depth_5_ask,
        "n_bids":     len(bids),
        "n_asks":     len(asks),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=1800, help="seconds to run (default 1800 = 30 min)")
    ap.add_argument("--interval", type=int, default=5,    help="seconds between polls per market")
    ap.add_argument("--markets",  type=int, default=8,    help="markets to recon")
    ap.add_argument("--side",     choices=["yes", "no"], default="no", help="side to recon (default NO since swarm trades NO)")
    args = ap.parse_args()

    print(f"== Polymarket MM recon ==  start={datetime.now(timezone.utc).isoformat()}")
    print(f"  duration={args.duration}s  interval={args.interval}s  markets={args.markets}  side={args.side}")

    mkts = pick_markets(args.markets)
    if not mkts:
        print("No markets matched filter — aborting"); return 1
    print(f"\nPicked {len(mkts)} markets:")
    for m in mkts:
        print(f"  vol=${m['volume']:>10,.0f}  end={m['end_date'][:10]}  {m['question'][:60]}")

    # Each market: list of snapshots
    snaps: dict[str, list[dict]] = defaultdict(list)
    DATA_OUT.unlink(missing_ok=True)
    out_f = DATA_OUT.open("a", encoding="utf-8")

    start = time.time()
    tick = 0
    while time.time() - start < args.duration:
        tick += 1
        for m in mkts:
            token = m["no_token"] if args.side == "no" else m["yes_token"]
            book = fetch_book(token)
            if "_err" in book:
                continue
            parsed = parse_book(book)
            if parsed is None:
                continue
            now = time.time()
            row = {
                "ts":        now,
                "market_id": m["id"],
                "question":  m["question"],
                **parsed,
            }
            snaps[m["id"]].append(parsed | {"ts": now})
            out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        elapsed = int(time.time() - start)
        if tick % 6 == 0:   # every ~30s
            print(f"  [t={elapsed:>5}s]  tick={tick}  rows_per_market={[len(v) for v in snaps.values()]}")
        # Sleep what's left of the interval
        sleep_for = args.interval - ((time.time() - start) % args.interval)
        if sleep_for > 0:
            time.sleep(sleep_for)
    out_f.close()

    # ── Analysis ──────────────────────────────────────────────────────────
    print(f"\n== summary ==  end={datetime.now(timezone.utc).isoformat()}")
    print(f"  total rows captured: {sum(len(v) for v in snaps.values())}")

    print(f"\n{'market':<55s} {'n':>4s} {'spread_med':>10s} {'spread_p90':>10s} "
          f"{'mid_med':>8s} {'depth5_med':>11s} {'mid_chg/s':>10s} {'bbo_flip%':>10s}")
    print("-" * 130)
    for m in mkts:
        rows = snaps.get(m["id"], [])
        if len(rows) < 5:
            print(f"  {m['question'][:53]:<55s} n={len(rows):>3d}  (too few)")
            continue
        spreads = [r["spread"] for r in rows]
        mids    = [r["mid"] for r in rows]
        depths  = [(r["depth_5b"] + r["depth_5a"]) / 2 for r in rows]
        # mid changes per second (mid volatility proxy)
        deltas = []
        for i in range(1, len(rows)):
            dt = rows[i]["ts"] - rows[i - 1]["ts"]
            if dt > 0:
                deltas.append(abs(rows[i]["mid"] - rows[i - 1]["mid"]) / dt)
        avg_delta = statistics.mean(deltas) if deltas else 0.0
        # BBO flip rate: how often does best_bid or best_ask change between samples?
        bbo_flips = sum(
            1 for i in range(1, len(rows))
            if rows[i]["best_bid"] != rows[i - 1]["best_bid"]
            or rows[i]["best_ask"] != rows[i - 1]["best_ask"]
        )
        flip_pct = (bbo_flips / max(1, len(rows) - 1)) * 100
        print(
            f"  {m['question'][:53]:<55s} {len(rows):>4d}"
            f"  {statistics.median(spreads):>10.4f}  {sorted(spreads)[int(len(spreads)*0.9)]:>10.4f}"
            f"  {statistics.median(mids):>8.3f}  ${statistics.median(depths):>10.1f}"
            f"  {avg_delta * 1000:>9.2f}m  {flip_pct:>9.1f}%"
        )

    # Aggregate
    all_spreads = [r["spread"] for v in snaps.values() for r in v]
    all_flip_rates = []
    for v in snaps.values():
        if len(v) < 2:
            continue
        flips = sum(1 for i in range(1, len(v)) if v[i]["best_bid"] != v[i-1]["best_bid"] or v[i]["best_ask"] != v[i-1]["best_ask"])
        all_flip_rates.append((flips / (len(v) - 1)) * 100)

    if all_spreads:
        print(f"\nAGGREGATE: median spread = ${statistics.median(all_spreads):.4f} = "
              f"{statistics.median(all_spreads)*100:.2f}c   "
              f"avg BBO flip rate = {statistics.mean(all_flip_rates) if all_flip_rates else 0:.1f}%")
        print(f"Note: flip rate is per {args.interval}s sample interval.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
