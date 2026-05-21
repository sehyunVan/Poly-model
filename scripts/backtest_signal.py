"""
scripts/backtest_signal.py

Validates whether the Polymarket flow signal predicts actual BTC/ETH price
moves on Hyperliquid (spot perpetuals).

For each labeled row in data/crypto_cache.jsonl:
  1. Extract window_start_ts from the slug field.
  2. Fetch the Hyperliquid 5-min candle covering that window.
  3. Determine: did BTC/ETH actually move in the signal's predicted direction?
  4. Aggregate win rates and PnL by score bucket.
  5. Recommend TP/SL percentages based on observed move distribution.

Key question answered:
  "When Polymarket crowd score > 0.35 for UP, does BTC actually go up?"
  If yes → HL perp trading amplifies our edge.
  If no  → Polymarket edge is prediction-market-specific (crowd psychology),
            not underlying price prediction.

Outputs:
  Console table of results by score bucket
  data/backtest_results.json  (machine-readable)

Usage:
    python scripts/backtest_signal.py
    python scripts/backtest_signal.py --min-score 0.30 --output data/bt_custom.json
    python scripts/backtest_signal.py --cache data/crypto_cache.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import httpx

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT = Path(__file__).resolve()
_ROOT   = _SCRIPT.parent.parent
_SRC    = _ROOT / "src"
sys.path.insert(0, str(_SRC))

_DEFAULT_CACHE  = _ROOT / "data" / "crypto_cache.jsonl"
_DEFAULT_OUTPUT = _ROOT / "data" / "backtest_results.json"

_HL_BASE   = "https://api.hyperliquid.xyz/info"
_HL_CLIENT = httpx.Client(timeout=10.0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cache_rows(path: Path, min_score: float = 0.0) -> list[dict]:
    """
    Load crypto_cache.jsonl and return only rows that:
      - have label (0 or 1) — i.e. settled markets
      - have abs(score) >= min_score
      - have a slug we can parse for window_start_ts
    """
    rows   = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label") is None:
                continue
            score = abs(float(r.get("features", {}).get("score", 0)))
            if score < min_score:
                skipped += 1
                continue
            slug = r.get("slug", "")
            if not _parse_window_ts(slug):
                skipped += 1
                continue
            rows.append(r)
    print(f"Loaded {len(rows)} labeled rows  ({skipped} skipped by min_score/missing slug)")
    return rows


def _parse_window_ts(slug: str) -> Optional[int]:
    """
    Extract window start timestamp from slug.
    'btc-updown-5m-1773967500' → 1773967500
    Returns None if pattern not found.
    """
    m = re.search(r"(\d{10})", slug)
    return int(m.group(1)) if m else None


# ── Hyperliquid candle fetch ───────────────────────────────────────────────────

def _hl_post(body: dict) -> object:
    r = _HL_CLIENT.post(_HL_BASE, json=body)
    r.raise_for_status()
    return r.json()


def fetch_hl_candle(
    coin: str,
    window_start_ts: int,  # Unix timestamp (seconds)
) -> Optional[dict]:
    """
    Fetch the Hyperliquid 5-min candle that covers window_start_ts.

    Returns dict with keys: open, high, low, close, volume, open_time_ms.
    Returns None if the candle is not available (market not yet settled when
    the cache row was written, or API error).

    IMPORTANT: window_start_ts is in seconds; HL API expects milliseconds.
    We request [start_ms, start_ms + 300_000) and take the first candle.
    """
    start_ms = window_start_ts * 1000
    end_ms   = start_ms + 300_000

    try:
        data = _hl_post({
            "type": "candleSnapshot",
            "req": {
                "coin":      coin,
                "interval":  "5m",
                "startTime": start_ms,
                "endTime":   end_ms,
            },
        })
        if not data:
            return None
        # Take candle whose open_time is closest to start_ms
        candle = min(data, key=lambda c: abs(int(c["t"]) - start_ms))
        return {
            "open_time_ms": int(candle["t"]),
            "open":         float(candle["o"]),
            "high":         float(candle["h"]),
            "low":          float(candle["l"]),
            "close":        float(candle["c"]),
            "volume":       float(candle["v"]),
        }
    except Exception:
        return None


# ── Outcome computation ────────────────────────────────────────────────────────

def compute_outcome(row: dict, candle: dict) -> dict:
    """
    Combine cache row + HL candle into a single analysis record.

    Returns dict with:
        symbol          : "BTC" | "ETH"
        prediction      : "UP" | "DOWN"
        poly_label      : 1 (UP won) | 0 (DOWN won)
        poly_correct    : bool — Polymarket prediction matched Polymarket outcome
        hl_return_pct   : float — BTC/ETH 5-min return (close/open - 1)
        hl_direction    : "UP" | "DOWN" — actual move on HL
        hl_correct      : bool — Polymarket signal direction matched HL move
        score           : float — absolute signal score
        score_bucket    : str  — e.g. "0.25-0.30"
        clob_fill       : float | None
    """
    features   = row.get("features", {})
    score      = float(features.get("score", 0))
    prediction = row.get("prediction", "UP")
    poly_label = int(row.get("label", 0))

    poly_correct = (
        (prediction == "UP"   and poly_label == 1) or
        (prediction == "DOWN" and poly_label == 0)
    )

    hl_return  = (candle["close"] / candle["open"]) - 1.0 if candle["open"] != 0 else 0.0
    hl_dir     = "UP" if hl_return > 0 else "DOWN"
    hl_correct = (prediction == hl_dir)

    abs_score = abs(score)
    # Bucket width 0.05 from 0.20 to 0.60+
    lo = round((abs_score // 0.05) * 0.05, 2)
    hi = round(lo + 0.05, 2)
    bucket = f"{lo:.2f}-{hi:.2f}"

    return {
        "symbol":        row.get("symbol", "BTC"),
        "prediction":    prediction,
        "poly_label":    poly_label,
        "poly_correct":  poly_correct,
        "hl_return_pct": round(hl_return * 100, 4),
        "hl_direction":  hl_dir,
        "hl_correct":    hl_correct,
        "score":         round(score, 4),
        "abs_score":     round(abs_score, 4),
        "score_bucket":  bucket,
        "clob_fill":     features.get("clob_fill"),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def analyze_buckets(outcomes: list[dict]) -> list[dict]:
    """
    Group outcomes by score_bucket and compute statistics.
    Returns list of bucket dicts sorted by score bucket.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        buckets[o["score_bucket"]].append(o)

    results = []
    for bucket, items in sorted(buckets.items()):
        n = len(items)
        if n < 3:
            continue

        poly_wins = sum(1 for o in items if o["poly_correct"])
        hl_wins   = sum(1 for o in items if o["hl_correct"])

        # When signal is correct on Polymarket AND on HL
        both_correct = sum(
            1 for o in items if o["poly_correct"] and o["hl_correct"]
        )

        # HL returns when signal was correct
        correct_returns  = sorted([abs(o["hl_return_pct"]) for o in items if o["hl_correct"]])
        wrong_returns    = sorted([abs(o["hl_return_pct"]) for o in items if not o["hl_correct"]])

        # TP recommendation: 50th percentile of correct moves
        rec_tp = percentile(correct_returns, 0.50) / 100.0

        # SL recommendation: 25th percentile of incorrect moves (how far wrong moves go)
        rec_sl = percentile(wrong_returns, 0.25) / 100.0 if wrong_returns else rec_tp * 0.5

        # Mean HL return (signed, when correct)
        mean_correct_ret = (
            sum(abs(o["hl_return_pct"]) for o in items if o["hl_correct"]) / max(hl_wins, 1)
        )
        mean_wrong_ret = (
            sum(abs(o["hl_return_pct"]) for o in items if not o["hl_correct"]) / max(n - hl_wins, 1)
        )

        # Simple EV at recommended TP/SL
        hl_wr   = hl_wins / n
        ev_trade = hl_wr * rec_tp - (1 - hl_wr) * rec_sl

        results.append({
            "bucket":           bucket,
            "n":                n,
            "poly_win_rate":    round(poly_wins / n * 100, 1),
            "hl_win_rate":      round(hl_wr * 100, 1),
            "both_correct_pct": round(both_correct / n * 100, 1),
            "mean_correct_ret": round(mean_correct_ret, 3),
            "mean_wrong_ret":   round(mean_wrong_ret, 3),
            "rec_tp_pct":       round(rec_tp * 100, 3),
            "rec_sl_pct":       round(rec_sl * 100, 3),
            "ev_per_trade_pct": round(ev_trade * 100, 3),
        })

    return results


# ── Alignment check ───────────────────────────────────────────────────────────

def compute_alignment(outcomes: list[dict]) -> dict:
    """
    Overall: how often does Polymarket outcome agree with HL move?
    This answers: "Is Polymarket crowd signal tracking BTC/ETH direction?"
    """
    n       = len(outcomes)
    aligned = sum(1 for o in outcomes if o["poly_label"] == (1 if o["hl_direction"] == "UP" else 0))
    poly_wr = sum(1 for o in outcomes if o["poly_correct"]) / max(n, 1)
    hl_wr   = sum(1 for o in outcomes if o["hl_correct"])  / max(n, 1)

    return {
        "total_rows":           n,
        "poly_to_hl_alignment": round(aligned / max(n, 1) * 100, 1),
        "polymarket_win_rate":  round(poly_wr * 100, 1),
        "hl_win_rate_overall":  round(hl_wr * 100, 1),
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_table(buckets: list[dict], alignment: dict) -> None:
    print("\n" + "=" * 95)
    print(f"  BACKTEST RESULTS — Polymarket Signal vs Hyperliquid 5-min BTC/ETH moves")
    print("=" * 95)
    print(
        f"  {'Score':12s} {'N':>5s} {'Poly WR':>8s} {'HL WR':>7s} "
        f"{'Both%':>7s} {'Avg Win':>8s} {'Avg Loss':>9s} "
        f"{'Rec TP':>7s} {'Rec SL':>7s} {'EV/trade':>9s}"
    )
    print("-" * 95)

    for b in buckets:
        ev_str = f"+{b['ev_per_trade_pct']:.2f}%" if b['ev_per_trade_pct'] >= 0 else f"{b['ev_per_trade_pct']:.2f}%"
        print(
            f"  {b['bucket']:12s} {b['n']:>5d} {b['poly_win_rate']:>7.1f}% {b['hl_win_rate']:>6.1f}% "
            f"{b['both_correct_pct']:>6.1f}% {b['mean_correct_ret']:>7.3f}% "
            f"{b['mean_wrong_ret']:>8.3f}% "
            f"{b['rec_tp_pct']:>6.3f}% {b['rec_sl_pct']:>6.3f}% {ev_str:>9s}"
        )

    print("=" * 95)
    print(f"\n  OVERALL (all scores >= filter):")
    print(f"  Total rows            : {alignment['total_rows']}")
    print(f"  Poly outcome == HL dir: {alignment['poly_to_hl_alignment']}%  "
          f"(how often Poly crowd tracks actual BTC move)")
    print(f"  Polymarket win rate   : {alignment['polymarket_win_rate']}%")
    print(f"  HL win rate (overall) : {alignment['hl_win_rate_overall']}%")
    print()

    # Interpretation
    al = alignment["poly_to_hl_alignment"]
    if al >= 60:
        print(f"  VERDICT: Strong alignment ({al}%) — Polymarket signal predicts real BTC moves.")
        print(f"  HL perp trading is LIKELY VIABLE with confirmed TP/SL levels above.")
    elif al >= 52:
        print(f"  VERDICT: Moderate alignment ({al}%) — some edge, but HL trades need tight TP/SL.")
        print(f"  Consider paper trading HL loop before going live.")
    else:
        print(f"  VERDICT: Weak alignment ({al}%) — Polymarket edge is prediction-market-specific.")
        print(f"  The crowd signal reflects market psychology, not underlying BTC price direction.")
        print(f"  HL perp trading with this signal is NOT recommended based on current data.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cache_path  = Path(args.cache)
    output_path = Path(args.output)
    min_score   = float(args.min_score)

    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        sys.exit(1)

    print(f"\nLoading cache: {cache_path}")
    rows = load_cache_rows(cache_path, min_score=min_score)

    if len(rows) < 10:
        print(f"ERROR: fewer than 10 labeled rows with score >= {min_score}. "
              f"Gather more real trade data first.")
        sys.exit(1)

    print(f"Fetching Hyperliquid 5-min candles for {len(rows)} rows...")
    print("(Rate-limited: 5 req/s with 0.2s delay between requests)")

    outcomes: list[dict] = []
    missing   = 0
    fetch_errors = 0

    for i, row in enumerate(rows):
        slug    = row.get("slug", "")
        symbol  = row.get("symbol", "BTC")
        coin    = symbol   # HL uses "BTC", "ETH" directly
        wts     = _parse_window_ts(slug)

        if wts is None:
            missing += 1
            continue

        candle = fetch_hl_candle(coin, wts)
        if candle is None:
            fetch_errors += 1
            if fetch_errors <= 5:
                print(f"  No HL candle for {coin} window {wts} (slug: {slug})")
            time.sleep(0.2)
            continue

        outcome = compute_outcome(row, candle)
        outcomes.append(outcome)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(rows)} rows...")

        time.sleep(0.2)   # rate limiting

    print(f"\nOutcomes computed: {len(outcomes)}")
    print(f"Missing window ts: {missing} | Candle fetch errors: {fetch_errors}")

    if len(outcomes) < 10:
        print("ERROR: not enough outcomes to analyze. "
              "Check HL API connectivity and cache data.")
        sys.exit(1)

    # ── Analysis ──────────────────────────────────────────────────────────────
    buckets   = analyze_buckets(outcomes)
    alignment = compute_alignment(outcomes)

    print_table(buckets, alignment)

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "alignment":     alignment,
        "buckets":       buckets,
        "raw_outcomes":  outcomes[:500],   # cap raw rows to keep file small
        "config": {
            "min_score":  min_score,
            "cache_path": str(cache_path),
            "n_rows":     len(rows),
            "n_outcomes": len(outcomes),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved: {output_path}")

    # ── HL params recommendation ───────────────────────────────────────────────
    # Find the best bucket (highest EV with n >= 20)
    good_buckets = [b for b in buckets if b["n"] >= 20 and b["ev_per_trade_pct"] > 0]
    if good_buckets:
        best = max(good_buckets, key=lambda b: b["ev_per_trade_pct"])
        print(f"\n  RECOMMENDED hl_params.yaml settings (based on bucket {best['bucket']}):")
        print(f"    signal_threshold_for_hl: {float(best['bucket'].split('-')[0]):.2f}")
        print(f"    tp_pct: {best['rec_tp_pct'] / 100:.4f}   # {best['rec_tp_pct']:.3f}%")
        print(f"    sl_pct: {best['rec_sl_pct'] / 100:.4f}   # {best['rec_sl_pct']:.3f}%")
        print(f"    (EV per trade: {best['ev_per_trade_pct']:+.2f}%  HL WR: {best['hl_win_rate']:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Polymarket signal vs Hyperliquid BTC/ETH moves")
    parser.add_argument("--cache",     default=str(_DEFAULT_CACHE),  help="crypto_cache.jsonl path")
    parser.add_argument("--output",    default=str(_DEFAULT_OUTPUT), help="output JSON path")
    parser.add_argument("--min-score", default=0.25, type=float,     help="min abs(score) to include")
    args = parser.parse_args()
    main(args)
