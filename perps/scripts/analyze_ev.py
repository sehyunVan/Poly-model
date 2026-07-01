"""
perps/scripts/analyze_ev.py — EV breakdown for Phase-0 paper trades.

Reads perps_paper.jsonl, pairs opens/closes by trade_id, prints bucketed EV
analysis that mirrors the structure of the polymarket EV audits.

Bucket axes:
  - direction        (LONG / SHORT / ALL)
  - score magnitude  ([0.15, 0.20), [0.20, 0.30), [0.30, 0.50), [0.50, 1.00])
  - utc_hour         (00..23)
  - signal component dominance  (which component had the largest |contribution|?)
  - realised volatility during hold  ((exit_fill - entry_fill) / entry_fill in %)

For each bucket prints: n, WR, avg_pnl, total_pnl, breakeven status.
A bucket is "live" if avg_pnl > 0; "dead" if avg_pnl < 0; flagged "tiny" if n<10.

Usage:
    python perps/scripts/analyze_ev.py
    python perps/scripts/analyze_ev.py --log perps/data/perps_paper.jsonl
    python perps/scripts/analyze_ev.py --csv > trades.csv         # per-trade dump
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def load_trades(log_path: Path) -> list[dict]:
    """Pair open/close events by trade_id. Returns a list of closed-trade dicts."""
    opens: dict[str, dict] = {}
    closes: list[dict] = []
    with log_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["event"] == "open":
                opens[row["trade_id"]] = row
            elif row["event"] == "close":
                open_row = opens.get(row["trade_id"], {})
                merged = {
                    "trade_id":     row["trade_id"],
                    "coin":         row["coin"],
                    "side":         row["side"],
                    "open_ts":      open_row.get("ts", row.get("entry_ts")),
                    "close_ts":     row["ts"],
                    "entry_fill":   row["entry_fill"],
                    "exit_fill":    row["exit_fill"],
                    "qty":          row["qty"],
                    "notional":     row["notional_usd"],
                    "hours_held":   row["hours_held"],
                    "price_pnl":    row["price_pnl"],
                    "fee_total":    row["fee_total"],
                    "funding_pnl":  row["funding_pnl"],
                    "net_pnl":      row["net_pnl"],
                    "score":        row["score"],
                    "components":   row["components"],
                }
                # realised price move during the hold, in basis points, signed by direction
                sign = +1 if row["side"] == "LONG" else -1
                if row["entry_fill"] > 0:
                    merged["realised_bps"] = (
                        (row["exit_fill"] - row["entry_fill"]) / row["entry_fill"] * 1e4 * sign
                    )
                else:
                    merged["realised_bps"] = 0.0
                closes.append(merged)
    return closes


def bucket_stats(label: str, trades: list[dict]) -> dict:
    if not trades:
        return {"label": label, "n": 0, "wins": 0, "wr": None, "avg_pnl": None, "total_pnl": 0.0}
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    total = sum(t["net_pnl"] for t in trades)
    return {
        "label":     label,
        "n":         len(trades),
        "wins":      wins,
        "wr":        wins / len(trades),
        "avg_pnl":   total / len(trades),
        "total_pnl": total,
        "avg_realised_bps": sum(t["realised_bps"] for t in trades) / len(trades),
    }


def fmt_row(s: dict) -> str:
    if s["n"] == 0:
        return f"  {s['label']:<28s}  n=  0  —"
    flag = "tiny" if s["n"] < 10 else ("DEAD" if s["avg_pnl"] < 0 else " EV+")
    return (
        f"  {s['label']:<28s}  n={s['n']:>3d}  WR={s['wr']*100:5.1f}%  "
        f"avg_pnl={s['avg_pnl']:+.4f}  tot={s['total_pnl']:+7.3f}  "
        f"avg_move={s['avg_realised_bps']:+6.1f}bps  {flag}"
    )


def score_bucket(score: float) -> str:
    a = abs(score)
    if a < 0.20:  return "[0.15, 0.20)"
    if a < 0.30:  return "[0.20, 0.30)"
    if a < 0.50:  return "[0.30, 0.50)"
    return         "[0.50, 1.00]"


def dominant_component(comps: dict) -> str:
    """Which component had the largest |contribution|? (component value, not weight*value)"""
    # `comps` is {"drift": float, "ob": float, "cvd": float, "lag": float, "funding": float, "mom30": float, "used": int}
    keys = ("drift", "ob", "cvd", "lag", "funding", "mom30")
    best = max(keys, key=lambda k: abs(comps.get(k, 0.0)))
    return best


def realised_vol_bucket(bps: float) -> str:
    a = abs(bps)
    if a <  5:  return "|move|<5bps   (flat)"
    if a < 15:  return "|move| 5-15bps"
    if a < 30:  return "|move| 15-30bps"
    return         "|move| 30+bps"


def analyze(trades: list[dict]) -> None:
    if not trades:
        print("No closed trades in log.")
        return

    # Header
    first_ts = min(t["open_ts"] for t in trades)
    last_ts  = max(t["close_ts"] for t in trades)
    print("=" * 78)
    print(f"PERPS PHASE-0 EV ANALYSIS — n={len(trades)}")
    print(f"window: {datetime.fromtimestamp(first_ts, tz=timezone.utc)} -> "
          f"{datetime.fromtimestamp(last_ts, tz=timezone.utc)}")
    duration_hours = (last_ts - first_ts) / 3600.0
    print(f"duration: {duration_hours:.1f}h   trades/h: {len(trades) / duration_hours:.2f}")
    print("=" * 78)

    # Overall + by direction
    print("\n── OVERALL ──")
    print(fmt_row(bucket_stats("ALL", trades)))
    for side in ("LONG", "SHORT"):
        print(fmt_row(bucket_stats(side, [t for t in trades if t["side"] == side])))

    # Fee + funding decomposition
    total_price  = sum(t["price_pnl"] for t in trades)
    total_fee    = sum(t["fee_total"] for t in trades)
    total_fund   = sum(t["funding_pnl"] for t in trades)
    total_net    = sum(t["net_pnl"] for t in trades)
    print(f"\n  PnL decomposition (sum across all trades):")
    print(f"    price PnL    : {total_price:+8.3f}")
    print(f"    fees         : {-total_fee:+8.3f}  ({total_fee / len(trades):.4f}/trade)")
    print(f"    funding PnL  : {total_fund:+8.3f}")
    print(f"    net PnL      : {total_net:+8.3f}  ({total_net / len(trades):.4f}/trade)")

    # Score magnitude buckets, sub-split by direction
    print("\n── BY SCORE MAGNITUDE ──")
    score_groups = defaultdict(list)
    for t in trades:
        score_groups[score_bucket(t["score"])].append(t)
    for bucket in ("[0.15, 0.20)", "[0.20, 0.30)", "[0.30, 0.50)", "[0.50, 1.00]"):
        ts = score_groups.get(bucket, [])
        print(fmt_row(bucket_stats(bucket, ts)))

    # UTC hour
    print("\n── BY UTC HOUR ──")
    hour_groups: dict[int, list] = defaultdict(list)
    for t in trades:
        h = datetime.fromtimestamp(t["open_ts"], tz=timezone.utc).hour
        hour_groups[h].append(t)
    for h in range(24):
        ts = hour_groups.get(h, [])
        if ts:
            print(fmt_row(bucket_stats(f"UTC {h:02d}:00", ts)))

    # Dominant signal component
    print("\n── BY DOMINANT SIGNAL COMPONENT ──")
    comp_groups: dict[str, list] = defaultdict(list)
    for t in trades:
        comp_groups[dominant_component(t["components"])].append(t)
    for c in ("drift", "ob", "cvd", "lag", "funding", "mom30"):
        ts = comp_groups.get(c, [])
        print(fmt_row(bucket_stats(f"dominant={c}", ts)))

    # Realised move during hold
    print("\n── BY REALISED MOVE (signed by side) ──")
    vol_groups: dict[str, list] = defaultdict(list)
    for t in trades:
        vol_groups[realised_vol_bucket(t["realised_bps"])].append(t)
    for b in ("|move|<5bps   (flat)", "|move| 5-15bps",
              "|move| 15-30bps",      "|move| 30+bps"):
        ts = vol_groups.get(b, [])
        print(fmt_row(bucket_stats(b, ts)))

    # Breakeven math
    print("\n── BREAKEVEN MATH ──")
    avg_fee_per_trade = total_fee / len(trades)
    avg_notional = sum(t["notional"] for t in trades) / len(trades)
    fee_bps_per_trade = (avg_fee_per_trade / avg_notional) * 1e4
    print(f"  avg notional: ${avg_notional:.2f}    avg fee/trade: ${avg_fee_per_trade:.4f}  ({fee_bps_per_trade:.1f} bps)")
    print(f"  breakeven: avg realised move (sign-correct) must exceed {fee_bps_per_trade:.1f} bps to clear fees alone")
    actual_avg_bps = sum(t["realised_bps"] for t in trades) / len(trades)
    print(f"  actual:    {actual_avg_bps:+.1f} bps  ({'OK' if actual_avg_bps > fee_bps_per_trade else 'INSUFFICIENT'})")


def dump_csv(trades: list[dict]) -> None:
    import csv
    w = csv.writer(sys.stdout)
    w.writerow([
        "trade_id", "coin", "side", "open_iso", "close_iso", "score",
        "entry_fill", "exit_fill", "realised_bps",
        "price_pnl", "fee_total", "funding_pnl", "net_pnl",
        "drift", "ob", "cvd", "lag", "funding", "mom30",
    ])
    for t in trades:
        c = t["components"]
        w.writerow([
            t["trade_id"], t["coin"], t["side"],
            datetime.fromtimestamp(t["open_ts"], tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(t["close_ts"], tz=timezone.utc).isoformat(),
            f"{t['score']:.4f}", f"{t['entry_fill']:.2f}", f"{t['exit_fill']:.2f}",
            f"{t['realised_bps']:.2f}",
            f"{t['price_pnl']:.4f}", f"{t['fee_total']:.4f}",
            f"{t['funding_pnl']:.4f}", f"{t['net_pnl']:.4f}",
            c.get("drift", 0), c.get("ob", 0), c.get("cvd", 0),
            c.get("lag", 0), c.get("funding", 0), c.get("mom30", 0),
        ])


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(here / "data" / "perps_paper.jsonl"),
                    help="path to perps_paper.jsonl")
    ap.add_argument("--csv", action="store_true",
                    help="dump per-trade CSV to stdout instead of analysis")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"No log at {log_path}", file=sys.stderr)
        return 1

    trades = load_trades(log_path)
    if not trades:
        print(f"No closed trades in {log_path}", file=sys.stderr)
        return 0

    if args.csv:
        dump_csv(trades)
    else:
        analyze(trades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
