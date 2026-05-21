#!/usr/bin/env python3
"""
Filter audit — uses correct field names from each file.
  crypto_cache.jsonl : features.{score, clob_fill, ob_imbalance, ...}, prediction, label
  signal_log.jsonl   : {score, clob_ask, window_elapsed, executed, skip_reason, settled, won}
"""
import json
from collections import defaultdict
from datetime import datetime, timezone

ROOT = "/home/ubuntu/poly-model"

# ── Load crypto_cache ─────────────────────────────────────────────────────────
cache = []
with open(ROOT + "/data/crypto_cache.jsonl") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("label") is None:
            continue
        feats = row.get("features", {})
        cf = feats.get("clob_fill")
        if cf is None:
            continue
        tw = 1 if ((row["prediction"] == "UP") == (row["label"] == 1)) else 0
        ts = datetime.fromisoformat(row["ts"]).astimezone(timezone.utc)
        cache.append({
            "symbol": row.get("symbol", "BTC"),
            "score":  abs(feats.get("score", 0.0)),
            "fill":   cf,
            "win":    tw,
            "hour":   ts.hour,
        })

# ── Load signal_log ───────────────────────────────────────────────────────────
siglog = []
with open(ROOT + "/data/signal_log.jsonl") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if not row.get("settled"):
            continue
        siglog.append({
            "symbol":   row.get("symbol", "BTC"),
            "score":    abs(row.get("score", 0.0)),
            "fill":     row.get("clob_ask"),
            "win":      int(row.get("won", False)),
            "executed": row.get("executed", False),
            "skip":     row.get("skip_reason", "executed") if not row.get("executed") else "executed",
            "elapsed":  row.get("window_elapsed"),
        })


def ev(win, fill, stake=5.0):
    if fill is None or fill <= 0 or fill >= 1:
        return None
    net = stake * (1.0 - fill) / fill - stake * 0.02
    return win * net - (1.0 - win) * stake


def row_summary(trades, label, stake=5.0):
    if not trades:
        print("  %-42s  n=  0" % label)
        return
    n    = len(trades)
    wr   = sum(t["win"] for t in trades) / n
    evs  = [ev(t["win"], t["fill"], stake) for t in trades]
    evs  = [e for e in evs if e is not None]
    fills = [t["fill"] for t in trades if t.get("fill")]
    avg_ev = sum(evs) / len(evs) if evs else float("nan")
    avg_f  = sum(fills) / len(fills) if fills else 0.0
    be     = avg_f
    ev_ok  = "✅" if avg_ev > 0 else "❌"
    print("  %-42s  n=%3d  WR=%5.1f%%  BE=%5.1f%%  EV=$%+.3f  %s" % (
        label, n, wr * 100, be * 100, avg_ev, ev_ok))


print("=" * 75)
print("FILTER AUDIT — crypto_cache.jsonl (all executed labeled fills)")
print("=" * 75)

# 1. By symbol
print("\n[1] ALL EXECUTED BY SYMBOL")
for sym in ["BTC", "ETH"]:
    row_summary([c for c in cache if c["symbol"] == sym], sym)

# 2. CLOB band — BTC only, all history
print("\n[2] BTC — ALL EXECUTED BY FILL BAND")
btc = [c for c in cache if c["symbol"] == "BTC"]
bands = [(0.60,0.68),(0.68,0.72),(0.72,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.0)]
for lo, hi in bands:
    t = [c for c in btc if lo <= c["fill"] < hi]
    active = " ← ACTIVE" if lo >= 0.72 and hi <= 0.80 else (" ← BLOCKED (score cap zone)" if lo >= 0.65 else " ← BLOCKED")
    row_summary(t, "[%.2f-%.2f)%s" % (lo, hi, active))

# 3. Score range — BTC in-band [0.72-0.80]
print("\n[3] BTC in-band [0.72-0.80] — BY SIGNAL SCORE")
inband = [c for c in btc if 0.72 <= c["fill"] < 0.80]
for lo, hi in [(0.25,0.30),(0.30,0.35),(0.35,0.40),(0.40,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,1.0)]:
    active = " ← ACTIVE" if hi <= 0.65 else " ← BLOCKED (score cap)"
    row_summary([c for c in inband if lo <= c["score"] < hi], "score [%.2f-%.2f)%s" % (lo, hi, active))

# 4. Hour of day — BTC in-band
print("\n[4] BTC in-band [0.72-0.80] — BY HOUR UTC (EV sorted)")
by_hour = defaultdict(list)
for c in inband:
    by_hour[c["hour"]].append(c)
stats = []
for h, ts in sorted(by_hour.items()):
    if len(ts) < 5:
        continue
    wr = sum(t["win"] for t in ts) / len(ts)
    evs = [ev(t["win"], t["fill"]) for t in ts]
    evs = [e for e in evs if e is not None]
    stats.append((h, len(ts), wr, sum(evs)/len(evs) if evs else 0))
stats.sort(key=lambda x: x[3])
print("  Bottom 6 hours:")
for h, n, wr, avg_ev in stats[:6]:
    print("    UTC %02d:00  n=%3d  WR=%.1f%%  EV=$%+.4f" % (h, n, wr*100, avg_ev))
print("  Top 6 hours:")
for h, n, wr, avg_ev in stats[-6:]:
    print("    UTC %02d:00  n=%3d  WR=%.1f%%  EV=$%+.4f" % (h, n, wr*100, avg_ev))

print("\n" + "=" * 75)
print("SIGNAL LOG — what's fired + blocked + their actual WR")
print("=" * 75)

# 5. Skip reason breakdown
print("\n[5] SKIP REASON BREAKDOWN (signal_log, BTC, settled)")
btc_sl = [s for s in siglog if s["symbol"] == "BTC"]
by_skip = defaultdict(list)
for s in btc_sl:
    by_skip[s["skip"]].append(s)
for reason, trades in sorted(by_skip.items(), key=lambda x: -len(x[1])):
    row_summary(trades, reason)

# 6. Signal log band breakdown for skipped signals
print("\n[6] SKIPPED SIGNALS — FILL DISTRIBUTION + WR")
skipped_sl = [s for s in btc_sl if not s["executed"] and s.get("fill")]
for lo, hi in [(0.50,0.68),(0.68,0.72),(0.72,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.95)]:
    t = [s for s in skipped_sl if s["fill"] and lo <= s["fill"] < hi]
    row_summary(t, "skipped fill [%.2f-%.2f)" % (lo, hi))

# 7. Signal log window_elapsed for executed vs skipped
print("\n[7] ENTRY WINDOW ELAPSED — signal_log (BTC in-band [0.72-0.80])")
inband_sl = [s for s in btc_sl if s.get("fill") and 0.72 <= s["fill"] < 0.80]
for lo, hi in [(0,100),(100,150),(150,200),(200,230),(230,270),(270,300),(300,999)]:
    t = [s for s in inband_sl if s.get("elapsed") and lo <= s["elapsed"] < hi]
    active = " ← ACTIVE" if 200 <= lo and hi <= 270 else ""
    row_summary(t, "[%d-%d)s%s" % (lo, hi, active))

# 8. Score cap analysis on signal_log
print("\n[8] SCORE CAP — what gets blocked at |score| >= 0.65 (BTC)")
cap_blocked = [s for s in btc_sl if s["score"] >= 0.65 and s.get("fill") and 0.72 <= s["fill"] < 0.80]
cap_passed  = [s for s in btc_sl if s["score"] <  0.65 and s.get("fill") and 0.72 <= s["fill"] < 0.80]
row_summary(cap_blocked, "|score| >= 0.65  BLOCKED")
row_summary(cap_passed,  "|score| <  0.65  ACTIVE")

print()
