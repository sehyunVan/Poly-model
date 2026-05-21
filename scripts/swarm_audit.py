#!/usr/bin/env python3
"""Swarm real-trade performance audit."""
import json
from collections import defaultdict
from datetime import datetime, timezone

ROOT = "/home/ubuntu/poly-model"

trades = []
with open(ROOT + "/data/swarm_real_trades.jsonl") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        r = json.loads(raw)
        if r.get("outcome") is None:
            continue
        r["win"] = 1 if r["outcome"] == "WIN" else 0
        r["ts"]  = datetime.fromisoformat(r["entry_ts"]).replace(tzinfo=timezone.utc)
        trades.append(r)

def ev_trade(win, ask, bet):
    net_win = bet * (1.0 - ask) / ask
    return win * net_win - (1 - win) * bet

def show(label, rows, width=46):
    if not rows:
        print("  %-*s  n=  0" % (width, label))
        return
    n   = len(rows)
    wr  = sum(r["win"] for r in rows) / n
    evs = [ev_trade(r["win"], r["ask"], r["bet"]) for r in rows]
    avg_ev  = sum(evs) / n
    avg_ask = sum(r["ask"] for r in rows) / n
    avg_bet = sum(r["bet"] for r in rows) / n
    total_pnl = sum(r.get("pnl", ev_trade(r["win"], r["ask"], r["bet"])) for r in rows)
    be = avg_ask
    flag = "✅" if avg_ev > 0 else "❌"
    print("  %-*s  n=%3d  WR=%5.1f%%  BE=%5.1f%%  EV=$%+.3f  bet=$%.2f  PnL=$%+.2f  %s" % (
        width, label, n, wr*100, be*100, avg_ev, avg_bet, total_pnl, flag))

n_total   = len(trades)
n_yes     = sum(1 for r in trades if r["direction"] == "YES")
n_no      = sum(1 for r in trades if r["direction"] == "NO")
no_trades = [r for r in trades if r["direction"] == "NO"]

print("=" * 85)
print("SWARM AUDIT — %d settled real trades (2026-04-07 to 2026-04-14)" % n_total)
print("=" * 85)

# 1. Overall
print("\n[1] OVERALL")
show("ALL settled", trades)
show("YES direction (blocked since 04-11)", [r for r in trades if r["direction"]=="YES"])
show("NO direction", no_trades)

# 2. By NO ask price bucket
print("\n[2] NO DIRECTION — BY ASK PRICE (BE = ask)")
for lo, hi in [(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.75),(0.75,0.80)]:
    t = [r for r in no_trades if lo <= r["ask"] < hi]
    active = " ← ACTIVE" if hi <= 0.75 else " ← BLOCKED (ask ceiling)"
    show("NO ask [%.2f-%.2f)%s" % (lo, hi, active), t)

# 3. By score bucket
print("\n[3] NO DIRECTION — BY CONSENSUS SCORE")
for lo, hi in [(0.65,0.68),(0.68,0.71),(0.71,0.74),(0.74,0.77),(0.77,0.80)]:
    t = [r for r in no_trades if lo <= r["score"] < hi]
    show("score [%.2f-%.2f)" % (lo, hi), t)

# 4. By yes_price (ep < 0.40 gate)
print("\n[4] NO DIRECTION — BY YES_PRICE (ep<0.40 gate)")
for lo, hi in [(0.0,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.70)]:
    t = [r for r in no_trades if r.get("yes_price") and lo <= r["yes_price"] < hi]
    active = " ← ACTIVE" if hi <= 0.40 else " ← BLOCKED (yes_price>=0.40)"
    show("yes_price [%.2f-%.2f)%s" % (lo, hi, active), t)

# 5. By bet size (variable sizing effect)
print("\n[5] NO DIRECTION — BY BET SIZE")
for lo, hi in [(0,8),(8,12),(12,17),(17,22),(22,35)]:
    t = [r for r in no_trades if lo <= r["bet"] < hi]
    show("bet [$%d-$%d)" % (lo, hi), t)

# 6. By date era
print("\n[6] BY ERA")
era1 = [r for r in trades if r["ts"] < datetime(2026,4,9, tzinfo=timezone.utc)]
era2 = [r for r in trades if datetime(2026,4,9,tzinfo=timezone.utc) <= r["ts"] < datetime(2026,4,12,tzinfo=timezone.utc)]
era3 = [r for r in trades if r["ts"] >= datetime(2026,4,12, tzinfo=timezone.utc)]
show("pre-blind (before 04-09)", era1)
show("blind-vote (04-09 to 04-11)", era2)
show("synthesis era (04-12+)", era3)

# 7. Recent NO-only era (04-11+)
print("\n[7] NO-ONLY ERA (2026-04-11+) — detailed")
no_recent = [r for r in no_trades if r["ts"] >= datetime(2026,4,11,tzinfo=timezone.utc)]
show("NO trades 04-11+", no_recent)
# By ask price in NO-only era
for lo, hi in [(0.30,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.75)]:
    t = [r for r in no_recent if lo <= r["ask"] < hi]
    show("  NO ask [%.2f-%.2f)" % (lo, hi), t)

# 8. Look at what's being skipped — count from swarm log
print("\n[8] RECENT SWARM LOG SKIP COUNTS (last 500 lines)")
import subprocess
try:
    result = subprocess.run(
        ["tail", "-n", "2000", ROOT + "/logs/swarm.log"],
        capture_output=True, text=True, timeout=10
    )
    lines = result.stdout.splitlines()
    skip_counts = defaultdict(int)
    exec_count = 0
    for line in lines:
        if "SKIP" in line.upper() or "skip" in line:
            for kw in ["Synthesis-veto","Synthesis-direction","Direction-mismatch",
                       "No-majority","score <","score >=","avg_conf","zero-dissent",
                       "SKIP YES","SKIP divergence","SKIP declined","ask >","ep<0.40",
                       "dead-zone","no whale"]:
                if kw.lower() in line.lower():
                    skip_counts[kw] += 1
                    break
            else:
                skip_counts["other_skip"] += 1
        if "FILLED" in line or "EXEC" in line or "real fill" in line.lower():
            exec_count += 1
    print("  Executions (last 2000 log lines): %d" % exec_count)
    for k, v in sorted(skip_counts.items(), key=lambda x: -x[1]):
        print("  %-35s  %d" % (k, v))
except Exception as e:
    print("  (log parse failed: %s)" % e)

print()
