"""Compare swarm performance before vs after 2026-06-02 01:03 UTC (trigger-window gate deployed)."""
import json
from datetime import datetime, timezone

PATH = "/home/ubuntu/poly-model/data/swarm_real_trades.jsonl"
GATE_DEPLOY = datetime(2026, 6, 2, 1, 3, tzinfo=timezone.utc)

rows = []
with open(PATH) as f:
    for line in f:
        rows.append(json.loads(line))

def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

settled = [r for r in rows if r.get("settled")]
# Window: 14d before deploy vs since deploy (both relax-era so apples-to-apples)
from datetime import timedelta
before_window_start = GATE_DEPLOY - timedelta(days=14)
pre  = [r for r in settled if before_window_start <= parse(r["entry_ts"]) < GATE_DEPLOY]
post = [r for r in settled if parse(r["entry_ts"]) >= GATE_DEPLOY]

def stats(name, ts):
    if not ts: print(f"{name}: 0"); return
    wins = [t for t in ts if t["pnl"] > 0]
    pnl = sum(t["pnl"] for t in ts)
    bet = sum(t["bet"] for t in ts)
    n_days = max(1, (parse(ts[-1]["entry_ts"]) - parse(ts[0]["entry_ts"])).total_seconds() / 86400)
    print(f"{name:24s} n={len(ts):4d}  WR={len(wins)/len(ts)*100:5.1f}%  pnl={pnl:+8.2f}  "
          f"bet={bet:7.1f}  ROI={pnl/bet*100:+5.1f}%  pnl/day={pnl/n_days:+6.2f}  trades/day={len(ts)/n_days:5.1f}")

print(f"Gate deployed: {GATE_DEPLOY}")
print(f"Comparing 14d before vs since deploy (both in relax-era to control for the 2026-05-14 gate changes)\n")
stats("PRE-gate  (14d window)", pre)
stats("POST-gate (since)",      post)

print("\nBet size distribution:")
for label, ts in [("PRE", pre), ("POST", post)]:
    if not ts: continue
    bets = sorted([t["bet"] for t in ts])
    print(f"  {label:5s}  n={len(ts):3d}  bet min={bets[0]:.1f} med={bets[len(bets)//2]:.1f} max={bets[-1]:.1f}")

# Daily trade count: did the gate reduce volume as expected?
print("\nLast 7d daily:")
import collections
recent = [r for r in settled if parse(r["entry_ts"]) > datetime.now(tz=timezone.utc) - timedelta(days=7)]
by_day = collections.defaultdict(list)
for t in recent:
    by_day[parse(t["entry_ts"]).date()].append(t)
for d in sorted(by_day):
    ts = by_day[d]
    wins = [t for t in ts if t["pnl"] > 0]
    pnl = sum(t["pnl"] for t in ts)
    print(f"  {d}  n={len(ts):3d}  WR={len(wins)/len(ts)*100:5.1f}%  pnl={pnl:+7.2f}")
