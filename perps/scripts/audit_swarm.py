"""Quick polymarket swarm audit. Run on server."""
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

PATH = "/home/ubuntu/poly-model/data/swarm_real_trades.jsonl"

rows = []
with open(PATH) as f:
    for line in f:
        rows.append(json.loads(line))


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


settled   = [r for r in rows if r.get("settled")]
unsettled = [r for r in rows if not r.get("settled")]

RELAX = datetime(2026, 5, 14, tzinfo=timezone.utc)
now = datetime.now(tz=timezone.utc)

pre    = [r for r in settled if parse(r["entry_ts"]) <  RELAX]
post   = [r for r in settled if parse(r["entry_ts"]) >= RELAX]
last14 = [r for r in settled if parse(r["entry_ts"]) > now - timedelta(days=14)]
last7  = [r for r in settled if parse(r["entry_ts"]) > now - timedelta(days=7)]


def stats(name, ts):
    if not ts:
        print(f"{name}: 0"); return
    wins = [t for t in ts if t["pnl"] > 0]
    pnl  = sum(t["pnl"] for t in ts)
    bet  = sum(t["bet"] for t in ts)
    roi  = pnl / bet * 100 if bet else 0
    print(f"{name:18s} n={len(ts):4d}  WR={len(wins)/len(ts)*100:5.1f}%  pnl={pnl:+8.2f}  bet={bet:7.1f}  ROI={roi:+5.1f}%")


print(f"SWARM AUDIT (rows: {len(rows)}, settled: {len(settled)}, open: {len(unsettled)})")
if settled:
    earliest = parse(settled[0]["entry_ts"])
    latest_e = parse(rows[-1]["entry_ts"])
    print(f"Earliest entry: {earliest}")
    print(f"Latest entry:   {latest_e}")
print()
stats("ALL settled",   settled)
stats("PRE-relax",     pre)
stats("POST-relax",    post)
print()

pno  = [t for t in post if t.get("direction") == "NO"]
pyes = [t for t in post if t.get("direction") == "YES"]
stats("POST NO",  pno)
stats("POST YES", pyes)
print()
stats("LAST 14d", last14)
stats("LAST 7d",  last7)

if post:
    bets = sorted([t["bet"] for t in post])
    print(f"\nPOST-relax bet size: min={bets[0]:.2f} med={bets[len(bets)//2]:.2f} max={bets[-1]:.2f}")

def bucket(s):
    if s < 0.60: return "<0.60"
    if s < 0.70: return "[0.60-0.70)"
    if s < 0.80: return "[0.70-0.80)"
    if s < 0.90: return "[0.80-0.90)"
    return         "[0.90-1.00]"

g = defaultdict(list)
for t in post:
    g[bucket(t["score"])].append(t)
print("\nPOST-relax by score:")
for k in ["<0.60", "[0.60-0.70)", "[0.70-0.80)", "[0.80-0.90)", "[0.90-1.00]"]:
    stats(f"  {k}", g.get(k, []))

# Last 30 settled trades (chronological order)
print("\nLast 10 settled:")
chron = sorted(settled, key=lambda r: parse(r["entry_ts"]))[-10:]
for t in chron:
    print(f"  {parse(t['entry_ts']).strftime('%m-%d %H:%M')}  {t['direction']:3s}  ask={t['ask']:.3f}  bet=${t['bet']:5.2f}  score={t['score']:+.3f}  outcome={t['outcome']}  pnl={t['pnl']:+.2f}")
