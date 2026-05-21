#!/usr/bin/env python3
"""
Health Check — Live Trade Status + Deployment Info + Profit Metrics
Run: python health_check.py
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import sys

def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def load_jsonl(path, limit=None):
    """Load JSONL file, return list of dicts (most recent first if limit set)"""
    try:
        lines = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        lines.append(json.loads(line))
                    except:
                        pass
        return list(reversed(lines[:limit])) if limit else lines
    except Exception:
        return []

def print_section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

def print_subsection(title):
    print(f"\n>> {title}")
    print("-" * 72)

def health_check():
    root = Path(__file__).parent

    print_section("LIVE TRADING HEALTH CHECK")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    # CRYPTO LIVE TRADES
    print_subsection("CRYPTO LOOP (5m LIVE)")

    real_state = load_json(root / "data" / "real_state.json")
    if real_state:
        closed = real_state.get("closed_positions", [])
        crypto = [p for p in closed if p.get("category") == "crypto"]
        wins = [p for p in crypto if p.get("realized_pnl", 0) > 0]
        losses = [p for p in crypto if p.get("realized_pnl", 0) < 0]

        pnl = sum(p.get("realized_pnl", 0) for p in crypto)
        avg_win = sum(p.get("realized_pnl", 0) for p in wins) / len(wins) if wins else 0
        avg_loss = sum(p.get("realized_pnl", 0) for p in losses) / len(losses) if losses else 0

        print(f"Total Trades:     {len(crypto)}")
        print(f"Win Rate:         {len(wins)}/{len(crypto)} = {len(wins)/len(crypto)*100:.1f}%" if crypto else "N/A")
        print(f"Wins:             {len(wins)} trades, avg +${avg_win:.2f}")
        print(f"Losses:           {len(losses)} trades, avg ${avg_loss:.2f}")
        print(f"Net PnL:          ${pnl:+.2f}")
        print(f"Available USDC:   ${real_state.get('available_usdc', 0):.2f}")
    else:
        print("[ERROR] No real_state.json found")

    # SWARM REAL TRADES
    print_subsection("SWARM AI BOT (LIVE, REAL CLOB ORDERS)")

    swarm_trades = load_jsonl(root / "data" / "swarm_real_trades.jsonl")
    if swarm_trades:
        settled = [t for t in swarm_trades if t.get("settled")]
        wins = [t for t in settled if t.get("outcome") == "WIN"]

        pnl = sum(t.get("pnl", 0) for t in settled)

        print(f"Total Settled:    {len(settled)}")
        print(f"Win Rate:         {len(wins)}/{len(settled)} = {len(wins)/len(settled)*100:.1f}%" if settled else "N/A")
        print(f"Net PnL (real):   ${pnl:+.2f}")

        # Show last 3 trades
        if settled:
            print(f"\nLast 3 trades:")
            for t in settled[:3]:
                outcome = t.get("outcome", "?")
                pnl = t.get("pnl", 0)
                print(f"  - {t.get('question', 'unknown')[:40]}... ask={t.get('ask', 0):.3f} bet=${t.get('bet', 0):.0f} {outcome} pnl=${pnl:+.2f}")
    else:
        print("No real swarm trades yet")

    # DEPLOYMENT INFO
    print_subsection("DEPLOYED CHANGES (2026-04-27 01:04 UTC)")

    print("CRYPTO FILTER REMOVALS:")
    print("  [REMOVED] Score cap DISABLED (max_signal_score: 1.0)")
    print("  [REMOVED] Hour block REMOVED (trade_hour_block: [])")
    print("  [REMOVED] SOL hour block REMOVED (sol_hour_block: [])")
    print("  Expected: 4-5x more trades, ask-aware sizing fixes payout")

    print("\nSWARM GATE RELAXATIONS:")
    print("  [LOWERED] Avg_confidence: 70% -> 65% (+2x volume)")
    print("  [CHANGED] Direction-mismatch: SKIP -> 50% sizing penalty")
    print("  [UNBLOCKED] YES direction (was hard-blocked)")
    print("  [LOWERED] Score floor: 0.70 -> 0.65")
    print("  Expected: 50-60 trades/week, profitability pattern emerges")

    # PROFIT STATION
    print_subsection("PROFIT STATION - COMBINED METRICS")

    total_pnl = 0
    total_trades = 0
    total_wins = 0

    if real_state:
        crypto = [p for p in real_state.get("closed_positions", []) if p.get("category") == "crypto"]
        total_pnl += sum(p.get("realized_pnl", 0) for p in crypto)
        total_trades += len(crypto)
        total_wins += len([p for p in crypto if p.get("realized_pnl", 0) > 0])

    if swarm_trades:
        settled = [t for t in swarm_trades if t.get("settled")]
        total_pnl += sum(t.get("pnl", 0) for t in settled)
        total_trades += len(settled)
        total_wins += len([t for t in settled if t.get("outcome") == "WIN"])

    print(f"Combined Trades:  {total_trades}")
    print(f"Combined WR:      {total_wins}/{total_trades} = {total_wins/total_trades*100:.1f}%" if total_trades else "N/A")
    print(f"Combined PnL:     ${total_pnl:+.2f}")

    if total_pnl > 0:
        print(f"Status:           [PROFITABLE]")
    elif total_pnl < -50:
        print(f"Status:           [LOSING] (needs attention)")
    else:
        print(f"Status:           [MONITORING] (watching filters)")

    print(f"\nTest Duration:    Until 2026-05-04 (1 week from deployment)")
    print(f"Success Metric:   Both systems show positive PnL via volume + sizing")

    print("\n" + "=" * 72)

if __name__ == "__main__":
    health_check()
