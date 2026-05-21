#!/usr/bin/env python3
"""Check actual SOLUSDT position on Binance and suggest recovery action."""
import os
import sys
from pathlib import Path

# Setup path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from dotenv import load_dotenv
load_dotenv()

from funding_arb.src.futures_client import FuturesClient
from funding_arb.src.spot_client import SpotClient
import json

print("=" * 70)
print("ARB POSITION CHECK — SOLUSDT")
print("=" * 70)

try:
    fc = FuturesClient(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    sc = SpotClient(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

    # Check futures SHORT position
    print("\n[1] FUTURES POSITION (SHORT):")
    sol_short = fc.get_position("SOLUSDT", side="SHORT")

    if sol_short and float(sol_short.get("positionAmt", 0)) != 0:
        qty = float(sol_short["positionAmt"])
        print(f"    ✓ SHORT Position EXISTS")
        print(f"      Quantity: {abs(qty)}")
        print(f"      Entry Price: {sol_short.get('entryPrice', 'N/A')}")
        print(f"      Mark Price: {sol_short.get('markPrice', 'N/A')}")
        unrealized_pnl = float(sol_short.get("unrealizedProfit", 0))
        print(f"      Unrealized PnL: ${unrealized_pnl:.4f}")
    else:
        print(f"    ✗ No SHORT position (closed or none)")
        qty = 0

    # Check spot
    print("\n[2] SPOT POSITION:")
    sol_spot = sc.get_balance("SOL")
    usdt_spot = sc.get_balance("USDT")
    print(f"    SOL: {sol_spot:.4f}")
    print(f"    USDT: {usdt_spot:.2f}")

    # Check arb state
    print("\n[3] ARB STATE FILE:")
    with open(repo_root / "funding_arb/data/arb_state.json") as f:
        state = json.load(f)
    print(f"    Status: {state['status']}")
    print(f"    Active symbol: {state['active_symbol']}")
    print(f"    Spot qty (recorded): {state['spot_qty']}")
    print(f"    Futures qty (recorded): {state['futures_qty']}")
    print(f"    Error count: {state['error_count']}")

    # Recommendation
    print("\n[4] RECOMMENDATION:")
    if qty == 0 and sol_spot > 0.5:
        print("    ✓ Futures CLOSED, Spot OPEN")
        print("    → Position was partially closed. Spot SOL needs manual redemption.")
        print("    → Action: Close spot SOL manually or via spot_client.sell()")
        print("    → Then reset arb_state.json: set error_count=0, status=READY")
        print("    → Then restart arb bot")
    elif qty > 0:
        print("    ✗ Futures position STILL OPEN")
        print("    → Binance reject was real, position not closed.")
        print("    → Action: Either:")
        print("       (a) Manual close on Binance futures (BUY to close SHORT)")
        print("       (b) Run: arb_close_manual.py (create script)")
        print("    → Then reset arb_state.json and restart")
    else:
        print("    ✓ Both CLOSED")
        print("    → Position successfully closed despite error.")
        print("    → Action: Reset error_count=0, status=READY in state file")
        print("    → Then restart arb bot")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
