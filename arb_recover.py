#!/usr/bin/env python3
"""Recover arb bot: close remaining spot SOL and reset error count."""
import os
import sys
import json
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from dotenv import load_dotenv
load_dotenv()

from funding_arb.src.spot_client import SpotClient

print("=" * 70)
print("ARB RECOVERY — CLOSE SPOT SOL & RESET STATE")
print("=" * 70)

try:
    sc = SpotClient(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

    # Check current spot SOL
    sol_bal = sc.get_balance("SOL")
    print(f"\n[1] Current spot SOL: {sol_bal:.4f}")

    if sol_bal > 0.001:
        print(f"[2] Selling {sol_bal:.4f} SOL...")
        result = sc.place_market_sell("SOLUSDT", sol_bal)
        print(f"    ✓ Sold: {result.get('executedQty', sol_bal)} SOL")
        print(f"    Cummulative quote: {result.get('cummulativeQuoteQty', 'N/A')} USDT")
        print(f"    Order ID: {result.get('orderId', 'N/A')}")
    else:
        print(f"[2] SOL balance near zero, skipping sell")

    # Reset arb state
    print(f"\n[3] Resetting arb_state.json...")
    state_path = repo_root / "funding_arb/data/arb_state.json"
    with open(state_path) as f:
        state = json.load(f)

    state["status"] = "READY"
    state["error_count"] = 0
    state["active_symbol"] = None

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"    ✓ Reset: status=READY, error_count=0, active_symbol=None")
    print(f"\n[4] Recovery complete!")
    print(f"    Next: restart arb bot with:")
    print(f"    screen -S arb -X quit")
    print(f"    sleep 3")
    print(f"    cd ~/poly-model/funding_arb && screen -dmS arb bash -c 'source ../.venv/bin/activate && python main.py'")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
