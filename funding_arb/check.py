"""
Quick sanity check — run before going live.
Verifies API connectivity, hedge mode, balances, and current funding rates
across all configured symbols.

Usage:
  python check.py
"""
import os
import sys
import time
import yaml
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

if not api_key or not api_secret:
    print("ERROR: BINANCE_API_KEY and BINANCE_API_SECRET not set in .env")
    sys.exit(1)

from src.futures_client import FuturesClient
from src.spot_client import SpotClient
from src.risk import liquidation_price, funding_apy, should_enter

with open("config/arb_params.yaml") as f:
    cfg = yaml.safe_load(f)

symbols = cfg.get("symbols", [cfg.get("symbol", "BTCUSDT")])
entry_threshold = cfg["entry_funding_rate"]

print(f"\n=== Funding Arb Pre-flight Check ===\n")

spot = SpotClient(api_key, api_secret)
futures = FuturesClient(api_key, api_secret)

# 1. Funding rates across all symbols
print(f"1. Current funding rates (threshold={entry_threshold:.6f}/8h = {funding_apy(entry_threshold):.2%} APY):")
all_rates = futures.get_all_funding_rates(symbols)

best = None
for r in all_rates:
    sym = r["symbol"]
    rate = r["rate"]
    apy = r["apy"]
    mins = (r["next_funding_ms"] / 1000 - time.time()) / 60
    would_enter = should_enter(rate, entry_threshold)
    marker = "← WOULD ENTER ✓" if would_enter else ""
    print(f"   {sym:12s}  rate={rate:.6f}/8h  APY={apy:>7.2%}  next_funding={mins:.0f}m  {marker}")
    if would_enter and best is None:
        best = r

if best:
    print(f"\n   Best entry: {best['symbol']} at {best['apy']:.2%} APY")
else:
    # Show how far the best rate is from threshold
    if all_rates:
        top = all_rates[0]
        pct = top["rate"] / entry_threshold * 100
        print(f"\n   No entry signal. Best: {top['symbol']} {top['apy']:.2%} APY ({pct:.0f}% of threshold)")

# 2. Spot balance
print("\n2. Spot balances (USDT):")
usdt_spot = spot.get_balance("USDT")
print(f"   USDT: {usdt_spot:.2f}")
usdt_ok = usdt_spot >= cfg["position_usdt"]
print(f"   Need {cfg['position_usdt']:.2f} USDT for position → {'OK ✓' if usdt_ok else 'INSUFFICIENT ✗'}")

# 3. Futures wallet
print("\n3. Futures wallet balance:")
fut_usdt = futures.get_wallet_balance("USDT")
print(f"   USDT: {fut_usdt:.2f}")
margin_needed = cfg["position_usdt"] / cfg["futures_leverage"]
fut_ok = fut_usdt >= margin_needed
print(f"   Need ~{margin_needed:.2f} USDT margin ({cfg['futures_leverage']}x lev) → {'OK ✓' if fut_ok else 'INSUFFICIENT ✗'}")

# 4. Liquidation estimates for each symbol
print("\n4. Liquidation estimates (using current mark prices):")
for r in all_rates:
    price = r["mark_price"]
    liq = liquidation_price(price, cfg["futures_leverage"])
    distance_pct = (liq / price - 1) * 100
    print(f"   {r['symbol']:12s}  mark={price:.2f}  liq≈{liq:.2f}  (+{distance_pct:.1f}% move to liquidate SHORT)")

# 5. Hedge mode
print("\n5. Enabling hedge mode...")
result = futures.enable_hedge_mode()
print(f"   {result.get('msg', 'enabled')} ✓")

# Summary
print("\n=== Summary ===")
all_ok = usdt_ok and fut_ok
print(f"   Spot USDT:      {'✓' if usdt_ok else '✗'}")
print(f"   Futures margin: {'✓' if fut_ok else '✗'}")
print(f"   Hedge mode:     ✓")
entry_signal_str = f"YES — {best['symbol']} {best['apy']:.2%} APY ✓" if best else "NO — rate too low (bot will wait)"
print(f"   Entry signal:   {entry_signal_str}")
print()
if all_ok:
    print("Ready to run: python main.py")
else:
    print("Fix the issues above before running main.py")
    if not usdt_ok:
        print(f"  — Deposit {cfg['position_usdt'] - usdt_spot:.2f} more USDT to Spot")
    if not fut_ok:
        print(f"  — Transfer {margin_needed - fut_usdt:.2f} USDT to Futures wallet")
