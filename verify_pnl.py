#!/usr/bin/env python3
import json
import sys

# Load conviction state from server (via SCP we'll fetch locally)
# For now, let's use the data we already have

# Example data from the state file
closed_positions = [
    {"symbol": "BTC", "timeframe": "15m", "direction": "DOWN", "conviction": 0.955, "entry_price": 0.955, "bet": 10.0, "resolved_direction": "DOWN", "pnl": 212.02222222222204},
    {"symbol": "BTC", "timeframe": "15m", "direction": "DOWN", "conviction": 0.955, "entry_price": 0.955, "bet": 10.0, "resolved_direction": "DOWN", "pnl": 212.02222222222204},
    {"symbol": "SOL", "timeframe": "15m", "direction": "DOWN", "conviction": 0.87, "entry_price": 0.87, "bet": 8.171428571428573, "resolved_direction": "DOWN", "pnl": 54.52228571428572},
    {"symbol": "XRP", "timeframe": "15m", "direction": "DOWN", "conviction": 0.85, "entry_price": 0.85, "bet": 7.714285714285714, "resolved_direction": "DOWN", "pnl": 43.56},
    {"symbol": "ETH", "timeframe": "5m", "direction": "DOWN", "conviction": 0.795, "entry_price": 0.795, "bet": 6.457142857142859, "resolved_direction": "DOWN", "pnl": 24.911972125435554},
    {"symbol": "XRP", "timeframe": "5m", "direction": "DOWN", "conviction": 0.635, "entry_price": 0.635, "bet": 2.8000000000000007, "resolved_direction": "DOWN", "pnl": 4.81523287671233},
]

print("=== VERIFICATION OF CONVICTION BOT PnL CALCULATION ===\n")
print(f"{'Symbol':<6} {'TF':<5} {'Dir':<4} {'Entry':<8} {'Bet':<8} {'Result':<7} {'Payout':<8} {'PnL Stored':<15} {'PnL Calc':<15} {'Match':<5}")
print("-" * 105)

total_calc = 0
total_stored = 0

for pos in closed_positions:
    symbol = pos["symbol"]
    tf = pos["timeframe"]
    direction = pos["direction"]
    entry_price = pos["entry_price"]
    bet = pos["bet"]
    resolved_dir = pos["resolved_direction"]
    pnl_stored = pos["pnl"]

    # Calculate payout using the formula
    if direction == resolved_dir:  # Won
        if direction == "UP":
            payout = (1.0 - entry_price) / entry_price
        else:  # DOWN
            payout = entry_price / (1.0 - entry_price)
        payout_net = payout - 0.02  # 2% fee
        pnl_calc = bet * payout_net
        result = "WIN"
    else:  # Lost
        payout = 0
        payout_net = 0
        pnl_calc = -bet
        result = "LOSS"

    match = "OK" if abs(pnl_calc - pnl_stored) < 0.01 else "DIFF"
    print(f"{symbol:<6} {tf:<5} {direction:<4} {entry_price:.4f}   ${bet:>6.2f}  {result:<7} {payout:>7.4f}   ${pnl_stored:>13.2f}  ${pnl_calc:>13.2f}  {match:<5}")

    total_calc += pnl_calc
    total_stored += pnl_stored

print("-" * 105)
print(f"\nSum (stored): ${total_stored:.2f}")
print(f"Sum (calculated): ${total_calc:.2f}")
print(f"Match: {'CORRECT' if abs(total_calc - total_stored) < 0.01 else 'MISMATCH'}")

# Detailed breakdown for one example
print("\n=== DETAILED BREAKDOWN: 95% CONVICTION BTC DOWN BET ===")
ex = closed_positions[0]
print(f"Trade: {ex['symbol']} {ex['timeframe']} {ex['direction']}")
print(f"Conviction: {ex['conviction']*100:.1f}%")
print(f"Entry Price (NO token cost): ${ex['entry_price']}")
print(f"YES token price (implied): ${1.0 - ex['entry_price']:.4f}")
print(f"Bet Amount: ${ex['bet']:.2f}")
print(f"\nWhat happens when you bet $10 on NO at ${ex['entry_price']}:")
print(f"  1. Number of tokens you can buy: ${ex['bet']} ÷ ${ex['entry_price']} = {ex['bet'] / ex['entry_price']:.2f} tokens")
print(f"  2. If NO wins (resolves to $1.00): {ex['bet'] / ex['entry_price']:.2f} × $1.00 = ${ex['bet'] / ex['entry_price']:.2f}")
print(f"  3. Minus your original bet: ${ex['bet'] / ex['entry_price']:.2f} - ${ex['bet']:.2f} = ${ex['bet'] / ex['entry_price'] - ex['bet']:.2f}")
print(f"  4. Minus 2% fee on bet: ${ex['bet'] / ex['entry_price'] - ex['bet']:.2f} - ${ex['bet'] * 0.02:.2f} = ${ex['bet'] / ex['entry_price'] - ex['bet'] - ex['bet'] * 0.02:.2f}")
print(f"\nUsing the payout formula:")
print(f"  Payout multiplier = entry / (1 - entry) = {ex['entry_price']} / {1 - ex['entry_price']:.4f} = {ex['entry_price'] / (1 - ex['entry_price']):.4f}x")
print(f"  Net payout (after 2% fee) = {ex['entry_price'] / (1 - ex['entry_price']):.4f} - 0.02 = {ex['entry_price'] / (1 - ex['entry_price']) - 0.02:.4f}")
print(f"  PnL = ${ex['bet']:.2f} × {ex['entry_price'] / (1 - ex['entry_price']) - 0.02:.4f} = ${ex['bet'] * (ex['entry_price'] / (1 - ex['entry_price']) - 0.02):.2f}")
print(f"\nStored in state: ${ex['pnl']:.2f}")
print(f"Match: {'OK' if abs(ex['bet'] * (ex['entry_price'] / (1 - ex['entry_price']) - 0.02) - ex['pnl']) < 0.01 else 'DIFF'}")
