"""
Investigate why redemption txs revert despite curPrice=1.0 in the data API.

Calls CTF.payoutDenominator(conditionId) for each candidate — returns 0 if
market not resolved on-chain. Also simulates the redeem via eth_call to
extract the actual revert reason.
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv
from web3 import Web3

load_dotenv("/home/ubuntu/poly-model/.env")

WALLET   = "0xC883DF1cfa4DACd89aEf4EFcc328219B1c914ea1"
CTF      = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"   # Polymarket CTF on Polygon
USDC_E   = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
PUSD     = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"

CTF_ABI = [
    {"inputs": [{"name": "conditionId", "type": "bytes32"}], "name": "payoutDenominator",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "conditionId", "type": "bytes32"}, {"name": "index", "type": "uint256"}],
     "name": "payoutNumerators",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [
        {"name": "collateralToken", "type": "address"},
        {"name": "parentCollectionId", "type": "bytes32"},
        {"name": "conditionId", "type": "bytes32"},
        {"name": "indexSets", "type": "uint256[]"}],
     "name": "redeemPositions", "outputs": [],
     "stateMutability": "nonpayable", "type": "function"},
]

w3 = Web3(Web3.HTTPProvider("https://rpc-mainnet.matic.quiknode.pro"))
ctf = w3.eth.contract(w3.to_checksum_address(CTF), abi=CTF_ABI)

# Pull 14 winners
url = f"https://data-api.polymarket.com/positions?user={WALLET}&sizeThreshold=0&limit=500"
positions = requests.get(url, timeout=30).json()
winners = [p for p in positions if p.get("curPrice", 0.0) >= 0.97]
print(f"Got {len(winners)} winners; checking on-chain status for each\n")
print(f"{'curPrice':>9} {'redeemable':>10} {'denom':>8} {'pay[0]':>8} {'pay[1]':>8}  title")
print("-" * 100)

for w in winners:
    cid = w.get("conditionId", "")
    if not cid:
        continue
    cid_bytes = bytes.fromhex(cid.removeprefix("0x"))
    try:
        denom = ctf.functions.payoutDenominator(cid_bytes).call()
        pay0  = ctf.functions.payoutNumerators(cid_bytes, 0).call() if denom > 0 else "-"
        pay1  = ctf.functions.payoutNumerators(cid_bytes, 1).call() if denom > 0 else "-"
    except Exception as exc:
        denom, pay0, pay1 = f"err: {exc}", "-", "-"
    print(f"{w.get('curPrice',0):>9.3f} {str(w.get('redeemable')):>10} "
          f"{str(denom):>8} {str(pay0):>8} {str(pay1):>8}  "
          f"{(w.get('title') or w.get('eventTitle') or '')[:60]}")

# Now simulate one redemption to get the revert reason
print("\nSimulating redemption on the highest-curPrice candidate via eth_call...")
top = max(winners, key=lambda p: p.get("curPrice", 0))
cid_bytes = bytes.fromhex(top["conditionId"].removeprefix("0x"))
print(f"  cid={top['conditionId'][:30]}  title={(top.get('title') or '')[:60]}")
print(f"  curPrice={top['curPrice']}  redeemable={top.get('redeemable')}")

for coll, name in [(PUSD, "pUSD"), (USDC_E, "USDC.e")]:
    print(f"\n  → simulating redeemPositions(collateral={name}, conditionId, [1, 2])")
    try:
        ctf.functions.redeemPositions(
            w3.to_checksum_address(coll),
            b"\x00" * 32,
            cid_bytes,
            [1, 2],
        ).call({"from": WALLET})
        print(f"    ✓  simulation OK with {name}")
    except Exception as exc:
        print(f"    ❌ revert: {exc}")
