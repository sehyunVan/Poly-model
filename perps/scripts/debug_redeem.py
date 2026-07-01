"""
Debug why _fetch_redeemable() returns 0 despite $477 pending CTF.

Pulls raw Polymarket data-api positions for the wallet, shows the schema,
and tries the existing redeem.py filters against it.

Usage:
    python /tmp/debug_redeem.py
"""
from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/poly-model/.env")

WALLET = "0xC883DF1cfa4DACd89aEf4EFcc328219B1c914ea1"
API    = "https://data-api.polymarket.com/positions"

# Pull big page; the filter in redeem.py uses sizeThreshold=0 and pagination
url = f"{API}?user={WALLET}&sizeThreshold=0&limit=500"
print(f"GET {url}")
r = requests.get(url, timeout=30)
print(f"  status={r.status_code}  len={len(r.text)}\n")
if r.status_code != 200:
    print(r.text[:500])
    sys.exit(1)

positions = r.json()
print(f"Total positions returned: {len(positions)}\n")
if not positions:
    print("No positions found at all. Maybe wrong wallet address or API down.")
    sys.exit(0)

# Show schema of the first position
print("First position schema:")
print(json.dumps(positions[0], indent=2)[:2000])
print()

# Stats
redeemable_flag   = [p for p in positions if p.get("redeemable")]
high_curprice     = [p for p in positions if p.get("curPrice", 0.0) >= 0.97]
low_curprice      = [p for p in positions if p.get("curPrice", 0.0) <= 0.03]
yes_outcome       = [p for p in positions if p.get("outcome") == "Yes"]
no_outcome        = [p for p in positions if p.get("outcome") == "No"]

print(f"Positions with redeemable=true:        {len(redeemable_flag)}")
print(f"Positions with curPrice >= 0.97:       {len(high_curprice)}")
print(f"Positions with curPrice <= 0.03:       {len(low_curprice)}")
print(f"Positions with outcome=Yes:            {len(yes_outcome)}")
print(f"Positions with outcome=No:             {len(no_outcome)}")
print()

# Sum currentValue / size to compare to pending_ctf_usdc accounting
total_size_value = sum(float(p.get("size", 0)) * float(p.get("avgPrice", 0)) for p in positions)
total_cur_value  = sum(float(p.get("currentValue", 0)) for p in positions)
print(f"Sum of size * avgPrice across all positions: ${total_size_value:.2f}")
print(f"Sum of currentValue across all positions:    ${total_cur_value:.2f}")
print()

# Show positions where redeemable=true OR curPrice >= 0.97 (winners)
candidates = [p for p in positions if p.get("redeemable") or p.get("curPrice", 0) >= 0.97]
print(f"Winner candidates (redeemable=true OR curPrice>=0.97): {len(candidates)}")
if candidates:
    for p in candidates[:10]:
        print(f"  redeemable={p.get('redeemable')}  curPrice={p.get('curPrice',0):.3f}  "
              f"size={float(p.get('size',0)):.2f}  outcome={p.get('outcome')}  "
              f"title={(p.get('title') or p.get('eventTitle') or '')[:60]}")
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more")
print()

# Show a sample of high-curPrice winners and check if they have a negRiskMarketID
print("Top-10 highest curPrice positions:")
top = sorted(positions, key=lambda p: -float(p.get("curPrice", 0)))[:10]
for p in top:
    print(f"  curPrice={p.get('curPrice',0):.3f}  redeemable={p.get('redeemable')}  "
          f"negRisk={p.get('negativeRisk')}  size=${float(p.get('size',0))*float(p.get('avgPrice',0)):.2f}  "
          f"title={(p.get('title') or p.get('eventTitle') or '')[:60]}")
