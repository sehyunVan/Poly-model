"""
Debug the failing wrap_usdc_e_to_pusd() transaction.

Simulates the wrap via eth_call (no gas, no broadcast) and prints the revert reason.
Also verifies the CollateralOnramp address has code and inspects approvals.

Usage on server:
    cd ~/poly-model && source .venv/bin/activate && python /tmp/debug_wrap.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from web3 import Web3

load_dotenv("/home/ubuntu/poly-model/.env")

USDC_E   = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
PUSD     = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
ONRAMP   = "0x93070a847efEf7F70739046A929D47a521F5B8ee"  # CollateralOnramp per redeem.py

RPCS = [
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
    "https://polygon.llamarpc.com",
    "https://1rpc.io/matic",
]

ERC20_ABI = [
    {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "name": "allowance",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
]
ONRAMP_ABI_WRAP = [
    {"inputs": [
        {"name": "collateralAddress", "type": "address"},
        {"name": "recipient",         "type": "address"},
        {"name": "amount",            "type": "uint256"}],
     "name": "wrap", "outputs": [],
     "stateMutability": "nonpayable", "type": "function"},
]


def connect():
    for url in RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                print(f"Connected via {url}")
                return w3
        except Exception as e:
            print(f"  {url} failed: {e}")
    raise RuntimeError("No Polygon RPC reachable")


def main() -> int:
    key = os.getenv("KEY")
    if not key:
        print("KEY not in env"); return 1
    w3 = connect()
    account = w3.eth.account.from_key(key)
    wallet = account.address
    print(f"\nWallet: {wallet}")

    # ── 1. Code at ONRAMP address?
    code = w3.eth.get_code(w3.to_checksum_address(ONRAMP))
    if not code or code == b"" or code == "0x":
        print(f"\n❌ NO CODE at CollateralOnramp address {ONRAMP}")
        print("   Either the address is wrong, or the contract was destroyed.")
    else:
        print(f"\n✓  CollateralOnramp {ONRAMP} has {len(code)} bytes of code")

    # ── 2. Wallet balances
    usdc_e = w3.eth.contract(w3.to_checksum_address(USDC_E), abi=ERC20_ABI)
    pusd   = w3.eth.contract(w3.to_checksum_address(PUSD),   abi=ERC20_ABI)
    bal_usdce_raw = usdc_e.functions.balanceOf(wallet).call()
    bal_pusd_raw  = pusd.functions.balanceOf(wallet).call()
    print(f"\nUSDC.e balance: ${bal_usdce_raw / 1e6:.4f}  (raw {bal_usdce_raw})")
    print(f"pUSD  balance: ${bal_pusd_raw  / 1e6:.4f}  (raw {bal_pusd_raw})")

    # ── 3. Approval to onramp
    allow = usdc_e.functions.allowance(wallet, w3.to_checksum_address(ONRAMP)).call()
    print(f"USDC.e → onramp allowance: {allow}  ({'OK (max)' if allow > 1e30 else 'limited'})")

    # ── 4. Simulate wrap via eth_call (no gas, no broadcast)
    if bal_usdce_raw == 0:
        print("\nSkipping wrap simulation — no USDC.e to wrap")
        return 0
    onramp = w3.eth.contract(w3.to_checksum_address(ONRAMP), abi=ONRAMP_ABI_WRAP)
    print(f"\nSimulating wrap({USDC_E}, {wallet}, {bal_usdce_raw}) via eth_call...")
    try:
        result = onramp.functions.wrap(
            w3.to_checksum_address(USDC_E),
            wallet,
            bal_usdce_raw,
        ).call({"from": wallet})
        print(f"✓  Simulation succeeded — wrap function returned: {result}")
    except Exception as exc:
        print(f"\n❌ wrap REVERTED")
        print(f"   Error: {exc}")
        # Try to extract revert reason
        s = str(exc)
        if "execution reverted" in s.lower():
            print(f"   This is a real contract revert — see error message above for reason.")

    # ── 5. Try wrap with a SMALL amount in case it's a 'wrap exact' issue
    if bal_usdce_raw > 1_000_000:  # > $1
        print(f"\nTrying small-amount wrap simulation (1 USDC.e)...")
        try:
            onramp.functions.wrap(
                w3.to_checksum_address(USDC_E), wallet, 1_000_000,
            ).call({"from": wallet})
            print("✓  Small-amount wrap simulation succeeded — full amount may be the issue")
        except Exception as exc:
            print(f"❌ also reverted: {exc}")

    # ── 6. Probe alternative wrap function signatures
    print("\nProbing alternative function selectors at CollateralOnramp...")
    for sig in [
        "wrap(uint256)",
        "wrap(address,uint256)",
        "wrap(address,address,uint256)",
        "deposit(uint256)",
        "deposit(address,uint256)",
        "mint(uint256)",
    ]:
        selector = Web3.keccak(text=sig)[:4].hex()
        print(f"  {sig:36s}  selector 0x{selector}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
