"""
scripts/setup_v2_approvals.py — One-shot Polymarket V2 upgrade approval script.

Run this ONCE after the April 28, 2026 exchange upgrade (~12:00 UTC / ~21:00 KST).
Sets unlimited pUSD approval for the two new V2 exchange contracts so the bot
can resume placing orders.

Usage:
  source .venv/bin/activate && python scripts/setup_v2_approvals.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PRIVATE_KEY = os.getenv("KEY")
WALLET      = os.getenv("FUNDER")

if not PRIVATE_KEY or not WALLET:
    raise SystemExit("Missing KEY or FUNDER in .env")

# ── Addresses ─────────────────────────────────────────────────────────────────
PUSD_ADDRESS         = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"   # pUSD (proxy)
CTF_EXCHANGE_V2      = "0xE111180000d2663C0091e4f400237545B87B996B"   # CTF Exchange V2
NEG_RISK_EXCHANGE_V2 = "0xe2222d279d744050d28e00520010520000310F59"   # NegRisk CTF Exchange V2
USDC_E               = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"   # USDC.e (V1 collateral)
COLLATERAL_ONRAMP    = "0x93070a847efEf7F70739046A929D47a521F5B8ee"   # CollateralOnramp

MAX_UINT256 = 2**256 - 1

ERC20_ABI = [
    {"inputs": [{"name": "owner","type": "address"},{"name": "spender","type": "address"}],
     "name": "allowance","outputs": [{"name": "","type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [{"name": "spender","type": "address"},{"name": "amount","type": "uint256"}],
     "name": "approve","outputs": [{"name": "","type": "bool"}],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [{"name": "account","type": "address"}],
     "name": "balanceOf","outputs": [{"name": "","type": "uint256"}],"stateMutability": "view","type": "function"},
]

ONRAMP_ABI = [
    {"inputs": [
        {"name": "collateralAddress","type": "address"},
        {"name": "recipient",        "type": "address"},
        {"name": "amount",           "type": "uint256"},
    ], "name": "wrap","outputs": [],"stateMutability": "nonpayable","type": "function"},
]

POLYGON_RPCS = [
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon.llamarpc.com",
]


def connect() -> Web3:
    for url in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                print(f"Connected: {url}")
                return w3
        except Exception:
            pass
    raise SystemExit("All Polygon RPCs failed")


def send_tx(w3: Web3, tx, account, label: str) -> bool:
    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  Submitted {label}: {w3.to_hex(tx_hash)[:20]}...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt and receipt.status == 1:
        print(f"  Confirmed {label} (gas={receipt.gasUsed})")
        return True
    print(f"  FAILED {label} (status={getattr(receipt, 'status', '?')})")
    return False


def approve_if_needed(w3, token, spender_addr, spender_name, account, nonce, gas_price) -> int:
    checksum_spender = w3.to_checksum_address(spender_addr)
    current = token.functions.allowance(account.address, checksum_spender).call()
    threshold = MAX_UINT256 // 2
    if current >= threshold:
        print(f"  {spender_name}: already approved (allowance={current})")
        return nonce
    print(f"  {spender_name}: approving unlimited...")
    tx = token.functions.approve(checksum_spender, MAX_UINT256).build_transaction({
        "from":     account.address,
        "nonce":    nonce,
        "gas":      80_000,
        "gasPrice": gas_price,
        "chainId":  137,
    })
    if send_tx(w3, tx, account, f"approve {spender_name}"):
        return nonce + 1
    return nonce


def main():
    w3      = connect()
    account = w3.eth.account.from_key(PRIVATE_KEY)
    print(f"\nWallet: {account.address}")

    # ── Balances ──────────────────────────────────────────────────────────────
    pol_bal   = float(w3.from_wei(w3.eth.get_balance(account.address), "ether"))
    usdc_e_ct = w3.eth.contract(w3.to_checksum_address(USDC_E), abi=ERC20_ABI)
    pusd_ct   = w3.eth.contract(w3.to_checksum_address(PUSD_ADDRESS), abi=ERC20_ABI)
    usdc_e_bal = usdc_e_ct.functions.balanceOf(account.address).call() / 1e6
    pusd_bal   = pusd_ct.functions.balanceOf(account.address).call() / 1e6

    print(f"POL balance : {pol_bal:.4f}")
    print(f"USDC.e      : ${usdc_e_bal:.4f}")
    print(f"pUSD        : ${pusd_bal:.4f}")

    if pol_bal < 0.05:
        raise SystemExit("Insufficient POL for gas. Top up wallet before running.")

    nonce     = w3.eth.get_transaction_count(account.address)
    gas_price = int(w3.eth.gas_price * 1.5)

    # ── Step 1: Wrap any on-chain USDC.e → pUSD via CollateralOnramp ─────────
    # The CLOB balance is migrated by Polymarket's backend automatically.
    # This step handles any leftover USDC.e sitting in the wallet itself.
    if usdc_e_bal > 0.10:
        print(f"\nStep 1: Wrapping {usdc_e_bal:.4f} USDC.e → pUSD via CollateralOnramp")
        onramp    = w3.eth.contract(w3.to_checksum_address(COLLATERAL_ONRAMP), abi=ONRAMP_ABI)
        raw_amount = int(usdc_e_bal * 1e6)

        # Approve USDC.e for OnRamp
        nonce = approve_if_needed(
            w3, usdc_e_ct, COLLATERAL_ONRAMP, "CollateralOnramp (USDC.e)", account, nonce, gas_price
        )

        tx = onramp.functions.wrap(
            w3.to_checksum_address(USDC_E),
            account.address,
            raw_amount,
        ).build_transaction({
            "from":     account.address,
            "nonce":    nonce,
            "gas":      150_000,
            "gasPrice": gas_price,
            "chainId":  137,
        })
        if send_tx(w3, tx, account, "wrap USDC.e→pUSD"):
            nonce += 1
        nonce = w3.eth.get_transaction_count(account.address)  # re-sync after wrap
    else:
        print(f"\nStep 1: No on-chain USDC.e to wrap ({usdc_e_bal:.4f}). CLOB balance is managed by backend.")

    # ── Step 2: Approve pUSD for new V2 exchange contracts ────────────────────
    print("\nStep 2: Setting pUSD approvals for V2 exchange contracts")
    nonce = approve_if_needed(
        w3, pusd_ct, CTF_EXCHANGE_V2, "CTF Exchange V2", account, nonce, gas_price
    )
    nonce = approve_if_needed(
        w3, pusd_ct, NEG_RISK_EXCHANGE_V2, "NegRisk CTF Exchange V2", account, nonce, gas_price
    )

    # ── Step 3: Sync backend via V2 CLOB client ───────────────────────────────
    print("\nStep 3: Syncing CLOB backend balance allowance")
    try:
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        from py_clob_client_v2.constants import POLYGON
        client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=POLYGON,
            funder=WALLET,
            signature_type=0,
        )
        client.set_api_creds(client.create_or_derive_api_key())
        result = client.update_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  Backend sync: {result}")

        balance = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"  CLOB balance: {balance}")
    except Exception as exc:
        print(f"  Backend sync failed (ok if CLOB is still in maintenance): {exc}")

    print("\nDone. Restart crypto/swarm bots now:")
    print("  screen -S crypto -X quit  (then wait 3s)")
    print("  cd ~/poly-model && screen -dmS crypto bash -c "
          "'source .venv/bin/activate && python src/crypto_main.py >> logs/crypto.log 2>&1'")
    print("  screen -S swarm -X quit  (then wait 3s)")
    print("  cd ~/poly-model && screen -dmS swarm bash -c "
          "'source .venv/bin/activate && python bot/main.py >> logs/swarm.log 2>&1'")


if __name__ == "__main__":
    main()
