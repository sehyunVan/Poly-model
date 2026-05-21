"""
Redeem settled Polymarket CTF positions on-chain.
Adapted from https://gitlab.com/-/snippets/3757273 for web3.py v7 + our .env setup.

Supports both standard Gnosis CTF positions and NegRisk adapter positions.

Run from server:
  source .venv/bin/activate
  python scripts/redeem_positions.py
"""

import os
import time
import requests
from web3 import Web3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

PRIVATE_KEY = os.getenv("KEY") or os.getenv("PRIVATE_KEY")
WALLET      = os.getenv("FUNDER")

if not PRIVATE_KEY or not WALLET:
    raise SystemExit("Missing KEY or FUNDER in .env")

# Polygon RPC — try multiple in case one is down
RPC_URLS = [
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon.llamarpc.com",
]

w3 = None
for url in RPC_URLS:
    try:
        candidate = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
        if candidate.is_connected():
            w3 = candidate
            print(f"Connected to {url}")
            break
    except Exception:
        pass

if w3 is None:
    raise SystemExit("All RPCs failed — check network")

account = w3.eth.account.from_key(PRIVATE_KEY)
print(f"Wallet: {account.address}")
print(f"POL balance: {w3.from_wei(w3.eth.get_balance(account.address), 'ether'):.4f}")

# Polymarket CTF contract (Gnosis Conditional Token Framework on Polygon)
CTF_ADDRESS      = w3.to_checksum_address("0x4d97dcd97ec945f40cf65f87097ace5ea0476045")
NEG_RISK_ADAPTER = w3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")
USDC_E           = w3.to_checksum_address("0x2791bca1f2de4661ed88a30c99a7a9449aa84174")
PUSD             = w3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
HASH_ZERO        = b'\x00' * 32

CTF_ABI = [
    {
        "inputs": [
            {"internalType": "contract IERC20", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32",         "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32",         "name": "conditionId",        "type": "bytes32"},
            {"internalType": "uint256[]",       "name": "indexSets",          "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "id",    "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

NEG_RISK_ABI = [
    {
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "amounts",     "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

ctf         = w3.eth.contract(CTF_ADDRESS, abi=CTF_ABI)
neg_adapter = w3.eth.contract(NEG_RISK_ADAPTER, abi=NEG_RISK_ABI)


def fetch_winning_positions(wallet: str) -> list[dict]:
    """Pull redeemable winning positions from Polymarket data API (paginated)."""
    all_positions = []
    offset = 0
    while True:
        url  = f"https://data-api.polymarket.com/positions?user={wallet}&sizeThreshold=0.01&limit=100&offset={offset}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        all_positions.extend(page)
        if len(page) < 100:
            break
        offset += 100
    winners = [p for p in all_positions if p.get("redeemable") and p.get("curPrice", 0) >= 0.97]
    print(f"Found {len(all_positions)} positions, {len(winners)} winning & redeemable")
    return winners


def get_clob_market(cid: str) -> dict:
    """Fetch CLOB market data — returns neg_risk flag and token list."""
    try:
        resp = requests.get(f"https://clob.polymarket.com/markets/{cid}", timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "neg_risk": data.get("neg_risk", False),
            "tokens": data.get("tokens", []),
        }
    except Exception as e:
        print(f"  CLOB lookup failed for {cid[:20]}: {e}")
        return {}


def position_id(cid_bytes: bytes, index_set: int, collateral: str) -> int:
    """Compute Gnosis CTF ERC-1155 positionId (matches on-chain keccak logic)."""
    collection_id = w3.solidity_keccak(["bytes32", "bytes32", "uint256"], [HASH_ZERO, cid_bytes, index_set])
    pos_id = w3.solidity_keccak(["address", "bytes32"], [w3.to_checksum_address(collateral), collection_id])
    return int.from_bytes(pos_id, "big")


def detect_collateral(cid_bytes: bytes) -> str | None:
    """Return the collateral address (PUSD or USDC_E) that has a non-zero on-chain balance, or None."""
    for c_addr in [PUSD, USDC_E]:
        for idx_set in [1, 2]:
            pid = position_id(cid_bytes, idx_set, c_addr)
            if ctf.functions.balanceOf(account.address, pid).call() > 0:
                return c_addr
    return None


def redeem(pos: dict, nonce: int) -> str | None:
    """Submit one redeemPositions tx. Returns tx hash or None on failure."""
    cid       = pos["conditionId"]
    cid_bytes = bytes.fromhex(cid.removeprefix("0x"))
    gas_price = int(w3.eth.gas_price * 1.5)  # 50% above current

    mkt        = get_clob_market(cid)
    is_neg_risk = mkt.get("neg_risk", False)

    try:
        if is_neg_risk:
            tokens = mkt.get("tokens", [])
            if not tokens:
                print(f"  neg_risk {cid[:20]}: no tokens from CLOB — skipping")
                return None
            # Sort by outcome name (No < Yes alphabetically → [No, Yes])
            # BUT NegRisk adapter expects [YES_amount, NO_amount]
            # Sort by outcome descending so Yes comes first
            tokens_sorted = sorted(tokens, key=lambda t: t.get("outcome", "").lower(), reverse=True)
            amounts = []
            for t in tokens_sorted:
                tid     = t.get("token_id")
                bal_raw = ctf.functions.balanceOf(account.address, int(tid)).call() if tid else 0
                amounts.append(bal_raw)
                print(f"    outcome={t.get('outcome')}  token_id={str(tid)[:20]}  balance={bal_raw/1e6:.4f}")
            if sum(amounts) == 0:
                print(f"  neg_risk {cid[:20]}: zero on-chain balance — skipping")
                return None
            print(f"  neg_risk {cid[:20]}: amounts={amounts}")
            tx = neg_adapter.functions.redeemPositions(
                cid_bytes, amounts,
            ).build_transaction({
                "from":     account.address,
                "nonce":    nonce,
                "gas":      400_000,  # complex multi-outcome markets can use 300k+
                "gasPrice": gas_price,
                "chainId":  137,
            })
        else:
            # Detect V1 (USDC.e) vs V2 (pUSD) via on-chain balance — data API has no acquired_at
            collateral = detect_collateral(cid_bytes)
            if collateral is None:
                print(f"  standard CTF {cid[:20]}: zero on-chain balance — already redeemed?")
                return None
            label = "pUSD" if collateral == PUSD else "USDC.e"
            print(f"  standard CTF {cid[:20]}  collateral={label}")
            tx = ctf.functions.redeemPositions(
                collateral, HASH_ZERO, cid_bytes, [1, 2],
            ).build_transaction({
                "from":     account.address,
                "nonce":    nonce,
                "gas":      250_000,
                "gasPrice": gas_price,
                "chainId":  137,
            })
    except Exception as e:
        print(f"  build_transaction failed: {e}")
        return None

    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    return w3.to_hex(tx_hash)


def main():
    positions = fetch_winning_positions(WALLET)
    if not positions:
        print("Nothing to redeem.")
        return

    # Use pending nonce to avoid conflicts with the running bot process
    nonce = w3.eth.get_transaction_count(account.address, "pending")
    print(f"Starting nonce: {nonce}\n")

    receipts = []
    for i, pos in enumerate(positions):
        cid  = pos["conditionId"]
        size = pos.get("size", 0)
        print(f"[{i+1}/{len(positions)}] ${size:.2f}  {cid[:20]}...")
        tx_hash = redeem(pos, nonce)
        if tx_hash:
            print(f"  submitted: {tx_hash}")
            receipts.append((cid, tx_hash, size))
            nonce += 1
            time.sleep(1)
        else:
            print(f"  skipped")

    print(f"\nWaiting for {len(receipts)} receipts...")
    for cid, tx_hash, size in receipts:
        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            status  = "OK" if receipt.status == 1 else "FAILED"
            print(f"  {status}  ${size:.2f}  {tx_hash[:20]}  (gas used: {receipt.gasUsed})")
        except Exception as e:
            print(f"  TIMEOUT/ERROR {tx_hash[:20]}: {e}")

    print("\nDone. Check CLOB balance at https://clob.polymarket.com/balance")


if __name__ == "__main__":
    main()
