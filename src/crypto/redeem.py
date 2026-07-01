"""
crypto/redeem.py — Auto-redeem settled Polymarket CTF positions on-chain.

Called by loop.py when the CLOB balance drops below the trading threshold.
Fetches all redeemable winning positions via the data API, then submits
redeemPositions() transactions to the Polygon CTF contract.

Returns the total USDC.e amount redeemed (or 0.0 on failure / nothing to do).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infra.types import ConditionId  # noqa: E402

# Module-level deduplication: conditionId → unix timestamp when last submitted.
# Prevents retrying the same positions every 10 min (burns POL for no benefit).
_submitted_cids: dict[str, float] = {}
_SUBMIT_TTL = 7200.0  # 2 hours before we'll retry a conditionId

_CTF_ADDRESS        = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"
_NEG_RISK_ADAPTER   = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
_USDC_E             = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"  # V1 collateral
_PUSD               = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"  # V2 collateral
_COLLATERAL_ONRAMP  = "0x93070a847efEf7F70739046A929D47a521F5B8ee"  # wraps USDC.e → pUSD
_V2_UPGRADE_TS      = 1745834400  # 2026-04-28 11:00 UTC
_HASH_ZERO          = b"\x00" * 32
_POLYGON_RPCS = [
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon.llamarpc.com",
]
_CTF_ABI = [
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
    {
        "inputs": [
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId",        "type": "bytes32"},
            {"name": "indexSet",           "type": "uint256"},
        ],
        "name": "getCollectionId",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "collectionId",   "type": "bytes32"},
        ],
        "name": "getPositionId",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]
# NegRisk adapter ABI — redeemPositions(conditionId, amounts[])
# amounts[i] = raw token units for outcome i (YES=index 0, NO=index 1)
_NEG_RISK_ABI = [
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
_IS_APPROVED_ABI = [
    {
        "inputs": [{"name": "owner", "type": "address"}, {"name": "operator", "type": "address"}],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]
_ERC20_ABI = [
    {
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]
_ONRAMP_ABI = [
    {
        "inputs": [
            {"name": "collateralAddress", "type": "address"},
            {"name": "recipient",         "type": "address"},
            {"name": "amount",            "type": "uint256"},
        ],
        "name": "wrap",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


def _ensure_neg_risk_approved(w3, ctf, account, nonce: int, gas_price: int, log: logging.Logger) -> None:
    """
    Call CTF.setApprovalForAll(negRiskAdapter, true) if not already set.
    Required once per wallet for neg_risk redemptions to succeed.
    """
    try:
        ctf_with_approval = w3.eth.contract(
            w3.to_checksum_address(_CTF_ADDRESS), abi=_IS_APPROVED_ABI
        )
        already = ctf_with_approval.functions.isApprovedForAll(
            account.address, w3.to_checksum_address(_NEG_RISK_ADAPTER)
        ).call()
        if already:
            return
        log.info("auto-redeem: setting CTF.setApprovalForAll(negRiskAdapter) — one-time setup")
        tx = ctf_with_approval.functions.setApprovalForAll(
            w3.to_checksum_address(_NEG_RISK_ADAPTER), True
        ).build_transaction({
            "from":     account.address,
            "nonce":    nonce,
            "gas":      80_000,
            "gasPrice": gas_price,
            "chainId":  137,
        })
        signed  = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        if receipt and receipt.status == 1:
            log.info("auto-redeem: negRiskAdapter approval set (gas=%d)", receipt.gasUsed)
        else:
            log.warning("auto-redeem: setApprovalForAll failed — neg_risk redemptions will fail")
    except Exception as exc:
        log.warning("auto-redeem: could not check/set negRiskAdapter approval: %s", exc)


def _collateral_for_position(pos: dict) -> str:
    """
    Return the collateral token address for a given position dict.

    Positions created before the V2 upgrade (2026-04-28 11:00 UTC) use USDC.e.
    Positions created after the upgrade use pUSD.
    We infer this from the position's acquired_at timestamp if present, otherwise
    default to pUSD for safety (upgrade happened, most new positions will be pUSD).
    """
    import time as _time
    acquired = pos.get("acquired_at") or pos.get("created_at") or 0
    # acquired_at may be a Unix timestamp (int) or ISO string
    if isinstance(acquired, str):
        try:
            from datetime import datetime, timezone
            acquired = datetime.fromisoformat(acquired.replace("Z", "+00:00")).timestamp()
        except Exception:
            acquired = 0
    if acquired and float(acquired) < _V2_UPGRADE_TS:
        return _USDC_E  # pre-upgrade position — use USDC.e
    return _PUSD  # post-upgrade position (or unknown) — use pUSD


def _position_id(w3, ctf, cid_bytes: bytes, index_set: int, collateral: str) -> int:
    """
    Return the ERC-1155 positionId using the CTF contract's own view functions.

    Polymarket's CTF uses a different encoding than standard Gnosis keccak256,
    so we delegate to getCollectionId + getPositionId instead of reimplementing
    the formula in Python (which produced wrong IDs and always returned 0 on balanceOf).

    index_set: 1 = outcome-0 (YES), 2 = outcome-1 (NO)
    """
    try:
        collection_id = ctf.functions.getCollectionId(_HASH_ZERO, cid_bytes, index_set).call()
        return ctf.functions.getPositionId(w3.to_checksum_address(collateral), collection_id).call()
    except Exception:
        return 0


def _connect_polygon(log: logging.Logger):
    """Return a connected Web3 instance or None."""
    try:
        from web3 import Web3
    except ImportError:
        log.warning("auto-redeem: web3 not installed — skipping")
        return None

    for url in _POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 8}))
            if w3.is_connected():
                return w3
        except Exception:
            pass
    log.warning("auto-redeem: all Polygon RPCs failed")
    return None


def _fetch_redeemable(wallet: str, log: logging.Logger, w3=None) -> list[dict]:
    """Return list of redeemable winning positions (curPrice >= 0.97 — token won).

    Note: the Polymarket data API's redeemable=True flag is NEVER cleared after
    on-chain redemption, so ghost positions appear indefinitely.  We rely on the
    2-hour deduplication dict (_submitted_cids) to avoid retrying the same
    conditionId repeatedly.  Ghost redemptions succeed (status=1) but transfer
    $0, wasting only a tiny amount of POL gas.

    NOTE: The on-chain positionId computed via keccak256(USDC_E, keccak256(HASH_ZERO,
    conditionId, indexSet)) does NOT match the CLOB token IDs Polymarket assigns
    (e.g. 11489924403476231472 for Knicks NO token).  The CLOB token IDs can only
    be retrieved via the CLOB /markets/{cid} endpoint.  We therefore skip the
    on-chain balance pre-check to avoid incorrectly blocking live redeemable
    positions from being processed.
    """
    try:
        all_positions: list[dict] = []
        offset = 0
        while True:
            url  = (
                f"https://data-api.polymarket.com/positions"
                f"?user={wallet}&sizeThreshold=0.01&limit=100&offset={offset}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            all_positions.extend(page)
            if len(page) < 100:
                break
            offset += 100
        # Winner filter: redeemable=true AND curPrice >= 0.97.
        #
        # `redeemable` is Polymarket's signal that UMA has finalized the market
        # on-chain and CTF.redeemPositions() will succeed. It is set true on
        # BOTH winners and losers post-finalization (losers redeem $0).
        # `curPrice >= 0.97` further restricts to winners (the loser token
        # prices toward 0.0).
        #
        # Note (2026-06-01 audit): if you see many positions with curPrice ≈ 1.0
        # but redeemable=false, those are wins still in UMA's finalization window
        # (typically 24-72h after market end). They will transition to redeemable=
        # true automatically and be picked up by the next AR cycle. Do NOT drop
        # the `redeemable` AND check — that causes gas-wasting reverts and double-
        # counts pending_ctf via the "won but not yet redeemable" path below.
        winners = [
            p for p in all_positions
            if p.get("redeemable") and p.get("curPrice", 0.0) >= 0.97
        ]
        return winners
    except Exception as exc:
        log.warning("auto-redeem: failed to fetch positions — %s", exc)
        return []


def _get_clob_market(cid: str, log: logging.Logger) -> dict:
    """
    Fetch CLOB market data for a conditionId.
    Returns dict with keys: neg_risk (bool), tokens (list of {token_id, outcome, winner}).
    Returns empty dict on failure.
    """
    try:
        resp = requests.get(
            f"https://clob.polymarket.com/markets/{cid}",
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "neg_risk": data.get("neg_risk", False),
            "tokens": data.get("tokens", []),
        }
    except Exception as exc:
        log.warning("auto-redeem: CLOB market lookup failed for %s: %s", cid[:20], exc)
        return {}


def _get_clob_token_ids(cid: str, log: logging.Logger) -> list[int]:
    """Return list of ERC-1155 token IDs for a condition (used by get_pending_ctf_value)."""
    mkt = _get_clob_market(cid, log)
    return [int(t["token_id"]) for t in mkt.get("tokens", []) if t.get("token_id")]


def get_pending_ctf_value(wallet: str, log: logging.Logger) -> float:  # wallet = plain EOA address str
    """
    Return total USD value of won-but-not-yet-redeemed CTF positions.

    Uses the actual CLOB token IDs (from the CLOB /markets/{cid} endpoint) to
    query on-chain ERC-1155 balances.  The keccak256-computed positionIds do NOT
    match Polymarket's CLOB token IDs and always return 0 — do not use them here.

    Also includes positions with curPrice >= 0.90 and redeemable=False (won but
    UMA oracle not yet finalized) by summing their API size values directly.

    Fallback to API-sum if web3 / RPC is unavailable.
    """
    # Redeemable now (curPrice>=0.97, redeemable=True) — check on-chain balance
    winners = _fetch_redeemable(wallet, log)

    # Also count positions won but not yet redeemable (UMA pending)
    try:
        all_positions: list[dict] = []
        offset = 0
        while True:
            url = (
                f"https://data-api.polymarket.com/positions"
                f"?user={wallet}&sizeThreshold=0.01&limit=100&offset={offset}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            all_positions.extend(page)
            if len(page) < 100:
                break
            offset += 100
        # Won but UMA not finalized yet: curPrice >= 0.97 but redeemable=False
        pending_uma = [
            p for p in all_positions
            if not p.get("redeemable") and p.get("curPrice", 0.0) >= 0.97
        ]
    except Exception:
        pending_uma = []

    if not winners and not pending_uma:
        return 0.0

    w3 = _connect_polygon(log)

    total = 0.0

    # Redeemable positions: verify on-chain balance using real CLOB token IDs
    if winners:
        if w3 is None:
            log.warning("get_pending_ctf: web3 unavailable — using stale API estimate for redeemable")
            total += sum(p.get("size", 0) for p in winners)
        else:
            ctf = w3.eth.contract(w3.to_checksum_address(_CTF_ADDRESS), abi=_CTF_ABI)
            for pos in winners:
                cid_hex = pos.get("conditionId", "")
                if not cid_hex:
                    continue
                try:
                    token_ids = _get_clob_token_ids(cid_hex, log)
                    if not token_ids:
                        # Fallback to API size if CLOB lookup fails
                        total += pos.get("size", 0)
                        continue
                    pos_total = 0.0
                    for tid in token_ids:
                        balance_raw = ctf.functions.balanceOf(wallet, tid).call()
                        pos_total += balance_raw / 1e6
                    total += pos_total
                except Exception as exc:
                    log.warning("get_pending_ctf: balanceOf failed for %s: %s", cid_hex[:20], exc)
                    total += pos.get("size", 0)  # fallback to API size for this position

    # UMA-pending positions: won on event data but oracle not finalized on-chain.
    # Use API size directly — tokens are in wallet but not yet redeemable.
    for pos in pending_uma:
        total += pos.get("size", 0)

    return round(total, 4)


def wrap_usdc_e_to_pusd(
    wallet: str,
    private_key: str,
    log: logging.Logger,
    min_amount: float = 1.0,
) -> float:
    """
    Detect on-chain USDC.e in the wallet and wrap it → pUSD via CollateralOnramp.

    V1 CTF redemptions return USDC.e directly to the wallet.  After the V2 upgrade
    the CLOB only tracks pUSD, so that USDC.e is invisible to the trading loop until
    it is wrapped.  This function is called in the 12-hour AR cycle so it runs
    automatically alongside redemption.

    Returns the USD amount wrapped (0.0 if nothing done or an error occurred).
    """
    try:
        from web3 import Web3
    except ImportError:
        return 0.0

    w3 = _connect_polygon(log)
    if w3 is None:
        return 0.0

    account   = w3.eth.account.from_key(private_key)
    usdc_e    = w3.eth.contract(w3.to_checksum_address(_USDC_E), abi=_ERC20_ABI)
    raw_bal   = usdc_e.functions.balanceOf(account.address).call()
    amount    = raw_bal / 1e6

    if amount < min_amount:
        return 0.0

    log.info("wrap: $%.4f USDC.e found in wallet — wrapping to pUSD", amount)

    try:
        nonce     = w3.eth.get_transaction_count(account.address, "pending")
        gas_price = int(w3.eth.gas_price * 1.5)
        onramp_cs = w3.to_checksum_address(_COLLATERAL_ONRAMP)

        # Approve CollateralOnramp for USDC.e if allowance is insufficient
        if usdc_e.functions.allowance(account.address, onramp_cs).call() < raw_bal:
            approve_tx = usdc_e.functions.approve(onramp_cs, 2 ** 256 - 1).build_transaction({
                "from": account.address, "nonce": nonce,
                "gas": 80_000, "gasPrice": gas_price, "chainId": 137,
            })
            signed  = account.sign_transaction(approve_tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            if not receipt or receipt.status != 1:
                log.warning("wrap: USDC.e approve failed — skipping wrap")
                return 0.0
            nonce += 1

        onramp = w3.eth.contract(onramp_cs, abi=_ONRAMP_ABI)
        wrap_tx = onramp.functions.wrap(
            w3.to_checksum_address(_USDC_E), account.address, raw_bal,
        ).build_transaction({
            "from": account.address, "nonce": nonce,
            # wrap() measured at ~156k gas — 150k was too low and reverted out-of-gas
            # (status=0) on every AR cycle, leaving redeemed USDC.e unwrapped/unusable.
            "gas": 250_000, "gasPrice": gas_price, "chainId": 137,
        })
        signed  = account.sign_transaction(wrap_tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        if receipt and receipt.status == 1:
            log.info("wrap: confirmed $%.4f USDC.e → pUSD (gas=%d)", amount, receipt.gasUsed)
            return amount
        log.warning("wrap: wrap tx status=%s", getattr(receipt, "status", "?") if receipt else "no receipt")
    except Exception as exc:
        log.warning("wrap: error — %s", exc)
    return 0.0


def redeem_redeemable_positions(
    wallet: str,
    private_key: str,
    log: logging.Logger,
) -> float:
    """
    Redeem all winning redeemable CTF positions on-chain.

    Returns total USDC.e value redeemed (sum of position sizes), or 0.0 if
    nothing was redeemed or an error occurred.
    """
    try:
        from web3 import Web3
    except ImportError:
        log.warning("auto-redeem: web3 not installed — skipping")
        return 0.0

    # Prune expired deduplication entries
    now = time.time()
    expired = [k for k, ts in _submitted_cids.items() if now - ts > _SUBMIT_TTL]
    for k in expired:
        del _submitted_cids[k]

    w3 = _connect_polygon(log)
    if w3 is None:
        return 0.0

    winners = _fetch_redeemable(wallet, log)
    if not winners:
        log.info("auto-redeem: no redeemable winning positions found")
        return 0.0

    # Skip conditionIds submitted recently — they're either confirmed or in a retry backoff
    new_winners = [
        p for p in winners
        if p.get("conditionId", "") not in _submitted_cids
    ]
    skipped = len(winners) - len(new_winners)
    if skipped:
        log.info("auto-redeem: skipping %d already-submitted conditionIds (dedup)", skipped)
    if not new_winners:
        log.info("auto-redeem: all positions already submitted recently — nothing to do")
        return 0.0
    winners = new_winners

    total_value = sum(p.get("size", 0) for p in winners)
    log.info(
        "auto-redeem: found %d new redeemable positions worth $%.2f — redeeming...",
        len(winners), total_value,
    )

    account = w3.eth.account.from_key(private_key)
    pol_balance = w3.from_wei(w3.eth.get_balance(account.address), "ether")
    if float(pol_balance) < 0.01:
        log.warning(
            "auto-redeem: POL balance %.4f too low for gas — skipping redemption",
            float(pol_balance),
        )
        return 0.0

    ctf       = w3.eth.contract(w3.to_checksum_address(_CTF_ADDRESS), abi=_CTF_ABI)
    nonce     = w3.eth.get_transaction_count(account.address)
    gas_price = int(w3.eth.gas_price * 1.5)   # 50% premium — ensures priority on Polygon

    # Ensure NegRisk adapter is approved to transfer CTF tokens on our behalf.
    # Required once per wallet — costs ~46k gas; skipped automatically once set.
    _ensure_neg_risk_approved(w3, ctf, account, nonce, gas_price, log)
    # Approval tx uses a nonce if it fires — refresh nonce to stay in sync
    nonce = w3.eth.get_transaction_count(account.address)

    # Process one at a time: submit → wait for confirmation → proceed to next.
    # Batch submission with a shared 10s wait fails because sequential nonces on
    # Polygon take 2s/block each, so only 1-2 of N transactions confirm before
    # the receipt check runs.  Serial confirmation is slower (~15s/position) but
    # reliable, and AR fires at most once per 10 minutes so the latency is fine.
    neg_adapter = w3.eth.contract(w3.to_checksum_address(_NEG_RISK_ADAPTER), abi=_NEG_RISK_ABI)

    confirmed_value = 0.0
    for pos in winners:
        cid = ConditionId(pos.get("conditionId", ""))
        if not cid:
            continue
        cid_str = str(cid)

        # Check market type — neg_risk positions require a different adapter contract
        mkt = _get_clob_market(cid_str, log)
        is_neg_risk = mkt.get("neg_risk", False)

        try:
            cid_bytes = bytes.fromhex(cid.removeprefix("0x"))

            if is_neg_risk:
                # NegRisk adapter: redeemPositions(conditionId, amounts[])
                # amounts[i] = raw ERC-1155 balance for outcome i (YES=0, NO=1)
                tokens = mkt.get("tokens", [])
                if not tokens:
                    log.warning("auto-redeem: no tokens for neg_risk %s — skipping", cid[:20])
                    continue
                # Sort descending by outcome name: "Yes" > "No" → Yes first (index 0), No second (index 1)
                # NegRisk adapter expects amounts[0]=YES balance, amounts[1]=NO balance
                tokens_sorted = sorted(tokens, key=lambda t: t.get("outcome", "").lower(), reverse=True)
                amounts = []
                for t in tokens_sorted:
                    tid = t.get("token_id")
                    bal_raw = ctf.functions.balanceOf(account.address, int(tid)).call() if tid else 0
                    amounts.append(bal_raw)
                if sum(amounts) == 0:
                    log.info("auto-redeem: zero balance for neg_risk %s — marking as done", cid[:20])
                    _submitted_cids[cid_str] = time.time()  # don't re-check this one
                    continue
                log.info(
                    "auto-redeem: neg_risk %s amounts=%s (raw)",
                    cid[:20], amounts,
                )
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
                # Standard Gnosis CTF path.
                # Determine collateral (pUSD for V2, USDC.e for V1) by checking
                # on-chain ERC-1155 balance using the contract's own getPositionId view
                # function.  (Python keccak reimplementation produces wrong IDs because
                # Polymarket uses a different encoding — always returned 0 pre-fix.)
                collateral_addr = None
                for c_addr in [_PUSD, _USDC_E]:
                    for idx_set in [1, 2]:
                        pid = _position_id(w3, ctf, cid_bytes, idx_set, c_addr)
                        if ctf.functions.balanceOf(account.address, pid).call() > 0:
                            collateral_addr = c_addr
                            break
                    if collateral_addr:
                        break
                if collateral_addr is None:
                    log.info(
                        "auto-redeem: no on-chain balance for %s (already redeemed?) — skipping",
                        cid[:20],
                    )
                    _submitted_cids[cid_str] = time.time()
                    continue
                log.info(
                    "auto-redeem: standard CTF %s collateral=%s",
                    cid[:20], "pUSD" if collateral_addr == _PUSD else "USDC.e",
                )
                tx = ctf.functions.redeemPositions(
                    w3.to_checksum_address(collateral_addr), _HASH_ZERO, cid_bytes, [1, 2],
                ).build_transaction({
                    "from":     account.address,
                    "nonce":    nonce,
                    "gas":      250_000,
                    "gasPrice": gas_price,
                    "chainId":  137,
                })

            signed  = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            log.info(
                "auto-redeem: submitted %s... (nonce=%d, neg_risk=%s)",
                w3.to_hex(tx_hash)[:20], nonce, is_neg_risk,
            )
            _submitted_cids[cid_str] = time.time()   # dedup: won't retry for 2h
            nonce += 1
        except Exception as exc:
            log.warning("auto-redeem: failed to submit for %s: %s", cid[:20], exc)
            continue

        # Wait for this specific tx to confirm before moving to the next.
        # Timeout 20s = 10 blocks at 2s/block — well within Polygon's inclusion time.
        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=20)
            if receipt and receipt.status == 1:
                confirmed_value += pos.get("size", 0)
                log.info(
                    "auto-redeem: confirmed $%.2f for %s (gas=%d)",
                    pos.get("size", 0), cid[:20], receipt.gasUsed,
                )
            else:
                status = receipt.status if receipt else "no receipt"
                log.warning(
                    "auto-redeem: tx %s... status=%s — not counting",
                    w3.to_hex(tx_hash)[:20], status,
                )
                _submitted_cids.pop(cid_str, None)
        except Exception as exc:
            log.warning(
                "auto-redeem: tx %s timed out / not found: %s — keeping dedup (2h), tx likely confirmed",
                w3.to_hex(tx_hash)[:20], exc,
            )

    if confirmed_value > 0:
        log.info("auto-redeem: confirmed $%.2f redeemed on-chain", confirmed_value)
    return confirmed_value
