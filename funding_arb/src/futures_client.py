"""Binance USDM Futures API client."""
import hashlib
import hmac
import math
import time
from typing import Optional

import requests

FAPI = "https://fapi.binance.com"


class FuturesClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": api_key})
        self._precision_cache: dict[str, int] = {}  # symbol → qty decimal places

    # ------------------------------------------------------------------ auth

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        sig = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    def _get(self, path: str, params: dict | None = None, signed: bool = False) -> dict | list:
        if signed:
            params = self._sign(params or {})
        r = self._session.get(f"{FAPI}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        r = self._session.post(f"{FAPI}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        r = self._session.delete(f"{FAPI}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ setup

    def enable_hedge_mode(self) -> dict:
        """Enable dual-side position mode (required for separate LONG/SHORT positions)."""
        try:
            return self._post("/fapi/v1/positionSide/dual", {"dualSidePosition": "true"})
        except requests.HTTPError as e:
            if e.response is not None and "No need to change" in e.response.text:
                return {"msg": "already in hedge mode"}
            raise

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    # ------------------------------------------------------------------ market data

    def get_funding_rate(self, symbol: str) -> dict:
        """Current funding rate + mark price for one symbol."""
        d = self._get("/fapi/v1/premiumIndex", {"symbol": symbol})
        rate = float(d["lastFundingRate"])
        return {
            "symbol": symbol,
            "rate": rate,
            "apy": rate * 3 * 365,
            "next_funding_ms": int(d["nextFundingTime"]),
            "mark_price": float(d["markPrice"]),
        }

    def get_all_funding_rates(self, symbols: list[str]) -> list[dict]:
        """
        Fetch funding rates for multiple symbols in one pass.
        Returns list of rate-info dicts sorted by abs(rate) descending.
        Skips symbols that fail (exchange error, delisted, etc.).
        """
        results = []
        for sym in symbols:
            try:
                results.append(self.get_funding_rate(sym))
            except Exception:
                pass  # symbol may be temporarily unavailable
        results.sort(key=lambda x: abs(x["rate"]), reverse=True)
        return results

    def get_historical_rates(self, symbol: str, limit: int = 90) -> list[float]:
        """Last N funding periods (~8h each)."""
        data = self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
        return [float(x["fundingRate"]) for x in data]

    def get_price(self, symbol: str) -> float:
        d = self._get("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(d["price"])

    def get_qty_precision(self, symbol: str) -> int:
        """
        Number of decimal places for order quantity.
        Result is cached per symbol — exchange info is fetched once per session.
        """
        if symbol in self._precision_cache:
            return self._precision_cache[symbol]
        info = self._get("/fapi/v1/exchangeInfo")
        for s in info["symbols"]:
            prec = 3  # safe default (BTC)
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    prec = max(0, round(-math.log10(step))) if step > 0 else 3
            self._precision_cache[s["symbol"]] = prec
        return self._precision_cache.get(symbol, 3)

    # ------------------------------------------------------------------ account

    def get_account(self) -> dict:
        return self._get("/fapi/v2/account", signed=True)

    def get_wallet_balance(self, asset: str = "USDT") -> float:
        account = self.get_account()
        for b in account["assets"]:
            if b["asset"] == asset:
                return float(b["walletBalance"])
        return 0.0

    def get_position(self, symbol: str, side: str = "SHORT") -> Optional[dict]:
        """Returns the position dict for the given side, or None."""
        account = self.get_account()
        for pos in account["positions"]:
            if pos["symbol"] == symbol and pos["positionSide"] == side:
                amt = float(pos["positionAmt"])
                if abs(amt) > 0:
                    return pos
        return None

    def get_short_qty(self, symbol: str) -> float:
        """Absolute quantity currently shorted."""
        pos = self.get_position(symbol, "SHORT")
        return abs(float(pos["positionAmt"])) if pos else 0.0

    def get_long_qty(self, symbol: str) -> float:
        """Absolute quantity currently long on futures (for reverse position)."""
        pos = self.get_position(symbol, "LONG")
        return abs(float(pos["positionAmt"])) if pos else 0.0

    # ------------------------------------------------------------------ orders (SHORT leg — normal arb)

    def place_market_short(self, symbol: str, quantity: float) -> dict:
        """Open a SHORT position (hedge mode). Used for normal positive-funding arb."""
        precision = self.get_qty_precision(symbol)
        qty = round(quantity, precision)
        return self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": "SELL",
            "positionSide": "SHORT",
            "type": "MARKET",
            "quantity": qty,
        })

    def close_market_short(self, symbol: str, quantity: float) -> dict:
        """Close a SHORT position (hedge mode)."""
        precision = self.get_qty_precision(symbol)
        qty = round(quantity, precision)
        return self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": "BUY",
            "positionSide": "SHORT",
            "type": "MARKET",
            "quantity": qty,
        })

    def cancel_all_orders(self, symbol: str) -> dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    # ------------------------------------------------------------------ orders (LONG leg — reverse arb)

    def place_market_long(self, symbol: str, quantity: float) -> dict:
        """
        Open a LONG futures position (hedge mode).
        Used for reverse arb when funding is negative (shorts pay longs).
        """
        precision = self.get_qty_precision(symbol)
        qty = round(quantity, precision)
        return self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": "BUY",
            "positionSide": "LONG",
            "type": "MARKET",
            "quantity": qty,
        })

    def close_market_long(self, symbol: str, quantity: float) -> dict:
        """Close a LONG futures position (hedge mode)."""
        precision = self.get_qty_precision(symbol)
        qty = round(quantity, precision)
        return self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": "SELL",
            "positionSide": "LONG",
            "type": "MARKET",
            "quantity": qty,
        })
