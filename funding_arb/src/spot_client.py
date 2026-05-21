"""Binance Spot API client."""
import hashlib
import hmac
import math
import time

import requests

SPOT = "https://api.binance.com"


class SpotClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": api_key})
        self._step_cache: dict[str, float] = {}

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
        r = self._session.get(f"{SPOT}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        r = self._session.post(f"{SPOT}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ market data

    def get_price(self, symbol: str) -> float:
        d = self._get("/api/v3/ticker/price", {"symbol": symbol})
        return float(d["price"])

    def get_step_size(self, symbol: str) -> float:
        """Minimum quantity increment (cached)."""
        if symbol in self._step_cache:
            return self._step_cache[symbol]
        info = self._get("/api/v3/exchangeInfo", {"symbol": symbol})
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        self._step_cache[symbol] = step
                        return step
        return 0.00001

    def round_qty(self, symbol: str, qty: float) -> float:
        step = self.get_step_size(symbol)
        precision = max(0, round(-math.log10(step)))
        return round(math.floor(qty / step) * step, precision)

    # ------------------------------------------------------------------ account

    def get_balance(self, asset: str) -> float:
        account = self._get("/api/v3/account", signed=True)
        for b in account["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    # ------------------------------------------------------------------ orders

    def place_market_buy_quote(self, symbol: str, usdt_amount: float) -> dict:
        """Buy using a fixed USDT amount. Returns fill including executedQty (base asset)."""
        return self._post("/api/v3/order", {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": round(usdt_amount, 2),
        })

    def place_market_sell(self, symbol: str, quantity: float) -> dict:
        """Sell exact base quantity."""
        qty = self.round_qty(symbol, quantity)
        return self._post("/api/v3/order", {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": qty,
        })
