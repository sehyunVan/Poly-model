import json
import os
import time
from dataclasses import dataclass, field, asdict

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "arb_state.json")


@dataclass
class ArbState:
    # positions: symbol → {entry_price, spot_qty, futures_qty, position_usdt,
    #                       entry_time, last_funding_collected_time}
    positions: dict = field(default_factory=dict)
    total_funding_collected: float = 0.0
    total_realized_pnl: float = 0.0
    trade_count: int = 0
    error_count: int = 0
    current_rates: dict = field(default_factory=dict)  # latest rate scan for dashboard


def load_state() -> ArbState:
    path = os.path.abspath(STATE_FILE)
    if not os.path.exists(path):
        return ArbState()
    try:
        with open(path) as f:
            d = json.load(f)

        # New format: has "positions" key
        if "positions" in d and isinstance(d["positions"], dict):
            state = ArbState()
            state.positions = d.get("positions", {})
            state.total_funding_collected = d.get("total_funding_collected", 0.0)
            state.total_realized_pnl = d.get("total_realized_pnl", 0.0)
            state.trade_count = d.get("trade_count", 0)
            state.error_count = d.get("error_count", 0)
            return state

        # Old format: single-position fields — migrate
        state = ArbState()
        state.total_funding_collected = d.get("total_funding_collected", 0.0)
        state.total_realized_pnl = d.get("total_realized_pnl", 0.0)
        state.trade_count = d.get("trade_count", 0)
        active_sym = d.get("active_symbol", "") or ""
        old_status = d.get("status", "IDLE")
        # Only restore if was explicitly ACTIVE with a named symbol
        if old_status == "ACTIVE" and active_sym:
            state.positions[active_sym] = {
                "symbol": active_sym,
                "entry_price": d.get("entry_price", 0.0),
                "spot_qty": d.get("spot_qty", 0.0),
                "futures_qty": d.get("futures_qty", 0.0),
                "position_usdt": d.get("position_usdt", 50.0),
                "entry_time": d.get("entry_time", time.time()),
                "last_funding_collected_time": d.get("last_funding_collected_time", time.time()),
            }
            print(f"[state] Migrated old ACTIVE position: {active_sym}")
        else:
            print(f"[state] Old format migrated — no active position (status={old_status})")
        return state

    except Exception as e:
        print(f"[state] Failed to load state: {e} — starting fresh")
        return ArbState()


def save_state(state: ArbState) -> None:
    path = os.path.abspath(STATE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, path)
