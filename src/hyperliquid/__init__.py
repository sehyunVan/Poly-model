# hyperliquid — Hyperliquid perpetual trading module.
#
# Trades BTC-USD-PERP on Hyperliquid using the Polymarket crowd-flow signal
# as an entry trigger.  Runs independently of the Polymarket crypto loop in
# its own screen session (screen -S hl).
#
# Entry point: src/hl_main.py
# Config:      config/hl_params.yaml
# State:       data/hl_state.json  (live) | data/hl_virt_state.json (paper)
# Env vars:    HL_ADDRESS, HL_KEY, HL_VIRTUAL_MODE
