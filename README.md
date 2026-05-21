# Polymarket Auto-Trading System

A 24/7 Python trading stack for [Polymarket](https://polymarket.com) prediction
markets, paired with a Binance funding-rate arbitrage bot. Runs unattended on a
Linux VM, makes real CLOB orders on Polygon, and exposes a local web dashboard.

> The narrative of **why** the current strategies look the way they do — what we
> tried, what failed, what the data said — lives in
> [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Architecture

Four independent processes, each in its own `screen` session:

```
┌──────────────────────┐   ┌──────────────────────┐
│  crypto              │   │  swarm               │
│  ─────────────────── │   │  ─────────────────── │
│  Polymarket 5-min    │   │  Long-form markets   │
│  BTC/ETH/SOL up/down │   │  (sports, politics)  │
│  Flow signal +       │   │  Whale-trigger →     │
│  ML filter           │   │  5 LLMs blind-vote → │
│  LIVE                │   │  Deepseek gate, LIVE │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           ▼                          ▼
       ┌───────────────────────────────────┐
       │   Polygon wallet (USDC.e + POL)   │
       │   CTF auto-redemption every 12h   │
       └───────────────────────────────────┘

┌──────────────────────┐   ┌──────────────────────┐
│  arb                 │   │  dashboard           │
│  ─────────────────── │   │  ─────────────────── │
│  Binance funding-    │   │  Threaded HTTPServer │
│  rate arb (delta-    │   │  PnL, positions,     │
│  neutral) across     │   │  signal log, model   │
│  BTC/ETH/SOL/BNB     │   │  accuracy            │
│  LIVE                │   │  localhost:8765      │
└──────────────────────┘   └──────────────────────┘
```

Live since March 2026. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for what each
process does, what was tried, and why it ended up this way.

---

## Requirements

- **Python 3.10+** (3.11 recommended)
- A **Polygon wallet** with USDC.e for Polymarket and a small POL reserve for
  gas (~0.5 POL minimum, ~5 POL recommended). Funding flow:
  Coinbase → withdraw USDC to Polymarket UI → withdraw from Polymarket to your
  Polygon address (arrives as USDC.e, no swap needed).
- **API keys** for any LLM providers you want active in the swarm
  (Anthropic, OpenAI, DeepSeek, Grok, Qwen). Models auto-activate when their
  key is set — the swarm runs with whatever subset you have.
- **Binance Spot + USDM Futures** account for the funding-rate arb (optional).
- For 24/7 operation: a Linux server with `screen`. We use Oracle Cloud's
  Always-Free tier (1 OCPU / 1 GB RAM is sufficient). See
  [`docs/server_setup.md`](docs/server_setup.md).

---

## Quick start

```bash
# 1. Clone and enter
git clone <repo-url> poly-model
cd poly-model

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[ml]"

# 3. Configure
cp .env.example .env
$EDITOR .env                       # fill in KEY, FUNDER, API keys
```

The `.env` file is the only thing you need to edit to get started. Every
process reads it at startup. **Never commit `.env`** — it is in `.gitignore`.

### Paper trading first

By default the crypto loop runs in **paper mode** — no real orders are placed.
Verify the bot starts and produces signals before going live:

```bash
VIRTUAL_MODE=true python src/crypto_main.py
```

Watch `logs/crypto.log`. You should see lines like:
```
mode=VIRTUAL | interval=20s | entry_window=200-270s | threshold=0.02
Crypto flow loop started
```

### Going live (crypto)

Once you have validated paper performance (≥100 settled paper trades with
≥65% WR, positive EV per trade — see
[`EXPERIMENTS.md`](EXPERIMENTS.md#2-crypto-5-min-updown)):

1. Confirm `KEY` and `FUNDER` are set in `.env`.
2. Confirm the wallet has USDC.e and POL.
3. Confirm CLOB approvals are set:
   ```bash
   python scripts/setup_v2_approvals.py
   ```
4. Set `VIRTUAL_MODE=false` in `.env`.
5. Start under `screen`:
   ```bash
   screen -dmS crypto bash -c \
     'source .venv/bin/activate && python src/crypto_main.py >> logs/crypto.log 2>&1'
   ```

Sizing starts at `max_bet_abs: $2` (Phase 1) in
[`config/crypto_params.yaml`](config/crypto_params.yaml). Scale up only after
the gates in the same file are met.

---

## Running the other processes

```bash
# AI Swarm — long-form markets
screen -dmS swarm bash -c \
  'source .venv/bin/activate && python bot/main.py >> logs/swarm.log 2>&1'

# Funding-rate arb — paper first (virtual_mode: true in arb_params.yaml)
screen -dmS arb bash -c \
  'cd funding_arb && source ../.venv/bin/activate && python main.py'

# Dashboard — open http://<host>:8765 in a browser
screen -dmS dashboard bash -c \
  'source .venv/bin/activate && python src/dashboard.py --host 0.0.0.0 --port 8765'
```

Detach a screen with `Ctrl-A D`, reattach with `screen -r <name>`,
kill with `screen -S <name> -X quit`.

---

## Configuration

All tunables live in [`config/`](config/) as YAML — no code changes needed
to adjust thresholds, hours blocked, bet sizes, etc.

| File | Owns |
|---|---|
| [`crypto_params.yaml`](config/crypto_params.yaml) | 5-min loop: signal weights, band, hours blocked, sizing, circuit breaker |
| [`signal_params.yaml`](config/signal_params.yaml) | Alpha threshold, min confidence, max 24h volume |
| [`model_weights.yaml`](config/model_weights.yaml) | Ensemble weights per market category (currently LLM-dominant — see EXPERIMENTS.md) |
| [`risk_limits.yaml`](config/risk_limits.yaml) | Position/loss/Kelly caps |
| `funding_arb/config/arb_params.yaml` | Arb symbols, position size, hold time, alert thresholds |

Swarm tuning (gates, bet sizing) is in [`bot/config.py`](bot/config.py) and
[`bot/execution.py`](bot/execution.py).

---

## Repository layout

```
poly-model/
├── README.md                ← This file
├── EXPERIMENTS.md           ← What we tried, what we kept, why
├── .env.example             ← Template for credentials
│
├── src/
│   ├── crypto_main.py       ← Entry: 5-min crypto loop
│   ├── crypto/              ← Flow signal, CLOB feed, redemption
│   ├── dashboard.py         ← Web dashboard
│   └── ...
├── bot/                     ← AI Swarm
│   ├── main.py              ← Entry: swarm loop
│   ├── swarm/               ← Consensus, LLM calls, RAG search
│   └── execution.py         ← CLOB orders + filters
├── funding_arb/             ← Binance funding-rate arb (self-contained)
│   ├── main.py              ← Entry
│   └── src/                 ← futures/spot clients, risk, position manager
├── config/                  ← All tunable YAML
├── scripts/                 ← Backtests, retraining, on-chain redemption
├── data/                    ← State files, signal logs, model artifacts
├── logs/                    ← Per-process log files
└── docs/
    ├── server_setup.md      ← Oracle Cloud Always-Free setup
    └── deployment.md        ← Deployment notes
```

---

## Monitoring and alerts

- **ntfy.sh** push notifications. Set `NTFY_TOPIC=your-topic` in `.env`. The
  crypto loop fires on daily-loss circuit-breaker, abnormal per-position loss,
  and the midnight daily summary. The arb bot fires when a funding rate
  reaches 70% of the entry threshold.
- **Dashboard** at `http://<host>:8765` for live PnL, positions, and the
  per-model accuracy leaderboard.
- **Health check** — `python health_check.py` from the repo root prints live
  trade status, deployment info, and profit metrics.
- **PnL verify** — `python verify_pnl.py` cross-checks state-file PnL against
  the on-chain wallet balance.

---

## Safety notes

- **Start in paper mode.** Every loop has `VIRTUAL_MODE` / `virtual_mode` flags
  in its config. Validate the strategy on your own data before flipping.
- **Phase your sizing.** The scaling path is documented inline in
  [`config/crypto_params.yaml`](config/crypto_params.yaml). Phase 1 starts at
  $2 per trade.
- **Keep POL in the wallet.** CTF token redemption (turning won positions back
  into USDC.e) costs ~0.01 POL per transaction. The bot redeems every 12 hours
  automatically — running out of POL strands winnings.
- **`signature_type=0` (EOA) — do not change.** In
  `src/execution/order.py` and `src/crypto/execution.py`. `signature_type=1`
  is for proxy wallets; setting it on an EOA breaks every order.
- **USDC.e, not native USDC.** Polymarket settles in USDC.e
  (`0x2791Bca1...`). Withdrawals from the Polymarket UI arrive as USDC.e
  automatically; deposits from other sources may arrive as native USDC and
  require a swap (the QuickSwap V3 router was used in our setup).

---

## License

Add a `LICENSE` file at the repo root before publishing (MIT recommended).

---

## Disclaimer

Personal research project. Prediction-market trading is high-variance and the
strategies here have lost real money during development as well as made it.
Read [`EXPERIMENTS.md`](EXPERIMENTS.md) — including the killed-system
post-mortems — before reusing any of this code.
