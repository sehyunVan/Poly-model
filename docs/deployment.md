# Deployment Guide

## Overview

The bot runs as a plain Python process (`python src/main.py`).
No web server or MCP server is required for automated trading.

```
python src/main.py          ← trading loop (5-min cycle)
python src/dashboard.py     ← live dashboard  (optional, port 8765)
```

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python --version` |
| Dependencies installed | `pip install -e .` from project root |
| `.env` file configured | See section 2 |
| Polymarket account | Required for real-mode only |

---

## 2. Environment variables (`.env`)

Copy the example and fill in your values:

```dotenv
# ── Trading mode ──────────────────────────────────────────────────────────────
VIRTUAL_MODE=true           # true = paper trading, false = live trading
VIRTUAL_BUDGET=1000.0       # starting virtual budget (USDC)
VIRTUAL_STATE_PATH=data/virtual_state.json

# ── Polymarket credentials (live mode only) ───────────────────────────────────
KEY=0x...                   # your Polymarket private key
FUNDER=0x...                # your wallet address (funder)
CLOB_HOST=https://clob.polymarket.com

# ── Alerts (optional) ─────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
```

> **Never commit `.env` to git.** It is already listed in `.gitignore`.

---

## 3. Running manually

```bat
:: Virtual (paper trading)
python src\main.py

:: Live trading
:: Set VIRTUAL_MODE=false in .env first, then:
python src\main.py

:: One cycle only (for testing)
python src\main.py --once

:: With debug logging
python src\main.py --log-level DEBUG

:: Dashboard (separate terminal)
python src\dashboard.py --port 8765
```

---

## 4. Windows Task Scheduler (recommended for unattended operation)

This keeps the bot running after you close the terminal and restarts it automatically on failure.

### Step-by-step

1. Open **Task Scheduler** (`taskschd.msc`)
2. Click **Create Task** (not "Basic Task")

**General tab**
- Name: `Polymarket Bot`
- Run whether user is logged on or not: ✓
- Run with highest privileges: ✓

**Triggers tab** → New
- Begin the task: `At startup`
- Delay task for: `1 minute` (give network time to come up)

**Actions tab** → New
- Action: `Start a program`
- Program/script: `<YOUR_LOCAL_REPO_PATH>\start_bot.bat`
- Start in: `<YOUR_LOCAL_REPO_PATH>`

**Settings tab**
- If the task fails, restart every: `5 minutes`, up to `3 times`
- If the task is already running: `Do not start a new instance`

3. Click **OK** and enter your Windows password when prompted.

### Verify it's running

```bat
:: Check Task Scheduler status
schtasks /query /tn "Polymarket Bot" /fo LIST

:: Check the launcher log
type logs\launcher.log

:: Check the main bot log
type logs\main.log
```

---

## 5. PM2 (alternative — requires Node.js)

If you prefer PM2 for its process dashboard:

```bash
npm install -g pm2
pm2 start start_bot.bat --name polymarket-bot --restart-delay 5000
pm2 save
pm2 startup          # configure auto-start on boot

# Monitor
pm2 status
pm2 logs polymarket-bot
```

---

## 6. Log files

All logs are in `logs/` with automatic rotation (10 MB per file, 7 backups kept):

| File | Contents |
|---|---|
| `logs/main.log` | Main trading loop — cycles, signals, risk decisions |
| `logs/execution.log` | Order placement and fill status (real mode) |
| `logs/launcher.log` | start_bot.bat launch/exit events |
| `logs/reports/YYYY-MM-DD.md` | Daily trading reports |
| `logs/virtual_reports/YYYY-MM-DD.md` | Virtual mode daily reports |

Log rotation means the directory will never grow beyond ~140 MB total (`10 MB × 7 backups × 2 files`).

---

## 7. Switching from virtual to live mode

1. Confirm you have acceptable virtual performance (hit rate > 50%, no large drawdowns)
2. Deposit USDC into your Polymarket account
3. Edit `.env`:
   ```dotenv
   VIRTUAL_MODE=false
   KEY=0x<your_private_key>
   FUNDER=0x<your_wallet_address>
   ```
4. Restart the bot: `start_bot.bat` or restart the Task Scheduler task
5. The dashboard at `http://localhost:8765` will automatically show `REAL` mode

---

## 8. Stopping the bot

```bat
:: If running in a terminal — Ctrl+C

:: If running via Task Scheduler:
schtasks /end /tn "Polymarket Bot"

:: If running via PM2:
pm2 stop polymarket-bot
```
