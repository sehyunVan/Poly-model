# Notifications Setup

The bot sends three types of notifications every midnight UTC:

| Type | What it contains |
|---|---|
| **System warnings** | Risk limits hit, errors, crashes, halt_new_trades, large fills |
| **Daily report** | PnL, capital snapshot, positions summary |
| **Gate status** | Go-live readiness check (all 4 gates) |

---

## ⚠️ Discord Webhook Limitation (Oracle Cloud)

Discord webhooks return **error 1010 (Cloudflare block)** when called from
Oracle Cloud, AWS, or any major datacenter IP range. This is a permanent
Cloudflare policy — changing the webhook URL or code does not fix it.

**Recommended alternative: ntfy.sh** (free, works from any IP, mobile app available).

If you specifically need Discord, use the **Make.com relay** method at the bottom of this doc.

---

## Option A — ntfy.sh (Recommended)

ntfy.sh is a free push notification service. The Oracle server can reach it
without any restrictions. You view notifications in a browser or mobile app.

### Step 1 — Choose a private topic name

Pick any unique string, e.g. `polymarket-bot-abc123`.
Use something hard to guess (it's public-by-default on ntfy.sh).

### Step 2 — Add to .env on the server

```bash
ssh -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>
nano ~/poly-model/.env
```

Add this line:

```dotenv
NTFY_TOPIC=polymarket-bot-abc123
```

Save with **Ctrl+O**, **Enter**, **Ctrl+X**.

### Step 3 — Restart the bot

```bash
screen -S bot -X quit
screen -S bot
cd ~/poly-model && source .venv/bin/activate && python src/main.py
# Ctrl+A, D to detach
```

### Step 4 — Subscribe to notifications

**Browser:** open `https://ntfy.sh/polymarket-bot-abc123`
Click **Subscribe** to get live browser push notifications.

**Mobile (Android/iOS):**
1. Install the **ntfy** app ([Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) / [iOS](https://apps.apple.com/app/ntfy/id1625396347))
2. Tap **+** → enter your topic name → Subscribe

### Step 5 — Test

```bash
cd ~/poly-model && source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'src')
from monitoring.alerts import _send_ntfy
ok = _send_ntfy('polymarket-bot-abc123', 'Polymarket bot connected!', 'INFO')
print('Result:', ok)
"
```

You should immediately see the notification in your browser/phone.

---

## Option B — Discord via Make.com relay

This routes notifications through Make.com (free tier), which is not blocked by Cloudflare.

### Step 1 — Create a Discord webhook

1. Discord server → `#bot-alerts` channel → **Edit Channel**
2. **Integrations** → **Webhooks** → **New Webhook** → **Copy Webhook URL**

### Step 2 — Create a Make.com relay

1. Sign up free at [make.com](https://make.com)
2. **Create a new scenario**
3. Add module: **Webhooks → Custom webhook** → click **Add** → **Save** → copy the webhook URL (looks like `https://hook.eu2.make.com/...`)
4. Add second module: **HTTP → Make a request**
   - URL: your Discord webhook URL
   - Method: POST
   - Body type: Raw
   - Content type: `application/json`
   - Request content: `{"content": "{{1.body}}"}`
5. Connect the two modules → **Save** → **Turn on** the scenario

### Step 3 — Add the Make.com URL to .env

```bash
nano ~/poly-model/.env
```

Replace the Discord line:

```dotenv
DISCORD_WEBHOOK_URL=https://hook.eu2.make.com/YOUR_MAKE_WEBHOOK_ID
```

### Step 4 — Test

```bash
cd ~/poly-model && source .venv/bin/activate
python3 -c "
import sys, os; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv('.env')
from monitoring.alerts import _send_discord
ok = _send_discord(os.getenv('DISCORD_WEBHOOK_URL'), 'Polymarket bot connected via relay!')
print('Result:', ok)
"
```

---

## What each notification looks like

### ntfy.sh — System warning
```
Title:    ⚠️ WARNING
Message:  halt_new_trades triggered — daily loss limit exceeded (−5.1%)
Priority: default
Tag:      warning
```

### ntfy.sh — Daily report (midnight UTC)
```
Title:    ℹ️ INFO
Message:  Budget $1000 | Available $333.01
          Today: +$0.00 | Cumulative: +$88.60 (+8.9%)
          Hit rate: 60% (3W/2L) | Open: 15 markets (301 fills)
```

### ntfy.sh — Gate status (midnight UTC)
```
Title:    ⚠️ WARNING
Message:  ❌ Gate1: 3d/2mkt  ❌ Gate2: HR=60%(5pos)  ✅ Gate3: PnL=+88.60  ❌ Gate4: manual
          Status: 1/4 gates pass | Next: March 31 — end-of-month market settlements
```

When all gates pass:
```
Title:    ✅ INFO
Message:  ✅ Gate1: 35d/32mkt  ✅ Gate2: HR=54%(28pos)  ✅ Gate3: PnL=+142.30  ✅ Gate4: manual
          Status: READY | Next: Switch VIRTUAL_MODE=false
```

---

## Alert trigger points

| Event | Level | When |
|---|---|---|
| `halt_new_trades` activated | CRITICAL | Immediately |
| Large virtual fill (> 3% capital) | INFO | Immediately on fill |
| Model retraining complete | INFO | Midnight UTC |
| Model retraining failed | ERROR | Midnight UTC |
| Main loop crash | CRITICAL | Immediately |
| Daily report summary | INFO | Midnight UTC |
| Gate status check | INFO | Midnight UTC |

---

## Using both ntfy.sh and Discord together

Set both env vars — alerts will be sent to all configured channels:

```dotenv
NTFY_TOPIC=polymarket-bot-abc123
DISCORD_WEBHOOK_URL=https://hook.eu2.make.com/YOUR_MAKE_WEBHOOK_ID
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `HTTP 403 / error 1010` on Discord | Oracle Cloud IP is blocked by Cloudflare — use ntfy.sh or Make.com relay |
| ntfy test returns False | Check internet access: `curl https://ntfy.sh` from the server |
| No notification received on phone | Make sure you subscribed to the exact same topic name |
| Alerts work but daily report doesn't arrive | Report only sends at midnight UTC — use the test script in Step 5 to verify immediately |
