# Always-On Server Setup (Oracle Cloud Free Tier)

Oracle Cloud gives a **permanently free** Linux VM — no credit card expiry, no trial period.
1 OCPU + 1 GB RAM is enough to run this bot 24/7.

**Your server info:**
- Public IP: `<YOUR_SERVER_IP>`
- SSH key: `<YOUR_SSH_KEY_PATH>`

---

## Step 1 — Create Oracle Cloud account

1. Go to **https://cloud.oracle.com** → Sign Up
2. Enter name, email, country
3. Credit card is required for identity verification — **you will NOT be charged** as long as you only use Always Free resources
4. Choose a Home Region close to you (e.g., `ap-tokyo-1`, `us-ashburn-1`) — **cannot change later**
5. Complete sign-up and wait for the confirmation email (~5 minutes)

---

## Step 2 — Create the VM

1. Log in → top-left menu → **Compute** → **Instances** → **Create Instance**
2. Settings:
   - Name: `polymarket-bot`
   - Image: **Ubuntu 22.04** (Canonical) — click "Change Image" if needed
   - Shape: **VM.Standard.E2.1.Micro** — this is the Always Free shape
   - Networking: leave defaults (a public subnet will be created)
   - SSH keys: click **Generate a key pair for me** → **Download private key** (`*.key` file)
     - Save this file — you need it to connect
3. Click **Create**
4. Wait ~2 minutes until the status shows **Running**
5. Copy the **Public IP address** (shown on the instance detail page)

---

## Step 3 — Connect via SSH (from your laptop)

On Windows, use **PowerShell** or **Git Bash**:

```powershell
# First time only: fix key permissions so SSH accepts the file
icacls "<YOUR_SSH_KEY_PATH>" /inheritance:r /grant:r "%USERNAME%:R"

# Connect
ssh -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>
```

You should see the Ubuntu command prompt: `ubuntu@polymarket-bot:~$`

---

## Step 4 — Install Python and dependencies on the server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and tools
sudo apt install -y python3.11 python3.11-venv python3-pip git screen

# Confirm
python3.11 --version
```

---

## Step 5 — Upload the project to the server

In a **new PowerShell window** on your laptop (keep the SSH window open separately):

```powershell
# Upload the entire project
scp -i "<YOUR_SSH_KEY_PATH>" -r "<YOUR_LOCAL_REPO_PATH>" ubuntu@<YOUR_SERVER_IP>:~/poly-model
```

This takes 1–3 minutes depending on your internet speed.

---

## Step 6 — Set up the Python environment on the server

Back in your SSH window:

```bash
cd ~/poly-model

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install xgboost lightgbm   # optional ML extras
```

---

## Step 7 — Configure .env on the server

```powershell
# In PowerShell on your laptop — copy your .env to the server
scp -i "<YOUR_SSH_KEY_PATH>" "<YOUR_LOCAL_REPO_PATH>\.env" ubuntu@<YOUR_SERVER_IP>:~/poly-model/.env
```

Verify it arrived:
```bash
# On the server
cat ~/poly-model/.env
```

---

## Step 8 — Run the bot with screen (survives SSH disconnect)

`screen` keeps the process running after you close the SSH window.

```bash
cd ~/poly-model
source .venv/bin/activate

# Start a named screen session for the bot
screen -S bot
python src/main.py
# Press Ctrl+A, then D to detach (bot keeps running in background)

# Start a second screen session for the dashboard
screen -S dashboard
python src/dashboard.py --host 0.0.0.0 --port 8765
# Press Ctrl+A, then D to detach
```

**Useful screen commands:**
```bash
screen -ls                  # list all sessions
screen -r bot               # reattach to bot session
screen -r dashboard         # reattach to dashboard session
```

---

## Step 9 — Open firewall port for the dashboard

By default Oracle Cloud blocks all inbound ports. To access the dashboard from your browser:

1. Oracle Cloud console → **Networking** → **Virtual Cloud Networks** → click your VCN
2. Click **Security Lists** → **Default Security List**
3. **Add Ingress Rule**:
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination Port Range: `8765`
4. Also allow it at OS level:
   ```bash
   sudo iptables -I INPUT -p tcp --dport 8765 -j ACCEPT
   sudo netfilter-persistent save   # if installed
   ```

Then open in your browser:
```
http://<YOUR_SERVER_IP>:8765
```

---

## Step 10 — Auto-restart on reboot (cron)

```bash
crontab -e
```

Add this line at the bottom:

```cron
@reboot cd /home/ubuntu/poly-model && /home/ubuntu/poly-model/.venv/bin/python src/main.py >> logs/main.log 2>&1 &
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X` in nano). The bot will now start automatically whenever the server reboots.

---

## Syncing state back to your laptop (optional)

Pull the latest `virtual_state.json` to view locally:

```powershell
scp -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>:~/poly-model/data/virtual_state.json "<YOUR_LOCAL_REPO_PATH>\data\virtual_state.json"
```

Pull daily reports:
```powershell
scp -i "<YOUR_SSH_KEY_PATH>" -r ubuntu@<YOUR_SERVER_IP>:~/poly-model/logs/virtual_reports "<YOUR_LOCAL_REPO_PATH>\logs\"
```

---

## Pushing code updates to the server

After editing files locally, upload and restart:

```powershell
# Upload changed source files
scp -i "<YOUR_SSH_KEY_PATH>" -r "<YOUR_LOCAL_REPO_PATH>\src" ubuntu@<YOUR_SERVER_IP>:~/poly-model/

# Upload changed config files
scp -i "<YOUR_SSH_KEY_PATH>" -r "<YOUR_LOCAL_REPO_PATH>\config" ubuntu@<YOUR_SERVER_IP>:~/poly-model/
```

Then SSH in and restart:
```bash
ssh -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>
screen -r bot
# Ctrl+C to stop
python src/main.py
# Ctrl+A, D to detach
```

---

## Quick Command Reference

All commands you need for daily operation. Copy-paste ready.

### Connect to server

```powershell
# From your laptop (PowerShell)
ssh -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>
```

---

### Bot control (run these on the server after SSH)

```bash
# ── Check status ──────────────────────────────────────────────────────────────

# Are any bots running?
screen -ls

# How many python processes are running?
pgrep -a python

# Watch live logs (Ctrl+C to stop watching)
tail -f ~/poly-model/logs/main.log


# ── Start bot ─────────────────────────────────────────────────────────────────

cd ~/poly-model && source .venv/bin/activate
screen -S bot
python src/main.py
# Press Ctrl+A, then D  →  detach (bot keeps running)


# ── Stop bot (clean) ──────────────────────────────────────────────────────────

screen -r bot           # reattach first
# Press Ctrl+C           →  bot saves state and exits gracefully


# ── Stop bot (without reattaching) ───────────────────────────────────────────

screen -S bot -X quit


# ── Restart bot ───────────────────────────────────────────────────────────────

screen -S bot -X quit                                   # stop
screen -S bot                                           # new session
cd ~/poly-model && source .venv/bin/activate && python src/main.py
# Ctrl+A, D to detach


# ── Kill ALL duplicate bots (emergency) ──────────────────────────────────────

screen -ls | grep bot | awk '{print $1}' | xargs -I{} screen -S {} -X quit
# then start one clean bot (see Start bot above)


# ── Start dashboard ───────────────────────────────────────────────────────────

screen -S dashboard
cd ~/poly-model && source .venv/bin/activate
python src/dashboard.py --host 0.0.0.0 --port 8765
# Ctrl+A, D to detach
# Open in browser: http://<YOUR_SERVER_IP>:8765


# ── Stop dashboard ────────────────────────────────────────────────────────────

screen -S dashboard -X quit
```

---

### Reset virtual state (fresh start)

```bash
# On the server — wipes all positions and resets budget to $1000
echo '{"initial_budget": 1000.0, "available_usdc": 1000.0, "positions": [], "closed_positions": [], "pnl_history": [], "start_date": null, "last_updated": null}' > ~/poly-model/data/virtual_state.json
```

---

### Push code updates from laptop → server

```powershell
# Upload source code
scp -i "<YOUR_SSH_KEY_PATH>" -r "<YOUR_LOCAL_REPO_PATH>\src" ubuntu@<YOUR_SERVER_IP>:~/poly-model/

# Upload config files only (no restart needed for yaml — restart IS needed)
scp -i "<YOUR_SSH_KEY_PATH>" -r "<YOUR_LOCAL_REPO_PATH>\config" ubuntu@<YOUR_SERVER_IP>:~/poly-model/
```

Then restart the bot on the server (see Restart bot above).

---

### Pull data from server → laptop

```powershell
# Pull virtual state (to view locally)
scp -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>:~/poly-model/data/virtual_state.json "<YOUR_LOCAL_REPO_PATH>\data\virtual_state.json"

# Pull daily reports
scp -i "<YOUR_SSH_KEY_PATH>" -r ubuntu@<YOUR_SERVER_IP>:~/poly-model/logs/virtual_reports "<YOUR_LOCAL_REPO_PATH>\logs\"

# Pull main log
scp -i "<YOUR_SSH_KEY_PATH>" ubuntu@<YOUR_SERVER_IP>:~/poly-model/logs/main.log "<YOUR_LOCAL_REPO_PATH>\logs\main_server.log"
```

---

## Summary

| | |
|---|---|
| Server IP | `<YOUR_SERVER_IP>` |
| SSH key | `<YOUR_SSH_KEY_PATH>` |
| Dashboard URL | `http://<YOUR_SERVER_IP>:8765` |
| Running cost | $0.00 / month |
| Bot uptime | 24/7, survives laptop off |
