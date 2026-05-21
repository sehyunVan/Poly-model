#!/usr/bin/env bash
# deploy.sh — Upload changed source files to the Oracle Cloud server.
#
# USAGE
#   bash scripts/deploy.sh [file1] [file2] ...
#
#   With no arguments: prints usage and exits (never uploads blindly).
#   With file arguments: uploads each file individually to its exact remote path.
#
# WHY NOT scp dir/*?
#   SCP to a directory is ambiguous when the remote path has the same name as
#   a file.  On 2026-03-xx, `scp src/crypto/ ubuntu@...:~/poly-model/src/`
#   overwrote `data/schemas.py` because the destination resolved incorrectly.
#   This script uses one SCP command per file with an explicit destination path
#   to make the upload unambiguous and auditable.
#
# AFTER DEPLOY
#   Run verify_server.py to confirm the server is healthy:
#     ssh -i "$KEY_PATH" ubuntu@"$HOST" "cd ~/poly-model && source .venv/bin/activate && python scripts/verify_server.py"
#
# RESTARTING BOTS
#   Use two SEPARATE ssh commands — do NOT chain with &&.
#   screen -X quit returns exit 1 when no session exists, which aborts &&-chained commands.
#
#     # Restart crypto:
#     ssh -i "$KEY_PATH" ubuntu@"$HOST" "screen -S crypto -X quit"
#     sleep 3
#     ssh -i "$KEY_PATH" ubuntu@"$HOST" "cd ~/poly-model && screen -dmS crypto bash -c 'source .venv/bin/activate && python src/crypto_main.py >> logs/crypto.log 2>&1'"
#
#     # Restart swarm:
#     ssh -i "$KEY_PATH" ubuntu@"$HOST" "screen -S swarm -X quit"
#     sleep 3
#     ssh -i "$KEY_PATH" ubuntu@"$HOST" "cd ~/poly-model && screen -dmS swarm bash -c 'source .venv/bin/activate && python bot/main.py >> logs/swarm.log 2>&1'"

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
KEY_PATH="${SSH_KEY:?set SSH_KEY env var to your private key path}"
HOST="${HOST:?set HOST env var to your server IP or hostname}"
REMOTE_USER="ubuntu"
REMOTE_ROOT="/home/ubuntu/poly-model"

# ── Usage guard ───────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo ""
    echo "deploy.sh — upload files to the Oracle Cloud server"
    echo ""
    echo "Usage:"
    echo "  bash scripts/deploy.sh <file1> [file2] ..."
    echo ""
    echo "Examples:"
    echo "  bash scripts/deploy.sh src/crypto/loop.py"
    echo "  bash scripts/deploy.sh src/infra/types.py src/infra/http_client.py"
    echo ""
    echo "Each file is uploaded to the same relative path on the server."
    echo "The script will show you the exact scp command before running it."
    echo ""
    exit 1
fi

# ── Upload each file individually ─────────────────────────────────────────────
UPLOADED=0
FAILED=0

for LOCAL_FILE in "$@"; do
    # Strip leading ./ if present
    LOCAL_FILE="${LOCAL_FILE#./}"

    if [[ ! -f "$LOCAL_FILE" ]]; then
        echo "[ERROR] File not found: $LOCAL_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    REMOTE_FILE="$REMOTE_ROOT/$LOCAL_FILE"
    REMOTE_DIR="$(dirname "$REMOTE_FILE")"

    echo ""
    echo "── Uploading: $LOCAL_FILE ──────────────────────────────────"
    echo "    local:  $LOCAL_FILE"
    echo "    remote: $REMOTE_USER@$HOST:$REMOTE_FILE"

    # Ensure remote directory exists
    ssh -i "$KEY_PATH" "$REMOTE_USER@$HOST" "mkdir -p '$REMOTE_DIR'"

    # Upload to the exact file path (never to a directory)
    scp -i "$KEY_PATH" "$LOCAL_FILE" "$REMOTE_USER@$HOST:$REMOTE_FILE"

    echo "    ✓ uploaded"
    UPLOADED=$((UPLOADED + 1))
done

echo ""
echo "────────────────────────────────────────────────────────────"
echo "Uploaded: $UPLOADED file(s)   Failed: $FAILED file(s)"

if [[ $FAILED -gt 0 ]]; then
    echo "[WARN] Some files failed to upload — check errors above."
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Verify server health:"
echo "     ssh -i \"$KEY_PATH\" $REMOTE_USER@$HOST \"cd ~/poly-model && source .venv/bin/activate && python scripts/verify_server.py\""
echo ""
echo "  2. If you changed src/crypto/*.py — restart crypto loop (two separate commands):"
echo "     ssh -i \"$KEY_PATH\" $REMOTE_USER@$HOST \"screen -S crypto -X quit\""
echo "     sleep 3"
echo "     ssh -i \"$KEY_PATH\" $REMOTE_USER@$HOST \"cd ~/poly-model && screen -dmS crypto bash -c 'source .venv/bin/activate && python src/crypto_main.py >> logs/crypto.log 2>&1'\""
echo ""
echo "  3. If you changed bot/*.py — restart swarm loop (two separate commands):"
echo "     ssh -i \"$KEY_PATH\" $REMOTE_USER@$HOST \"screen -S swarm -X quit\""
echo "     sleep 3"
echo "     ssh -i \"$KEY_PATH\" $REMOTE_USER@$HOST \"cd ~/poly-model && screen -dmS swarm bash -c 'source .venv/bin/activate && python bot/main.py >> logs/swarm.log 2>&1'\""
