#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel)"
cd "$REPO_DIR"

EC2_USER="ubuntu"
EC2_HOST="13.40.61.155"
SSH_KEY="$REPO_DIR/apps/whiteboard_server/Whiteboard-final.pem"

ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "
    fuser -k -n tcp 5000 >/dev/null 2>&1 || true
    fuser -k -n tcp 8000 >/dev/null 2>&1 || true
    rm -f /home/ubuntu/server.pid /home/ubuntu/web_client.pid
    echo 'Stopped services on ports 5000 and 8000'
"
