#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel)"
cd "$REPO_DIR"

EC2_USER="ubuntu"
EC2_HOST="13.40.61.155"
SSH_KEY="$REPO_DIR/apps/whiteboard_server/Whiteboard-final.pem"
REMOTE_HOME="/home/ubuntu"
REMOTE_LOG="$REMOTE_HOME/server.log"
REMOTE_PID="$REMOTE_HOME/server.pid"
REMOTE_WEB_LOG="$REMOTE_HOME/web_client.log"
REMOTE_WEB_PID="$REMOTE_HOME/web_client.pid"

LOCAL_SERVER="$REPO_DIR/apps/whiteboard_server/server.py"
LOCAL_HTML="$REPO_DIR/apps/web_client/whiteboard.html"

#scp -i "$SSH_KEY" "$LOCAL_SERVER" "$EC2_USER@$EC2_HOST:~"
#scp -i "$SSH_KEY" "$LOCAL_HTML" "$EC2_USER@$EC2_HOST:~"

ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "
set -e
cd ~
source venv/bin/activate

fuser -k -n tcp 5000 >/dev/null 2>&1 || true

nohup python3 server.py > '$REMOTE_LOG' 2>&1 &
SERVER_PID=\$!
echo \$SERVER_PID > '$REMOTE_PID'

if kill -0 \"\$SERVER_PID\" 2>/dev/null; then
    echo 'Server is running on $EC2_HOST:5000'
else
    echo 'Server failed to start'
    tail -n 20 '$REMOTE_LOG'
    exit 1
fi

fuser -k -n tcp 8000 >/dev/null 2>&1 || true

nohup python3 -m http.server 8000 > '$REMOTE_WEB_LOG' 2>71 &
WEB_PID=\$!
echo \$WEB_PID > '$REMOTE_WEB_PID'

if kill -0 \"\$WEB_PID\" 2>/dev/null; then
    echo 'whiteboard.html is running on $EC2_HOST:8000/whiteboard.html'
else
    echo 'whiteboard.html failed to start'
    tail -n 20 '$REMOTE_WEB_LOG'
    exit 1
fi
"
#exec ssh -tt -i "$SSH_KEY" "$EC2_USER@$EC2_HOST"

