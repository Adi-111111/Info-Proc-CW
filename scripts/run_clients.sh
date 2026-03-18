#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel)"
cd "$REPO_DIR"

BRIDGE_SCRIPT="$REPO_DIR/apps/pynq_bridge/pynq_client.py"
CAPTURE_SCRIPT="$REPO_DIR/apps/capture_client/test.py"

if [[ -f "$REPO_DIR/venv/bin/activate" ]]; then
    # Reuse the project virtualenv when it exists.
    source "$REPO_DIR/venv/bin/activate"
fi

cleanup() {
    local exit_code=$?

    for pid in "${CAPTURE_PID:-}" "${BRIDGE_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    wait "${CAPTURE_PID:-}" "${BRIDGE_PID:-}" 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting PYNQ bridge..."
python3 "$BRIDGE_SCRIPT" &
BRIDGE_PID=$!

echo "Starting capture client..."
python3 "$CAPTURE_SCRIPT" &
CAPTURE_PID=$!

echo "Bridge PID: $BRIDGE_PID"
echo "Capture PID: $CAPTURE_PID"
echo "Press Ctrl+C to stop both processes."

wait n "$BRIDGE_PID" "$CAPTURE_PID"
