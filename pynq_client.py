"""
PYNQ Socket.IO client — connects to the whiteboard server and
periodically sends ADD_OBJECT events.

Usage:
    python3 pynq_client.py
    python3 pynq_client.py --url http://192.168.1.50:5000
"""

import argparse
import logging
import random
import time
import uuid

import socketio

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SERVER_URL    = "http://192.168.1.50:5000"   # override with --url
BOARD_ID      = "board1"
SEND_INTERVAL = 2.0          # seconds between dummy transmissions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pynq_client")

# ─────────────────────────────────────────────
# Socket.IO client
#   reconnection=True       – auto-reconnect on drop
#   reconnection_attempts=0 – retry forever (good for embedded)
#   reconnection_delay / _max – back-off window (seconds)
# ─────────────────────────────────────────────
sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=0,
    reconnection_delay=2,
    reconnection_delay_max=30,
    logger=False,
    engineio_logger=False,
)


# ─────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────

@sio.event
def connect():
    log.info("Connected (sid=%s) — joining board '%s'", sio.sid, BOARD_ID)
    sio.emit("join_board", {"board_id": BOARD_ID})


@sio.event
def disconnect():
    log.warning("Disconnected — client will auto-reconnect")


@sio.event
def connect_error(data):
    log.error("Connection error: %s", data)


@sio.on("LOAD_BOARD")
def on_load_board(data):
    log.info("LOAD_BOARD received: %d object(s) already on board", len(data))
    for obj_id, obj in data.items():
        log.info("  existing object  id=%s  type=%s", obj_id, obj.get("type"))


@sio.on("board_event")
def on_board_event(msg):
    event   = msg.get("event")
    payload = msg.get("payload", {})
    log.info("board_event  %-15s  id=%-12s  type=%s",
             event,
             payload.get("object_id", "?"),
             payload.get("type", "?"))


# ─────────────────────────────────────────────
# Payload builders
# ─────────────────────────────────────────────

def make_stroke_payload():
    """Dummy two-point polyline — proves the pipeline end-to-end."""
    return {
        "object_id": "pynq_" + uuid.uuid4().hex[:8],
        "type": "polyline",
        "points": [
            [random.randint(10, 900), random.randint(10, 550)],
            [random.randint(10, 900), random.randint(10, 550)],
        ],
    }


# ── Replace make_stroke_payload() with one of these when ready ──────────────

def make_polyline_payload(points: list):
    """
    Send a multi-point polyline from real data.
    points: list of [x, y] pairs, e.g. from shape recognition.

    Example:
        points = [[100, 200], [150, 300], [250, 280]]
        sio.emit("board_event", {"event": "ADD_OBJECT",
                                 "payload": make_polyline_payload(points)})
    """
    return {
        "object_id": "pynq_" + uuid.uuid4().hex[:8],
        "type": "polyline",
        "points": [[float(p[0]), float(p[1])] for p in points],
    }


def make_rectangle_payload(corners: list):
    """
    Send a (possibly rotated) rectangle from real data.
    corners: list of four [x, y] pairs in order.

    Example (mediapipe output):
        corners = [[515, 238], [686, 172], [762, 317], [572, 397]]
        sio.emit("board_event", {"event": "ADD_OBJECT",
                                 "payload": make_rectangle_payload(corners)})
    """
    return {
        "object_id": "pynq_" + uuid.uuid4().hex[:8],
        "type": "rectangle",
        "corners": [[float(c[0]), float(c[1])] for c in corners],
    }


def send_shape_event(shape_type: str, params: dict, shape_id: str = None):
    """
    Alternative path: emit a raw mediapipe-style shape_event.
    The server normalises it and re-broadcasts as ADD_OBJECT.

    shape_type: "polyline" or "rectangle"
    params:
        polyline   -> {"points": [[x, y], ...]}
        rectangle  -> {"corners": [[x, y], [x, y], [x, y], [x, y]]}

    Use this when feeding output straight from test.py / mediapipe.
    """
    sio.emit("shape_event", {
        "id":        shape_id or ("pynq_" + uuid.uuid4().hex[:8]),
        "timestamp": time.time(),
        "type":      shape_type,
        "params":    params,
    })


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main(url: str):
    log.info("Connecting to %s", url)

    # connect() blocks until the handshake succeeds (or raises on hard failure)
    try:
        sio.connect(url, wait_timeout=15)
    except socketio.exceptions.ConnectionError as exc:
        log.error("Initial connection failed: %s — will keep retrying in loop", exc)

    try:
        while True:
            if sio.connected:
                payload = make_stroke_payload()
                sio.emit("board_event", {
                    "event":   "ADD_OBJECT",
                    "payload": payload,
                })
                log.info("Sent ADD_OBJECT  id=%s", payload["object_id"])
            else:
                log.debug("Not connected — waiting for reconnect")

            time.sleep(SEND_INTERVAL)

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        if sio.connected:
            sio.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SERVER_URL,
                        help="Server URL (default: %(default)s)")
    args = parser.parse_args()
    main(args.url)
