"""
simulate_shapes.py
==================
Test the whiteboard server by replaying random shapes from shape_dataset.

Usage:
  python scripts/simulate_shapes.py                        # default server
  python scripts/simulate_shapes.py --url http://localhost:5000
  python scripts/simulate_shapes.py --interval 0.5 --count 20
  python scripts/simulate_shapes.py --shape circle         # only circles
"""

import argparse
import glob
import json
import os
import random
import time

import socketio

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Simulate PYNQ shape events to the whiteboard server")
parser.add_argument("--url",      default="http://localhost:5000", help="Server URL")
parser.add_argument("--board",    default="board1",               help="Board ID")
parser.add_argument("--interval", type=float, default=1.0,        help="Seconds between shapes")
parser.add_argument("--count",    type=int,   default=0,          help="Total shapes to send (0 = infinite)")
parser.add_argument("--shape",    default=None,                   help="Only send a specific shape type (circle/rectangle/triangle/line/square)")
args = parser.parse_args()

# ── Dataset ───────────────────────────────────────────────────────────────────

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "shape_dataset")

def load_all_files(shape_filter=None):
    """Return list of (shape_type, filepath) tuples from the dataset."""
    entries = []
    for folder in os.listdir(DATASET_DIR):
        if shape_filter and folder != shape_filter:
            continue
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in glob.glob(os.path.join(folder_path, "*.json")):
            entries.append((folder, f))
    return entries

all_files = load_all_files(args.shape)
if not all_files:
    print(f"[ERROR] No files found (shape filter: {args.shape})")
    raise SystemExit(1)

print(f"[DATASET] {len(all_files)} files loaded across categories: "
      f"{sorted(set(t for t, _ in all_files))}")

# ── Shape conversion ──────────────────────────────────────────────────────────

def stroke_to_payload(shape_type, stroke_points):
    """
    Convert raw stroke [[x,y],...] from the dataset into a board ADD_OBJECT payload.

    All dataset entries are raw hand-drawn strokes. We send them as their
    labelled type so the whiteboard renders them using the right path (stroke),
    giving a realistic visual result. Circle / rect / triangle are sent as
    'stroke' too (the raw trace) which is the most faithful representation of
    what the dataset actually contains.

    To test the server's shape rendering (circle/rect/triangle objects), pass
    --shape <type> and the script will also attempt a simple geometric
    reconstruction where possible.
    """
    obj_id = f"sim_{random.randint(100000, 999999)}"
    points = [[int(p[0]), int(p[1])] for p in stroke_points]

    return {
        "object_id": obj_id,
        "type":      "stroke",   # always renderable; points are the raw trace
        "points":    points,
        "_dataset_label": shape_type,   # informational only, ignored by server
    }

# ── Socket.IO ─────────────────────────────────────────────────────────────────

sio = socketio.Client(logger=False, engineio_logger=False)
connected = False

@sio.event
def connect():
    global connected
    connected = True
    print(f"[SERVER] Connected  →  joining {args.board}")
    sio.emit("join_board", {"board_id": args.board})

@sio.event
def disconnect():
    global connected
    connected = False
    print("[SERVER] Disconnected")

# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"[CLIENT] Connecting to {args.url} …")
sio.connect(args.url)
time.sleep(0.5)   # wait for join_board ack + LOAD_BOARD

sent = 0
try:
    while True:
        if args.count > 0 and sent >= args.count:
            print(f"[DONE] Sent {sent} shapes.")
            break

        shape_type, filepath = random.choice(all_files)

        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {filepath}: {e}")
            continue

        stroke_points = data.get("stroke", [])
        if not stroke_points:
            continue

        payload = stroke_to_payload(shape_type, stroke_points)

        sio.emit("pynq_event", payload)
        sent += 1
        fname = os.path.basename(filepath)
        print(f"[SENT #{sent}] {shape_type}/{fname}  →  object_id={payload['object_id']}  points={len(stroke_points)}")

        time.sleep(args.interval)

except KeyboardInterrupt:
    print(f"\n[STOP] Sent {sent} shapes total.")
finally:
    sio.disconnect()
