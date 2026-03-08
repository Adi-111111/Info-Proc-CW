"""
PYNQ-side shape recogniser + sender

Runs ON the PYNQ board.

Flow:
  capture_client (PC) --UDP:5005--> this script
      → hw_pipeline shape recognition
      → UDP:5006 --> pynq_client (PC)
"""

import socket
import json
import math
import sys
import os

# Load hw_pipeline from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_pipeline as hw

# =========================
# CONFIG
# =========================

LISTEN_PORT  = 5005          # receives strokes from capture_client
PC_IP        = "192.168.2.1" # IP of the PC running pynq_client.py
PC_PORT      = 5006          # pynq_client.py listens here

BUFFER_SIZE  = 65535

# Recognition thresholds (mirror capture_client/test.py)
CLOSE_THRESH_PX   = 60.0
CIRCLE_REL_TOL    = 0.18
RECT_ANGLE_TOL    = 50.0
RESAMPLE_STEP     = 8.0
RDP_EPS           = 12.0

# =========================
# HELPERS
# =========================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def recognise(points):
    """
    Run the hw_pipeline recognition chain.
    Returns a dict ready to JSON-encode and send.
    """
    pts = hw.resample_polyline(points, step=RESAMPLE_STEP)

    if len(pts) < 2:
        return None

    closed = (len(pts) >= 10 and dist(pts[0], pts[-1]) < CLOSE_THRESH_PX)
    if closed and pts[0] != pts[-1]:
        pts = pts + [pts[0]]

    # ---- circle ----
    if closed and len(pts) >= 25:
        result = hw.kasa_circle_fit(pts[:-1])
        if result is not None:
            cx, cy, r = result
            xs = [p[0] for p in pts[:-1]]
            ys = [p[1] for p in pts[:-1]]
            scale = max(20.0, max(max(xs) - min(xs), max(ys) - min(ys)))
            if 15 <= r <= 0.9 * scale:
                # compute rmse to check CIRCLE_REL_TOL
                import numpy as np
                arr = [(p[0], p[1]) for p in pts[:-1]]
                ds = [math.hypot(p[0] - cx, p[1] - cy) for p in arr]
                rmse = math.sqrt(sum((d - r) ** 2 for d in ds) / len(ds))
                if rmse / max(1.0, r) <= CIRCLE_REL_TOL:
                    return {
                        "type": "circle",
                        "cx": round(cx, 2),
                        "cy": round(cy, 2),
                        "r":  round(r,  2),
                    }

    # ---- rectangle ----
    if closed:
        simplified = hw.rdp_simplify(pts, epsilon=RDP_EPS)
        rect = hw.try_rectangle(simplified, angle_tol=RECT_ANGLE_TOL)
        if rect is not None:
            return {
                "type":    "rectangle",
                "corners": [[round(p[0], 2), round(p[1], 2)] for p in rect],
            }

    # ---- fallback: stroke ----
    return {
        "type":   "stroke",
        "points": [[round(p[0], 2), round(p[1], 2)] for p in points],
    }

# =========================
# SOCKETS
# =========================

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(("0.0.0.0", LISTEN_PORT))

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pc_addr   = (PC_IP, PC_PORT)

print(f"[hw_pipeline] loading overlay...")
hw.load_overlay()
print(f"[hw_pipeline] ready")

print(f"[UDP] listening on 0.0.0.0:{LISTEN_PORT}")
print(f"[UDP] will forward shapes to {PC_IP}:{PC_PORT}")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    try:
        while True:
            print("\n[UDP] waiting for stroke...")
            data, addr = recv_sock.recvfrom(BUFFER_SIZE)
            print(f"[UDP] packet from {addr}")

            try:
                msg = json.loads(data.decode())
            except Exception as e:
                print(f"[ERROR] JSON decode failed: {e}")
                continue

            # expect {"stroke": [[x,y], ...]}
            if "stroke" not in msg:
                print("[ERROR] missing 'stroke' key, skipping")
                continue

            points = [(float(p[0]), float(p[1])) for p in msg["stroke"]]
            print(f"[RECOG] received stroke with {len(points)} points")

            shape = recognise(points)
            if shape is None:
                print("[RECOG] recognition returned nothing, skipping")
                continue

            print(f"[RECOG] recognised as: {shape['type']}")

            payload = json.dumps(shape).encode("utf-8")
            send_sock.sendto(payload, pc_addr)
            print(f"[UDP] sent shape to PC ({len(payload)} bytes)")

    except KeyboardInterrupt:
        print("\n[pynq_shape_sender] shutting down")
    finally:
        recv_sock.close()
        send_sock.close()
