import cv2
import numpy as np
import time
import os
import json
import socket
from pathlib import Path
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(str(Path(__file__).resolve().parents[1] / "shape_classifier"))
from preprocess import preprocess_to_vector

# =========================================================
# PYNQ UDP CONNECTION
# =========================================================
PYNQ_IP = "192.168.2.99"
PYNQ_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pynq_addr = (PYNQ_IP, PYNQ_PORT)

# =========================================================
# CONFIG
# =========================================================
SMOOTH_ALPHA = 0.25
PINCH_THRESH_PX = 80
MIN_POINTS = 5

# Quantization rule that matched your FPGA
COORD_SCALE = 127
GEOM_SCALE = 32

# =========================================================
# HELPERS
# =========================================================
def quantize_features(features):
    features = np.array(features, dtype=float)

    # first 64 = coordinates
    features[:64] = np.round(features[:64] * COORD_SCALE)

    # last 6 = geometry
    features[64:] = np.round(features[64:] * GEOM_SCALE)

    features = np.clip(features, -128, 127).astype(np.int8)
    return features.tolist()

def send_features_to_pynq(points):
    vector = preprocess_to_vector(points, num_points=32, min_distance=2.0)

    if vector is None:
        print("[udp] skipped: preprocessing returned None")
        return

    if len(vector) != 70:
        print(f"[udp] skipped: expected 70 features, got {len(vector)}")
        return

    q = quantize_features(vector)

    payload = {
        "features": q
    }

    sock.sendto(json.dumps(payload).encode("utf-8"), pynq_addr)
    print(f"[udp] sent {len(q)} features to PYNQ")

def save_stroke_locally(points, label="test", root="dataset"):
    folder = os.path.join(root, label)
    os.makedirs(folder, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = os.path.join(folder, f"{label}_{timestamp}.json")

    data = {
        "stroke": [[int(p[0]), int(p[1])] for p in points]
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[save] saved {filename}")

def open_camera():
    for idx in range(2):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, idx
        cap.release()
    return None, None

# =========================================================
# MEDIAPIPE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# =========================================================
# CAMERA
# =========================================================
cap, cam_idx = open_camera()
if cap is None:
    print("No camera found.")
    raise SystemExit(1)

print("Using camera index:", cam_idx)

# =========================================================
# STATE
# =========================================================
canvas = None
current = []
pen_down_prev = False
last_t = None
xf = yf = None
t0 = time.time()

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.05)
        continue

    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    t = time.time()
    if last_t is None:
        last_t = t
    last_t = t

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((t - t0) * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    cursor = None
    pen_down = False

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        ix, iy = hand[8].x, hand[8].y   # index tip
        tx, ty = hand[4].x, hand[4].y   # thumb tip

        cursor_raw = (int(ix * W), int(iy * H))

        pinch_px = np.hypot((ix - tx) * W, (iy - ty) * H)
        pen_down = pinch_px < PINCH_THRESH_PX

        if xf is None:
            xf, yf = cursor_raw
        else:
            xf = (1 - SMOOTH_ALPHA) * xf + SMOOTH_ALPHA * cursor_raw[0]
            yf = (1 - SMOOTH_ALPHA) * yf + SMOOTH_ALPHA * cursor_raw[1]

        cursor = (int(xf), int(yf))

    # -------------------------
    # DRAWING
    # -------------------------
    if pen_down and cursor is not None:
        if not current:
            current.append(cursor)
        else:
            if np.hypot(cursor[0] - current[-1][0], cursor[1] - current[-1][1]) >= 2.0:
                current.append(cursor)

    # -------------------------
    # PEN UP → finalize stroke
    # -------------------------
    if (not pen_down) and pen_down_prev and current:
        if len(current) < MIN_POINTS:
            print("[stroke] ignored: too short")
        else:
            print("[stroke] total points:", len(current))
            print("[stroke] first 10 points:", current[:10])

            # draw committed stroke to canvas
            for i in range(1, len(current)):
                cv2.line(canvas, current[i - 1], current[i], (255, 255, 255), 3)

            # optional local save
            save_stroke_locally(current, label="test", root="holdout_test")

            # preprocess + send features to PYNQ
            vector = preprocess_to_vector(current, num_points=32, min_distance=2.0)
            if vector is not None:
                print(f"[preprocess] feature vector length = {len(vector)}")
                print(f"[preprocess] first 10 raw features = {vector[:10]}")
                q = quantize_features(vector)
                print(f"[preprocess] first 10 quantized features = {q[:10]}")
                send_features_to_pynq(current)
            else:
                print("[preprocess] skipped: vector is None")

        current = []

    pen_down_prev = pen_down

    # -------------------------
    # DISPLAY
    # -------------------------
    display = frame.copy()

    mask = canvas[:, :, 0] > 0
    display[mask] = canvas[mask]

    if len(current) >= 2:
        for i in range(1, len(current)):
            cv2.line(display, current[i - 1], current[i], (200, 200, 200), 2)

    if cursor is not None:
        color = (0, 255, 0) if pen_down else (0, 0, 255)
        cv2.circle(display, cursor, 8, color, -1)

    cv2.putText(display, f"PYNQ -> {PYNQ_IP}:{PYNQ_PORT}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, "Pinch to draw | C=clear | ESC=quit", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Hand Drawing -> FPGA Classifier", display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k in (ord('c'), ord('C')):
        canvas[:] = 0
        current = []

cap.release()
cv2.destroyAllWindows()
landmarker.close()