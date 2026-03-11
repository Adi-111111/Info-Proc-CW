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
# CONFIG
# =========================================================
PYNQ_IP = "192.168.2.99"
PYNQ_PORT = 5005
REPLY_PORT = 5006

SMOOTH_ALPHA = 0.25
PINCH_THRESH_PX = 80
MIN_POINTS = 5

COORD_SCALE = 127
GEOM_SCALE = 32

CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]

# =========================================================
# UDP SOCKETS
# =========================================================
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pynq_addr = (PYNQ_IP, PYNQ_PORT)

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(("0.0.0.0", REPLY_PORT))
recv_sock.setblocking(False)

# =========================================================
# HELPERS
# =========================================================
def save_stroke_locally(points, label="test", root="holdout_test"):
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

def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def point_line_distance(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom < 1e-6:
        return dist(p, a)
    return abs(vx * wy - vy * wx) / np.sqrt(denom)

def angle_deg(u, v):
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def rdp(points, eps):
    if len(points) < 3:
        return points
    a = np.array(points[0], dtype=float)
    b = np.array(points[-1], dtype=float)
    ab = b - a
    ab2 = float(ab @ ab)

    max_d = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        p = np.array(points[i], dtype=float)
        if ab2 < 1e-9:
            d = np.linalg.norm(p - a)
        else:
            t = float(((p - a) @ ab) / ab2)
            proj = a + np.clip(t, 0.0, 1.0) * ab
            d = np.linalg.norm(p - proj)
        if d > max_d:
            max_d = d
            idx = i

    if max_d > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]

def fit_circle_kasa(points):
    pts = np.array(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = 0.5 * c[0]
    cy = 0.5 * c[1]
    r = np.sqrt(max(1e-9, cx * cx + cy * cy + c[2]))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((d - r) ** 2)))
    return float(cx), float(cy), float(r), rmse

def try_rectangle(points, eps=12.0, right_angle_tol=45.0):
    simp = rdp(points, eps)
    if len(simp) >= 2 and simp[0] == simp[-1]:
        simp = simp[:-1]

    while len(simp) > 4:
        n = len(simp)
        lens = []
        for i in range(n):
            a = np.array(simp[i], float)
            b = np.array(simp[(i + 1) % n], float)
            lens.append(np.linalg.norm(b - a))
        simp.pop(int(np.argmin(lens)))

    if len(simp) != 4:
        return None

    for i in range(4):
        p_prev = np.array(simp[(i - 1) % 4], dtype=float)
        p = np.array(simp[i], dtype=float)
        p_next = np.array(simp[(i + 1) % 4], dtype=float)
        ang = angle_deg(p_prev - p, p_next - p)
        if abs(ang - 90.0) > right_angle_tol:
            return None

    return simp

def try_triangle(points, eps=12.0):
    simp = rdp(points, eps)
    if len(simp) >= 2 and simp[0] == simp[-1]:
        simp = simp[:-1]

    while len(simp) > 3:
        n = len(simp)
        lens = []
        for i in range(n):
            a = np.array(simp[i], float)
            b = np.array(simp[(i + 1) % n], float)
            lens.append(np.linalg.norm(b - a))
        simp.pop(int(np.argmin(lens)))

    if len(simp) != 3:
        return None

    a, b, c = [np.array(p, dtype=float) for p in simp]
    area = abs(np.cross(b - a, c - a)) * 0.5
    if area < 200:
        return None

    return simp

def classify_with_geometry(points):
    if len(points) < 10:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    closed = dist(points[0], points[-1]) < 0.18 * scale
    if not closed:
        return None

    fit = fit_circle_kasa(points)
    if fit is not None:
        cx, cy, r, rmse = fit
        if r > 10:
            rel = rmse / max(1.0, r)
            if rel < 0.16:
                return "circle"

    rect = try_rectangle(points, eps=0.06 * scale, right_angle_tol=45.0)
    if rect is not None:
        return "rectangle"

    tri = try_triangle(points, eps=0.06 * scale)
    if tri is not None:
        return "triangle"

    return None

def final_shape_decision(points, fpga_label):
    geom_label = classify_with_geometry(points)
    if geom_label in ("triangle", "rectangle"):
        return geom_label
    return fpga_label

def quantize_features(features):
    features = np.array(features, dtype=float)
    features[:64] = np.round(features[:64] * COORD_SCALE)
    features[64:] = np.round(features[64:] * GEOM_SCALE)
    features = np.clip(features, -128, 127).astype(np.int8)
    return features.tolist()

def send_features_to_pynq(points):
    vector = preprocess_to_vector(points, num_points=32, min_distance=2.0)

    if vector is None:
        print("[udp] skipped: preprocessing returned None")
        return None

    if len(vector) != 70:
        print(f"[udp] skipped: expected 70 features, got {len(vector)}")
        return None

    q = quantize_features(vector)

    payload = {
        "features": q,
        "reply_ip": "192.168.2.1",   # laptop IP on USB/Ethernet link; change if needed
        "reply_port": REPLY_PORT
    }

    send_sock.sendto(json.dumps(payload).encode("utf-8"), pynq_addr)
    print(f"[udp] sent {len(q)} quantized features to PYNQ")
    print(f"[udp] first 10 quantized features: {q[:10]}")
    return q

def poll_pynq_reply():
    try:
        data, addr = recv_sock.recvfrom(65535)
    except BlockingIOError:
        return None

    try:
        payload = json.loads(data.decode("utf-8"))
        print(f"[udp] got reply from {addr}: {payload}")
        return payload
    except Exception as e:
        print("[udp] bad reply:", e)
        return None

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

last_fpga_label = "none"
last_final_label = "none"

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

        ix, iy = hand[8].x, hand[8].y
        tx, ty = hand[4].x, hand[4].y

        cursor_raw = (int(ix * W), int(iy * H))
        pinch_px = np.hypot((ix - tx) * W, (iy - ty) * H)
        pen_down = pinch_px < PINCH_THRESH_PX

        if xf is None:
            xf, yf = cursor_raw
        else:
            xf = (1 - SMOOTH_ALPHA) * xf + SMOOTH_ALPHA * cursor_raw[0]
            yf = (1 - SMOOTH_ALPHA) * yf + SMOOTH_ALPHA * cursor_raw[1]

        cursor = (int(xf), int(yf))

    # Drawing
    if pen_down and cursor is not None:
        if not current:
            current.append(cursor)
        else:
            if np.hypot(cursor[0] - current[-1][0], cursor[1] - current[-1][1]) >= 2.0:
                current.append(cursor)

    # Pen-up -> finalize stroke
    if (not pen_down) and pen_down_prev and current:
        if len(current) < MIN_POINTS:
            print("[stroke] ignored: too short")
        else:
            print("[stroke] total points:", len(current))
            print("[stroke] first 10 points:", current[:10])

            for i in range(1, len(current)):
                cv2.line(canvas, current[i - 1], current[i], (255, 255, 255), 3)

            save_stroke_locally(current, label="test", root="holdout_test")

            vector = preprocess_to_vector(current, num_points=32, min_distance=2.0)
            if vector is not None:
                print(f"[preprocess] feature vector length = {len(vector)}")
                print(f"[preprocess] first 10 raw features = {vector[:10]}")
                q = send_features_to_pynq(current)
                if q is not None:
                    print(f"[preprocess] first 10 quantized features = {q[:10]}")
            else:
                print("[preprocess] skipped: vector is None")

            print("[stroke] finished drawing")

        current = []

    pen_down_prev = pen_down

    # Poll for FPGA replies
    reply = poll_pynq_reply()
    if reply is not None:
        fpga_label = reply.get("label", "none")
        last_fpga_label = fpga_label
        last_final_label = final_shape_decision(current if current else [], fpga_label)

    # Display
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
    cv2.putText(display, f"FPGA: {last_fpga_label} | FINAL: {last_final_label}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "Pinch to draw | C=clear | ESC=quit", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Hand Drawing -> FPGA Classifier", display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k in (ord('c'), ord('C')):
        canvas[:] = 0
        current = []
        last_fpga_label = "none"
        last_final_label = "none"

cap.release()
cv2.destroyAllWindows()
landmarker.close()
