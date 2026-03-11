import cv2
import numpy as np
import time
import json
import socket
from pathlib import Path
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# Import preprocessing
# =========================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "shape_classifier"))
from preprocess import preprocess_to_vector

# =========================================================
# CONFIG
# =========================================================
PYNQ_IP = "192.168.2.99"
PYNQ_PORT = 5005
REPLY_PORT = 5006

# Change this if your laptop's IP on the PYNQ link is different
LAPTOP_REPLY_IP = "192.168.2.1"

CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]

COORD_SCALE = 127
GEOM_SCALE = 32

PINCH_THRESH_PX = 80
SMOOTH_ALPHA = 0.25
MIN_POINTS = 5

# =========================================================
# UDP
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
    root = Path(root)
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = folder / f"{label}_{timestamp}.json"

    data = {
        "stroke": [[int(p[0]), int(p[1])] for p in points]
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[save] saved {filename}")


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


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
        return points[:]

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
    if len(pts) < 5:
        return None

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

    closed = dist(points[0], points[-1]) < 0.22 * scale
    if not closed:
        return None

    # Circle score
    circle_fit = fit_circle_kasa(points)
    circle_rel = None
    if circle_fit is not None:
        cx, cy, r, rmse = circle_fit
        if r > 10:
            circle_rel = rmse / max(1.0, r)

    # Strong rectangle only if:
    # 1) rectangle detector succeeds
    # 2) shape is NOT too circle-like
    rect = try_rectangle(points, eps=0.05 * scale, right_angle_tol=35.0)
    if rect is not None:
        if circle_rel is None or circle_rel > 0.12:
            return "rectangle"

    # Strong triangle only if:
    # 1) triangle detector succeeds
    # 2) shape is NOT too circle-like
    tri = try_triangle(points, eps=0.05 * scale)
    if tri is not None:
        if circle_rel is None or circle_rel > 0.12:
            return "triangle"

    # Only call circle if polygon tests did not strongly win
    if circle_rel is not None and circle_rel < 0.16:
        return "circle"

    return None


def final_shape_decision(points, fpga_label):
    # Never override freehand
    if fpga_label == "freehand":
        return "freehand"

    geom_label = classify_with_geometry(points)

    # Only override circles, and only into strong polygon classes
    if fpga_label == "circle" and geom_label in ("triangle", "rectangle"):
        return geom_label

    # Otherwise trust FPGA
    return fpga_label


def quantize_features(features):
    f = np.array(features, dtype=float)

    # first 64 = coordinates
    f[:64] = np.round(f[:64] * COORD_SCALE)

    # last 6 = geometry
    f[64:] = np.round(f[64:] * GEOM_SCALE)

    f = np.clip(f, -128, 127).astype(np.int8)
    return f.tolist()


def send_features_to_pynq(points):
    vector = preprocess_to_vector(points, num_points=32, min_distance=2.0)

    if vector is None:
        print("[preprocess] failed")
        return None

    if len(vector) != 70:
        print(f"[preprocess] expected 70 features, got {len(vector)}")
        return None

    q = quantize_features(vector)

    payload = {
        "features": q,
        "reply_ip": LAPTOP_REPLY_IP,
        "reply_port": REPLY_PORT
    }

    send_sock.sendto(json.dumps(payload).encode("utf-8"), pynq_addr)

    print("[udp] sent features to PYNQ")
    print("[preprocess] first 10 raw features:", vector[:10])
    print("[preprocess] first 10 quantized features:", q[:10])

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
last_completed_stroke = None

xf = yf = None
pen_down_prev = False

last_fpga_label = "none"
last_final_label = "none"

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

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((time.time() - t0) * 1000)
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

    # Draw current stroke
    if pen_down and cursor is not None:
        if not current:
            current.append(cursor)
        else:
            if dist(cursor, current[-1]) >= 2.0:
                current.append(cursor)

    # Pen-up -> finalize stroke
    if (not pen_down) and pen_down_prev and current:
        if len(current) >= MIN_POINTS:
            last_completed_stroke = current[:]

            for i in range(1, len(current)):
                cv2.line(canvas, current[i - 1], current[i], (255, 255, 255), 3)

            save_stroke_locally(current, label="test", root="holdout_test")
            send_features_to_pynq(current)
        else:
            print("[stroke] ignored: too short")

        current = []

    pen_down_prev = pen_down

    # Poll FPGA reply
    reply = poll_pynq_reply()
    if reply is not None:
        if "error" in reply:
            print("[udp] pynq error:", reply["error"])
        else:
            fpga_label = reply.get("label", "none")
            last_fpga_label = fpga_label

            if last_completed_stroke is not None:
                last_final_label = final_shape_decision(last_completed_stroke, fpga_label)
            else:
                last_final_label = fpga_label

            print(f"[decision] FPGA={last_fpga_label} FINAL={last_final_label}")

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

    cv2.putText(display, f"FPGA: {last_fpga_label}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(display, f"FINAL: {last_final_label}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(display, "Pinch to draw | C=clear | ESC=quit", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("FPGA Shape Classifier", display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k in (ord('c'), ord('C')):
        canvas[:] = 0
        current = []
        last_completed_stroke = None
        last_fpga_label = "none"
        last_final_label = "none"

cap.release()
cv2.destroyAllWindows()
landmarker.close()