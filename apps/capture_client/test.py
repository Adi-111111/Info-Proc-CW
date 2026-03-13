import cv2
import numpy as np
import time
import json
import socket
from pathlib import Path
import sys
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
from flask import Flask, Response
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
BRIDGE_IP = "127.0.0.1"
BRIDGE_PORT = 5010

# Change if needed
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

bridge_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
bridge_addr = (BRIDGE_IP, BRIDGE_PORT)

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

    circle_fit = fit_circle_kasa(points)
    circle_rel = None
    if circle_fit is not None:
        cx, cy, r, rmse = circle_fit
        if r > 10:
            circle_rel = rmse / max(1.0, r)

    rect = try_rectangle(points, eps=0.05 * scale, right_angle_tol=35.0)
    if rect is not None:
        if circle_rel is None or circle_rel > 0.12:
            return "rectangle"

    tri = try_triangle(points, eps=0.05 * scale)
    if tri is not None:
        if circle_rel is None or circle_rel > 0.12:
            return "triangle"

    if circle_rel is not None and circle_rel < 0.16:
        return "circle"

    return None


def final_shape_decision(points, fpga_label):
    if fpga_label == "freehand":
        return "freehand"

    geom_label = classify_with_geometry(points)

    if fpga_label == "circle" and geom_label in ("triangle", "rectangle"):
        return geom_label

    return fpga_label


def quantize_features(features):
    f = np.array(features, dtype=float)
    f[:64] = np.round(f[:64] * COORD_SCALE)
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


# =========================================================
# RENDER HELPERS
# =========================================================
def draw_polyline(canvas, points, thickness=3):
    for i in range(1, len(points)):
        cv2.line(
            canvas,
            tuple(map(int, points[i - 1])),
            tuple(map(int, points[i])),
            (255, 255, 255),
            thickness
        )


def draw_clean_line(canvas, points, thickness=3):
    if len(points) < 2:
        return
    a = points[0]
    b = points[-1]
    cv2.line(canvas, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), thickness)


def draw_clean_circle(canvas, points, thickness=3):
    fit = fit_circle_kasa(points)
    if fit is None:
        draw_polyline(canvas, points, thickness)
        return

    cx, cy, r, rmse = fit
    center = (int(round(cx)), int(round(cy)))
    rr = int(round(r))

    if rr > 0:
        cv2.circle(canvas, center, rr, (255, 255, 255), thickness)
    else:
        draw_polyline(canvas, points, thickness)


def order_polygon_vertices(corners):
    pts = np.array(corners, dtype=np.float32)
    c = np.mean(pts, axis=0)

    def ang(p):
        return np.arctan2(p[1] - c[1], p[0] - c[0])

    pts = sorted(pts.tolist(), key=ang)
    return [(int(round(x)), int(round(y))) for x, y in pts]


def draw_clean_rectangle(canvas, points, thickness=3):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    rect = try_rectangle(points, eps=0.05 * scale, right_angle_tol=35.0)
    if rect is None:
        draw_polyline(canvas, points, thickness)
        return

    rect = order_polygon_vertices(rect)
    pts = np.array(rect, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], True, (255, 255, 255), thickness)


def draw_clean_triangle(canvas, points, thickness=3):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    tri = try_triangle(points, eps=0.05 * scale)
    if tri is None:
        draw_polyline(canvas, points, thickness)
        return

    tri = order_polygon_vertices(tri)
    pts = np.array(tri, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], True, (255, 255, 255), thickness)


def draw_final_shape(canvas, points, label, thickness=3):
    if label == "line":
        draw_clean_line(canvas, points, thickness)
    elif label == "circle":
        draw_clean_circle(canvas, points, thickness)
    elif label == "rectangle":
        draw_clean_rectangle(canvas, points, thickness)
    elif label == "triangle":
        draw_clean_triangle(canvas, points, thickness)
    else:
        draw_polyline(canvas, points, thickness)


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

def build_whiteboard_shape(points, label):
    timestamp_ms = int(time.time() * 1000)
    shape = {
        "object_id": f"obj_{timestamp_ms}",
        "type": "stroke" if label in ("freehand", "line") else label,
        "created_at": timestamp_ms,
        "source": "capture_client",
    }

    if label == "circle":
        fit = fit_circle_kasa(points)
        if fit is None:
            return None
        cx, cy, r, _ = fit
        shape["cx"] = round(cx, 2)
        shape["cy"] = round(cy, 2)
        shape["r"] = round(r, 2)
        shape["params"] = {
            "cx": shape["cx"],
            "cy": shape["cy"],
            "r": shape["r"],
        }

    elif label == "rectangle":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        rect = try_rectangle(points, eps=0.05 * scale, right_angle_tol=35.0)
        if rect is None:
            return None
        rect = order_polygon_vertices(rect)
        corners = [[int(p[0]), int(p[1])] for p in rect]
        shape["corners"] = corners
        shape["params"] = {"corners": corners}

    elif label == "triangle":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        tri = try_triangle(points, eps=0.05 * scale)
        if tri is None:
            return None
        tri = order_polygon_vertices(tri)
        corners = [[int(p[0]), int(p[1])] for p in tri]
        shape["corners"] = corners
        shape["params"] = {"corners": corners}

    elif label == "line":
        line_points = [
            [int(points[0][0]), int(points[0][1])],
            [int(points[-1][0]), int(points[-1][1])],
        ]
        shape["points"] = line_points
        shape["params"] = {"points": line_points}

    elif label == "freehand":
        stroke_points = [[int(p[0]), int(p[1])] for p in points]
        shape["points"] = stroke_points
        shape["params"] = {"points": stroke_points}

    else:
        return None

    return shape


def send_shape_to_bridge(points, label):
    shape = build_whiteboard_shape(points, label)
    if shape is None:
        print(f"[bridge] failed to build shape for label '{label}'")
        return False

    bridge_sock.sendto(json.dumps(shape).encode("utf-8"), bridge_addr)
    print(f"[bridge] sent {shape['type']} to {BRIDGE_IP}:{BRIDGE_PORT}")
    return True



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

# Flask app

app = Flask(__name__)
latest_frame = None

def generate_frames():
    global latest_frame

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # improve latency
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_server():
    app.run(host="0.0.0.0", port=8001, threaded=True)

threading.Thread(target=run_server, daemon=True).start()

# =========================================================
# STATE
# =========================================================
canvas = None
current = []
last_completed_stroke = None
pending_stroke_for_render = None

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


    # improve latency

    small = cv2.resize(frame, (640, 360))

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
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

    # Draw current stroke preview
    if pen_down and cursor is not None:
        if not current:
            current.append(cursor)
        else:
            if dist(cursor, current[-1]) >= 2.0:
                current.append(cursor)

    # Pen-up -> send stroke, but do NOT commit to canvas yet
    if (not pen_down) and pen_down_prev and current:
        if len(current) >= MIN_POINTS:
            last_completed_stroke = current[:]
            pending_stroke_for_render = current[:]

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

            if pending_stroke_for_render is not None:
                draw_final_shape(canvas, pending_stroke_for_render, last_final_label, thickness=3)
                send_shape_to_bridge(pending_stroke_for_render, last_final_label)
                pending_stroke_for_render = None

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

    # cv2.imshow("FPGA Shape Classifier", display) -- debug window

    latest_frame = display

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k in (ord('c'), ord('C')):
        canvas[:] = 0
        current = []
        last_completed_stroke = None
        pending_stroke_for_render = None
        last_fpga_label = "none"
        last_final_label = "none"

cap.release()
cv2.destroyAllWindows()
landmarker.close()
