import cv2
import numpy as np
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# Helpers
# =========================================================
def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def snap_angle_endpoint(a, b, angle_step_deg=45):
    """Snap line AB to nearest multiple of angle_step_deg, preserving length."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return a, b
    ang = np.degrees(np.arctan2(dy, dx))
    snapped = round(ang / angle_step_deg) * angle_step_deg
    rad = np.radians(snapped)
    bx2 = int(round(ax + length * np.cos(rad)))
    by2 = int(round(ay + length * np.sin(rad)))
    return a, (bx2, by2)

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

def resample_polyline(points, step=8.0):
    """Resample stroke so consecutive points are ~step pixels apart."""
    if len(points) < 2:
        return points[:]
    out = [points[0]]
    acc = 0.0
    prev = np.array(points[0], dtype=float)
    for p in points[1:]:
        cur = np.array(p, dtype=float)
        seg = np.linalg.norm(cur - prev)
        if seg < 1e-6:
            continue
        while acc + seg >= step:
            t = (step - acc) / seg
            newp = prev + t * (cur - prev)
            out.append((int(round(newp[0])), int(round(newp[1]))))
            prev = newp
            seg = np.linalg.norm(cur - prev)
            acc = 0.0
        acc += seg
        prev = cur
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out

def rdp(points, eps):
    """Ramer–Douglas–Peucker polyline simplification."""
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
    """Return (cx, cy, r, rmse)."""
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

def angle_deg(u, v):
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def try_rectangle(points, eps=12.0, right_angle_tol=25.0):
    simp = rdp(points, eps)
    if len(simp) >= 2 and simp[0] == simp[-1]:
        simp = simp[:-1]

    # common: 5 points due to jitter; drop the shortest edge
    if len(simp) == 5:
        lens = []
        for i in range(5):
            a = np.array(simp[i], float)
            b = np.array(simp[(i + 1) % 5], float)
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

def draw_circle(canvas, cx, cy, r, thickness=3, fill=False):
    center = (int(round(cx)), int(round(cy)))
    rr = int(round(r))
    if fill:
        cv2.circle(canvas, center, rr, (255, 255, 255), -1)
    cv2.circle(canvas, center, rr, (255, 255, 255), thickness)

def draw_rect(canvas, corners, thickness=3, fill=False):
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    if fill:
        cv2.fillPoly(canvas, [pts], (255, 255, 255))
    cv2.polylines(canvas, [pts], True, (255, 255, 255), thickness)

# =========================================================
# Params
# =========================================================
SMOOTH_ALPHA = 0.25

PINCH_THRESH_PX = 80

PAUSE_SPEED_PX = 30.0
PAUSE_TIME_S = 0.25
SNAP_WINDOW = 20
LINE_TOL_PX = 6.0

CLOSE_THRESH_PX = 30.0
ANGLE_STEP_DEG = 45

# Shape recognition AFTER pen-up pause
SHAPE_PAUSE_TIME = 0.35

# Circle/rect tolerances
CIRCLE_REL_TOL = 0.18   # rmse/r
RECT_RIGHT_ANGLE_TOL = 25.0

# =========================================================
# MediaPipe Tasks: HandLandmarker
# =========================================================
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
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
# Camera
# =========================================================
def open_camera():
    for idx in range(2):  # your Mac reports 0-1 valid
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        print("probe", idx, "ret:", ok, "shape:", None if frame is None else frame.shape)
        if ok and frame is not None:
            return cap, idx
        cap.release()
    return None, None

cap, cam_idx = open_camera()
if cap is None:
    print("No camera found.")
    raise SystemExit(1)
print("Using camera index:", cam_idx)

# =========================================================
# State
# =========================================================
canvas = None
current = []
strokes = []

still_time = 0.0
last_t = None
last_pt = None
pen_down_prev = False
xf = yf = None

# Pen-up recognition buffer
shape_buffer = None
shape_timer = 0.0

t0 = time.time()

# =========================================================
# Main loop
# =========================================================
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.05)
        continue

    frame = np.ascontiguousarray(frame)
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    t = time.time()
    if last_t is None:
        last_t = t
    dt = t - last_t
    last_t = t

    # ---- MediaPipe ----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((t - t0) * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    cursor = None
    pen_down = False
    hands_n = 0

    if result.hand_landmarks:
        hands_n = len(result.hand_landmarks)
        hand = result.hand_landmarks[0]
        ix, iy = hand[8].x, hand[8].y
        tx, ty = hand[4].x, hand[4].y

        cursor = (int(ix * W), int(iy * H))
        pinch_px = np.hypot((ix - tx) * W, (iy - ty) * H)
        pen_down = pinch_px < PINCH_THRESH_PX

    # ---- Smooth cursor ----
    if cursor is not None:
        if xf is None:
            xf, yf = cursor
        else:
            xf = (1 - SMOOTH_ALPHA) * xf + SMOOTH_ALPHA * cursor[0]
            yf = (1 - SMOOTH_ALPHA) * yf + SMOOTH_ALPHA * cursor[1]
        cursor_f = (int(xf), int(yf))
    else:
        cursor_f = None

    # ---- Speed ----
    if cursor_f is not None and last_pt is not None and dt > 1e-6:
        speed = dist(cursor_f, last_pt) / dt
    else:
        speed = 0.0
    if cursor_f is not None:
        last_pt = cursor_f

    # =====================================================
    # Drawing (pen down)
    # =====================================================
    if pen_down and cursor_f is not None:
        if not current:
            current.append(cursor_f)
            still_time = 0.0
        else:
            if dist(cursor_f, current[-1]) >= 2.0:
                current.append(cursor_f)

        if speed < PAUSE_SPEED_PX:
            still_time += dt
        else:
            still_time = 0.0

        # snap-to-line on pause (+ angle snapping)
        if still_time >= PAUSE_TIME_S and len(current) >= SNAP_WINDOW:
            seg = current[-SNAP_WINDOW:]
            a, b = seg[0], seg[-1]
            max_d = max(point_line_distance(p, a, b) for p in seg)
            if max_d <= LINE_TOL_PX:
                a2, b2 = snap_angle_endpoint(a, b, ANGLE_STEP_DEG)
                current = current[:-SNAP_WINDOW] + [a2, b2]
            still_time = 0.0

    # =====================================================
    # Pen-up: store the stroke, recognise AFTER pause
    # =====================================================
    if (not pen_down) and pen_down_prev and current:
        shape_buffer = current[:]
        shape_timer = time.time()
        strokes.append(current)
        current = []
        still_time = 0.0

    pen_down_prev = pen_down

    # =====================================================
    # Shape recognition after pause
    # =====================================================
    shape_label = "none"
    if shape_buffer is not None and (time.time() - shape_timer) >= SHAPE_PAUSE_TIME:
        pts = resample_polyline(shape_buffer, step=8.0)

        closed = (len(pts) >= 10 and dist(pts[0], pts[-1]) < CLOSE_THRESH_PX)
        if closed and pts[0] != pts[-1]:
            pts.append(pts[0])

        drew = False

        # adaptive scale
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = max(20.0, max(max(xs) - min(xs), max(ys) - min(ys)))

        # ---- circle ----
        if closed and len(pts) >= 25:
            fit = fit_circle_kasa(pts[:-1])
            if fit is not None:
                cx, cy, r, rmse = fit
                rel = rmse / max(1.0, r)
                if 15 <= r <= 0.9 * scale and rel <= CIRCLE_REL_TOL:
                    draw_circle(canvas, cx, cy, r, thickness=3, fill=True)
                    shape_label = "circle"
                    drew = True

        # ---- rectangle ----
        if (not drew) and closed:
            eps = 0.03 * scale
            rect = try_rectangle(pts, eps=eps, right_angle_tol=RECT_RIGHT_ANGLE_TOL)
            if rect is not None:
                draw_rect(canvas, rect, thickness=3, fill=True)
                shape_label = "rect"
                drew = True

        # ---- fallback ----
        if not drew:
            for i in range(1, len(shape_buffer)):
                cv2.line(canvas, shape_buffer[i - 1], shape_buffer[i], (255, 255, 255), 3)
            shape_label = "stroke"

        shape_buffer = None

    # =====================================================
    # Display (overlay canvas without addWeighted)
    # =====================================================
    display = frame.copy()
    mask = canvas[:, :, 0] > 0
    display[mask] = canvas[mask]

    # preview current stroke
    if len(current) >= 2:
        for i in range(1, len(current)):
            cv2.line(display, current[i - 1], current[i], (200, 200, 200), 2)

    # cursor
    if cursor_f is not None:
        cv2.circle(display, cursor_f, 8, (0, 255, 0) if pen_down else (0, 0, 255), -1)

    cv2.putText(display, f"hands={hands_n} pinch={'DOWN' if pen_down else 'UP'} speed={speed:.0f}px/s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display, f"shape={shape_label} (pause after pen-up to snap)",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Hand CAD prototype (pause-to-recognise)", display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k in (ord('c'), ord('C')):
        canvas[:] = 0
        strokes.clear()
        current = []
        shape_buffer = None
        still_time = 0.0

cap.release()
cv2.destroyAllWindows()
landmarker.close()
