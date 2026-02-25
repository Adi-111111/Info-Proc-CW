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

def try_rectangle(points, eps=12.0, right_angle_tol=50.0):
    """
    Lenient rectangle detector:
    - RDP simplify
    - Drop shortest edges until 4 corners remain
    - Check approx right angles
    """
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

def to_world(p_screen, scale, ox, oy):
    """screen -> world"""
    return (int(round((p_screen[0] - ox) / scale)), int(round((p_screen[1] - oy) / scale)))

def to_screen(p_world, scale, ox, oy):
    """world -> screen"""
    return (int(round(p_world[0] * scale + ox)), int(round(p_world[1] * scale + oy)))

def snap_to_grid(p, g):
    return (int(round(p[0] / g) * g), int(round(p[1] / g) * g))

def is_open_palm(hand, W, H):
    """
    Simple open-palm heuristic:
    fingertips are far from wrist on average.
    Tune threshold if needed.
    """
    wrist = np.array([hand[0].x * W, hand[0].y * H], dtype=float)
    tips = [4, 8, 12, 16, 20]
    d = []
    for i in tips:
        p = np.array([hand[i].x * W, hand[i].y * H], dtype=float)
        d.append(np.linalg.norm(p - wrist))
    return float(np.mean(d)) > 170.0  # tune 150..220


# =========================================================
# Params
# =========================================================
SMOOTH_ALPHA = 0.25

PINCH_THRESH_PX = 80

PAUSE_SPEED_PX = 30.0
PAUSE_TIME_S = 0.25
SNAP_WINDOW = 20
LINE_TOL_PX = 6.0

CLOSE_THRESH_PX = 60.0
ANGLE_STEP_DEG = 45

SHAPE_PAUSE_TIME = 0.35

CIRCLE_REL_TOL = 0.18
RECT_RIGHT_ANGLE_TOL = 50.0

# Eraser
ERASER_RADIUS_WORLD = 35  # in world pixels

# Grid
GRID_SIZE = 25
SHOW_GRID = True
GRID_SNAP = False

# Zoom/pan
VIEW_SCALE_MIN = 0.6
VIEW_SCALE_MAX = 3.0

# =========================================================
# MediaPipe
# =========================================================
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,  # needed for zoom/pan
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# =========================================================
# Camera
# =========================================================
def open_camera():
    for idx in range(2):
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
last_pt_screen = None
pen_down_prev = False
xf = yf = None

# recognition buffer
shape_buffer = None
shape_timer = 0.0

# undo/redo
undo_stack = []
redo_stack = []

# eraser state
eraser_active_prev = False

# view transform (world -> screen)
view_scale = 1.0
view_ox = 0.0
view_oy = 0.0

# two-hand gesture state
twohand_prev_mid = None
twohand_prev_dist = None

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

    # defaults
    cursor_screen = None
    pen_down = False
    hands_n = 0
    open_palm_1 = False
    open_palm_2 = False
    hand1 = None
    hand2 = None

    if result.hand_landmarks:
        hands_n = len(result.hand_landmarks)
        hand1 = result.hand_landmarks[0]
        open_palm_1 = is_open_palm(hand1, W, H)

        # cursor from hand1 index tip
        ix, iy = hand1[8].x, hand1[8].y
        tx, ty = hand1[4].x, hand1[4].y
        cursor_screen = (int(ix * W), int(iy * H))

        pinch_px = np.hypot((ix - tx) * W, (iy - ty) * H)
        pen_down = pinch_px < PINCH_THRESH_PX

        if hands_n >= 2:
            hand2 = result.hand_landmarks[1]
            open_palm_2 = is_open_palm(hand2, W, H)

    # =====================================================
    # Two-hand zoom/pan (two open palms, not drawing)
    # =====================================================
    if hands_n >= 2 and open_palm_1 and open_palm_2 and (not pen_down):
        p1 = (int(hand1[8].x * W), int(hand1[8].y * H))
        p2 = (int(hand2[8].x * W), int(hand2[8].y * H))
        mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        d12 = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

        if twohand_prev_mid is not None and twohand_prev_dist is not None:
            # pan by midpoint delta
            dx = mid[0] - twohand_prev_mid[0]
            dy = mid[1] - twohand_prev_mid[1]
            view_ox += dx
            view_oy += dy

            # zoom by distance ratio, anchored at mid (screen)
            if twohand_prev_dist > 1e-6:
                ratio = d12 / twohand_prev_dist
                new_scale = float(np.clip(view_scale * ratio, VIEW_SCALE_MIN, VIEW_SCALE_MAX))

                # keep world point under "mid" fixed
                wx = (mid[0] - view_ox) / max(1e-6, view_scale)
                wy = (mid[1] - view_oy) / max(1e-6, view_scale)
                view_scale = new_scale
                view_ox = mid[0] - wx * view_scale
                view_oy = mid[1] - wy * view_scale

        twohand_prev_mid = mid
        twohand_prev_dist = d12
    else:
        twohand_prev_mid = None
        twohand_prev_dist = None

    # =====================================================
    # Smooth cursor in SCREEN space
    # =====================================================
    if cursor_screen is not None:
        if xf is None:
            xf, yf = cursor_screen
        else:
            xf = (1 - SMOOTH_ALPHA) * xf + SMOOTH_ALPHA * cursor_screen[0]
            yf = (1 - SMOOTH_ALPHA) * yf + SMOOTH_ALPHA * cursor_screen[1]
        cursor_f_screen = (int(xf), int(yf))
    else:
        cursor_f_screen = None

    # speed in screen px/s
    if cursor_f_screen is not None and last_pt_screen is not None and dt > 1e-6:
        speed = dist(cursor_f_screen, last_pt_screen) / dt
    else:
        speed = 0.0
    if cursor_f_screen is not None:
        last_pt_screen = cursor_f_screen

    # map to WORLD for drawing/erasing
    cursor_world = None
    if cursor_f_screen is not None:
        cursor_world = to_world(cursor_f_screen, view_scale, view_ox, view_oy)
        if GRID_SNAP and pen_down:
            cursor_world = snap_to_grid(cursor_world, GRID_SIZE)

    # =====================================================
    # Eraser gesture (open palm with hand1, not drawing)
    # =====================================================
    eraser_active = (hands_n >= 1 and open_palm_1 and (not pen_down) and cursor_world is not None)

    if eraser_active and (not eraser_active_prev):
        # snapshot once at eraser start
        undo_stack.append(canvas.copy())
        redo_stack.clear()

    if eraser_active and cursor_world is not None:
        cv2.circle(canvas, cursor_world, ERASER_RADIUS_WORLD, (0, 0, 0), -1)

    eraser_active_prev = eraser_active

    # =====================================================
    # Drawing (pen down)
    # =====================================================
    if pen_down and cursor_world is not None:
        if not current:
            current.append(cursor_world)
            still_time = 0.0
        else:
            if dist(cursor_world, current[-1]) >= 2.0:
                current.append(cursor_world)

        if speed < PAUSE_SPEED_PX:
            still_time += dt
        else:
            still_time = 0.0

        # snap-to-line on pause (+ angle snapping) in WORLD
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
    # Shape recognition after pause (commit to canvas)
    # =====================================================
    shape_label = "none"
    if shape_buffer is not None and (time.time() - shape_timer) >= SHAPE_PAUSE_TIME:
        pts = resample_polyline(shape_buffer, step=8.0)

        closed = (len(pts) >= 10 and dist(pts[0], pts[-1]) < CLOSE_THRESH_PX)
        if closed and pts[0] != pts[-1]:
            pts.append(pts[0])

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = max(20.0, max(max(xs) - min(xs), max(ys) - min(ys)))

        drew = False

        # snapshot BEFORE modifying canvas
        undo_stack.append(canvas.copy())
        redo_stack.clear()

        # circle
        if closed and len(pts) >= 25:
            fit = fit_circle_kasa(pts[:-1])
            if fit is not None:
                cx, cy, r, rmse = fit
                rel = rmse / max(1.0, r)
                if 15 <= r <= 0.9 * scale and rel <= CIRCLE_REL_TOL:
                    draw_circle(canvas, cx, cy, r, thickness=3, fill=True)
                    shape_label = "circle"
                    drew = True

        # rectangle (lenient)
        if (not drew) and closed:
            eps = 0.06 * scale
            rect = try_rectangle(pts, eps=eps, right_angle_tol=RECT_RIGHT_ANGLE_TOL)
            if rect is not None:
                draw_rect(canvas, rect, thickness=3, fill=True)
                shape_label = "rect"
                drew = True

        # fallback stroke
        if not drew:
            for i in range(1, len(shape_buffer)):
                cv2.line(canvas, shape_buffer[i - 1], shape_buffer[i], (255, 255, 255), 3)
            shape_label = "stroke"

        shape_buffer = None

    # =====================================================
    # Display: warp canvas using view transform
    # =====================================================
    display = frame.copy()

    # warp canvas into screen coordinates
    M = np.array([[view_scale, 0, view_ox],
                  [0, view_scale, view_oy]], dtype=np.float32)
    warped_canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))

    mask = warped_canvas[:, :, 0] > 0
    display[mask] = warped_canvas[mask]

    # grid overlay (screen space)
    if SHOW_GRID:
        step = GRID_SIZE
        for x in range(0, W, step):
            cv2.line(display, (x, 0), (x, H), (30, 30, 30), 1)
        for y in range(0, H, step):
            cv2.line(display, (0, y), (W, y), (30, 30, 30), 1)

    # preview current stroke (transform world -> screen)
    if len(current) >= 2:
        for i in range(1, len(current)):
            a = to_screen(current[i - 1], view_scale, view_ox, view_oy)
            b = to_screen(current[i], view_scale, view_ox, view_oy)
            cv2.line(display, a, b, (200, 200, 200), 2)

    # cursor
    if cursor_f_screen is not None:
        color = (0, 255, 0) if pen_down else (0, 0, 255)
        if eraser_active:
            color = (255, 0, 0)
        cv2.circle(display, cursor_f_screen, 8, color, -1)

    # HUD
    cv2.putText(display,
                f"hands={hands_n} pinch={'DOWN' if pen_down else 'UP'} "
                f"eraser={'ON' if eraser_active else 'OFF'} "
                f"zoom={view_scale:.2f} grid={'ON' if SHOW_GRID else 'OFF'} snap={'ON' if GRID_SNAP else 'OFF'}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(display,
                f"shape={shape_label} | Z=undo Y=redo | G=grid | S=snap | C=clear | ESC=quit",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(display,
                "Two open palms: pan + zoom (move hands apart/together)",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand CAD (Undo/Eraser/Grid + Two-hand Zoom/Pan)", display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # clear
    if k in (ord('c'), ord('C')):
        undo_stack.append(canvas.copy())
        redo_stack.clear()
        canvas[:] = 0
        strokes.clear()
        current = []
        shape_buffer = None
        still_time = 0.0

    # undo / redo
    if k in (ord('z'), ord('Z')) and undo_stack:
        redo_stack.append(canvas.copy())
        canvas[:] = undo_stack.pop()

    if k in (ord('y'), ord('Y')) and redo_stack:
        undo_stack.append(canvas.copy())
        canvas[:] = redo_stack.pop()

    # grid toggle
    if k in (ord('g'), ord('G')):
        SHOW_GRID = not SHOW_GRID

    # grid snap toggle
    if k in (ord('s'), ord('S')):
        GRID_SNAP = not GRID_SNAP

cap.release()
cv2.destroyAllWindows()
landmarker.close()
