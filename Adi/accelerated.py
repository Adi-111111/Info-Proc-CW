import cv2
import numpy as np
import time
import socket
import json
import threading
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Network config
PYNQ_IP     = "192.168.2.99"
PYNQ_PORT   = 5005
LAPTOP_PORT = 5006

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pynq_addr = (PYNQ_IP, PYNQ_PORT)

received_shapes = []
shapes_lock     = threading.Lock()
_running        = True

def receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', LAPTOP_PORT))
    sock.settimeout(1.0)
    print(f"[recv] listening on :{LAPTOP_PORT}")
    while _running:
        try:
            data, _ = sock.recvfrom(65535)
            shape = json.loads(data.decode('utf-8'))
            with shapes_lock:
                received_shapes.append(shape)
            print(f"[recv] {shape['type']:12s}  id={shape['id']}")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[recv] error: {e}")
    sock.close()

threading.Thread(target=receiver_thread, daemon=True).start()

# ── Geometry helpers (laptop-side only — cursor/eraser/snap, NOT shape recognition) ──
def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def point_line_distance(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom < 1e-6:
        return dist(p, a)
    return abs(vx * wy - vy * wx) / np.sqrt(denom)

def snap_angle_endpoint(a, b, angle_step_deg=45):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return a, b
    ang = np.degrees(np.arctan2(dy, dx))
    snapped = round(ang / angle_step_deg) * angle_step_deg
    rad = np.radians(snapped)
    return a, (int(round(ax + length * np.cos(rad))), int(round(ay + length * np.sin(rad))))

def to_world(p_screen, scale, ox, oy):
    return (int(round((p_screen[0] - ox) / scale)), int(round((p_screen[1] - oy) / scale)))

def to_screen(p_world, scale, ox, oy):
    return (int(round(p_world[0] * scale + ox)), int(round(p_world[1] * scale + oy)))

def snap_to_grid(p, g):
    return (int(round(p[0] / g) * g), int(round(p[1] / g) * g))

def is_open_palm(hand, W, H):
    wrist = np.array([hand[0].x * W, hand[0].y * H], dtype=float)
    tips = [4, 8, 12, 16, 20]
    d = [np.linalg.norm(np.array([hand[i].x * W, hand[i].y * H]) - wrist) for i in tips]
    return float(np.mean(d)) > 170.0

# Draw shapes received from PYNQ
def draw_pynq_shape(canvas, shape):
    t = shape['type']
    p = shape['params']
    if t == 'circle':
        c = (int(round(p['cx'])), int(round(p['cy'])))
        r = int(round(p['r']))
        cv2.circle(canvas, c, r, (255, 255, 255), -1)
        cv2.circle(canvas, c, r, (255, 255, 255), 3)
    elif t == 'rectangle':
        pts = np.array([[c[0], c[1]] for c in p['corners']], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], (255, 255, 255))
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 3)
    elif t == 'polyline':
        pts = [(int(pt[0]), int(pt[1])) for pt in p['points']]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (255, 255, 255), 3)

# Constants
SMOOTH_ALPHA     = 0.25
PINCH_THRESH_PX  = 80
PAUSE_SPEED_PX   = 30.0
PAUSE_TIME_S     = 0.25
SNAP_WINDOW      = 20
LINE_TOL_PX      = 6.0
SHAPE_PAUSE_TIME = 0.35
ERASER_RADIUS_WORLD = 35
GRID_SIZE        = 25
SHOW_GRID        = True
GRID_SNAP        = False
VIEW_SCALE_MIN   = 0.6
VIEW_SCALE_MAX   = 3.0

#MediaPipe setup
_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(_script_dir, "hand_landmarker.task")

if not os.path.exists(model_path):
    import urllib.request
    _MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print(f"[init] downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(_MODEL_URL, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

#Camera setup
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

cap, cam_idx = open_camera()
if cap is None:
    print("No camera found.")
    raise SystemExit(1)
print("Using camera index:", cam_idx)

# State
canvas         = None
current        = []
strokes        = []
still_time     = 0.0
last_t         = None
last_pt_screen = None
pen_down_prev  = False
xf = yf        = None
shape_buffer   = None
shape_timer    = 0.0
undo_stack     = []
redo_stack     = []
eraser_active_prev   = False
view_scale     = 1.0
view_ox        = 0.0
view_oy        = 0.0
twohand_prev_mid  = None
twohand_prev_dist = None

t0 = time.time()

#Main loop
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.05)
        continue

    frame = np.ascontiguousarray(cv2.flip(frame, 1))
    H, W  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    t = time.time()
    if last_t is None:
        last_t = t
    dt     = t - last_t
    last_t = t

    # MediaPipe hand detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result   = landmarker.detect_for_video(mp_image, int((t - t0) * 1000))

    cursor_screen = None
    pen_down      = False
    hands_n       = 0
    open_palm_1   = False
    open_palm_2   = False
    hand1 = hand2 = None

    if result.hand_landmarks:
        hands_n   = len(result.hand_landmarks)
        hand1     = result.hand_landmarks[0]
        open_palm_1 = is_open_palm(hand1, W, H)

        ix, iy = hand1[8].x, hand1[8].y
        tx, ty = hand1[4].x, hand1[4].y
        cursor_screen = (int(ix * W), int(iy * H))
        pen_down = np.hypot((ix - tx) * W, (iy - ty) * H) < PINCH_THRESH_PX

        if hands_n >= 2:
            hand2       = result.hand_landmarks[1]
            open_palm_2 = is_open_palm(hand2, W, H)

    #Two-hand zoom / pan
    if hands_n >= 2 and open_palm_1 and open_palm_2 and not pen_down:
        p1  = (int(hand1[8].x * W), int(hand1[8].y * H))
        p2  = (int(hand2[8].x * W), int(hand2[8].y * H))
        mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        d12 = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

        if twohand_prev_mid is not None and twohand_prev_dist is not None:
            view_ox += mid[0] - twohand_prev_mid[0]
            view_oy += mid[1] - twohand_prev_mid[1]
            if twohand_prev_dist > 1e-6:
                ratio     = d12 / twohand_prev_dist
                new_scale = float(np.clip(view_scale * ratio, VIEW_SCALE_MIN, VIEW_SCALE_MAX))
                wx = (mid[0] - view_ox) / max(1e-6, view_scale)
                wy = (mid[1] - view_oy) / max(1e-6, view_scale)
                view_scale = new_scale
                view_ox    = mid[0] - wx * view_scale
                view_oy    = mid[1] - wy * view_scale

        twohand_prev_mid  = mid
        twohand_prev_dist = d12
    else:
        twohand_prev_mid  = None
        twohand_prev_dist = None

    #EMA cursor smoothing
    if cursor_screen is not None:
        if xf is None:
            xf, yf = cursor_screen
        else:
            xf = (1 - SMOOTH_ALPHA) * xf + SMOOTH_ALPHA * cursor_screen[0]
            yf = (1 - SMOOTH_ALPHA) * yf + SMOOTH_ALPHA * cursor_screen[1]
        cursor_f_screen = (int(xf), int(yf))
    else:
        cursor_f_screen = None

    speed = (dist(cursor_f_screen, last_pt_screen) / dt
             if cursor_f_screen and last_pt_screen and dt > 1e-6 else 0.0)
    if cursor_f_screen:
        last_pt_screen = cursor_f_screen

    cursor_world = None
    if cursor_f_screen:
        cursor_world = to_world(cursor_f_screen, view_scale, view_ox, view_oy)
        if GRID_SNAP and pen_down:
            cursor_world = snap_to_grid(cursor_world, GRID_SIZE)

    # Eraser
    eraser_active = hands_n >= 1 and open_palm_1 and not pen_down and cursor_world is not None
    if eraser_active and not eraser_active_prev:
        undo_stack.append(canvas.copy())
        redo_stack.clear()
    if eraser_active and cursor_world:
        cv2.circle(canvas, cursor_world, ERASER_RADIUS_WORLD, (0, 0, 0), -1)
    eraser_active_prev = eraser_active

    # Drawing
    if pen_down and cursor_world:
        if not current:
            current.append(cursor_world)
            still_time = 0.0
        elif dist(cursor_world, current[-1]) >= 2.0:
            current.append(cursor_world)

        if speed < PAUSE_SPEED_PX:
            still_time += dt
        else:
            still_time = 0.0

        # Angle snap on slow pause
        if still_time >= PAUSE_TIME_S and len(current) >= SNAP_WINDOW:
            seg = current[-SNAP_WINDOW:]
            a, b = seg[0], seg[-1]
            if max(point_line_distance(p, a, b) for p in seg) <= LINE_TOL_PX:
                a2, b2 = snap_angle_endpoint(a, b, 45)
                current = current[:-SNAP_WINDOW] + [a2, b2]
            still_time = 0.0

    # Pen-up: send stroke to PYNQ
    if not pen_down and pen_down_prev and current:
        shape_buffer = current[:]
        shape_timer  = time.time()
        strokes.append(current)
        current    = []
        still_time = 0.0
    pen_down_prev = pen_down

    shape_label = "none"
    if shape_buffer is not None and (time.time() - shape_timer) >= SHAPE_PAUSE_TIME:
        undo_stack.append(canvas.copy())
        redo_stack.clear()

        # Draw grey preview
        for i in range(1, len(shape_buffer)):
            cv2.line(canvas, shape_buffer[i - 1], shape_buffer[i], (80, 80, 80), 2)

        # Send raw stroke to PYNQ — all shape recognition happens there
        try:
            payload = json.dumps({"stroke": [[p[0], p[1]] for p in shape_buffer]}).encode("utf-8")
            send_sock.sendto(payload, pynq_addr)
            print(f"[send] stroke pts={len(shape_buffer)}")
        except Exception as e:
            print(f"[send] UDP error: {e}")

        shape_label  = "sent→PYNQ"
        shape_buffer = None

    # Receive and draw PYNQ result
    with shapes_lock:
        for shape in received_shapes:
            # Revert canvas to pre-preview snapshot, then draw clean FPGA result
            if undo_stack:
                canvas[:] = undo_stack[-1]
            draw_pynq_shape(canvas, shape)
            shape_label = shape["type"]
        received_shapes.clear()

    # Display
    display = frame.copy()
    M = np.array([[view_scale, 0, view_ox],
                  [0, view_scale, view_oy]], dtype=np.float32)
    warped = cv2.warpAffine(canvas, M, (W, H),
                            flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    mask = warped[:, :, 0] > 0
    display[mask] = warped[mask]

    if SHOW_GRID:
        for x in range(0, W, GRID_SIZE):
            cv2.line(display, (x, 0), (x, H), (30, 30, 30), 1)
        for y in range(0, H, GRID_SIZE):
            cv2.line(display, (0, y), (W, y), (30, 30, 30), 1)

    # Live stroke preview
    if len(current) >= 2:
        for i in range(1, len(current)):
            a = to_screen(current[i - 1], view_scale, view_ox, view_oy)
            b = to_screen(current[i],     view_scale, view_ox, view_oy)
            cv2.line(display, a, b, (200, 200, 200), 2)

    # Cursor dot
    if cursor_f_screen:
        color = (0, 255, 0) if pen_down else ((255, 0, 0) if eraser_active else (0, 0, 255))
        cv2.circle(display, cursor_f_screen, 8, color, -1)

    # HUD
    cv2.putText(display,
                f"hands={hands_n} pinch={'DOWN' if pen_down else 'UP'} "
                f"eraser={'ON' if eraser_active else 'OFF'} "
                f"zoom={view_scale:.2f} grid={'ON' if SHOW_GRID else 'OFF'} "
                f"snap={'ON' if GRID_SNAP else 'OFF'}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display,
                f"shape={shape_label} | Z=undo Y=redo | G=grid | S=snap | C=clear | ESC=quit",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(display,
                "Two open palms: pan + zoom",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Whiteboard (FPGA accelerated)", display)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k in (ord('c'), ord('C')):
        undo_stack.append(canvas.copy())
        redo_stack.clear()
        canvas[:] = 0; strokes.clear(); current = []
        shape_buffer = None; still_time = 0.0
    if k in (ord('z'), ord('Z')) and undo_stack:
        redo_stack.append(canvas.copy())
        canvas[:] = undo_stack.pop()
    if k in (ord('y'), ord('Y')) and redo_stack:
        undo_stack.append(canvas.copy())
        canvas[:] = redo_stack.pop()
    if k in (ord('g'), ord('G')):
        SHOW_GRID = not SHOW_GRID
    if k in (ord('s'), ord('S')):
        GRID_SNAP = not GRID_SNAP

# Cleanup
_running = False
cap.release()
cv2.destroyAllWindows()
send_sock.close()
landmarker.close()