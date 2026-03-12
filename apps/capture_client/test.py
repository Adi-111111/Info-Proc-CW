import cv2
import numpy as np
import time

import socket 
import json

from flask import Flask, Response
import threading

import tensorflow as tf
tflite = tf.lite

PYNQ_IP = "192.168.2.99"
PYNQ_PORT = 5005
PYNQ_TCP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pynq_addr = (PYNQ_IP, PYNQ_PORT)


# =========================================================
# Helpers
# =========================================================

PALM_INPUT_SIZE = 192

def preprocess_palm_detector(frame):
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W > H:
        pad_top    = (W - H) // 2
        pad_bottom = W - H - pad_top
        pad_left   = 0
        pad_right  = 0
        sq = W
    else:
        pad_left   = (H - W) // 2
        pad_right  = H - W - pad_left
        pad_top    = 0
        pad_bottom = 0
        sq = H

    padded = cv2.copyMakeBorder(rgb, pad_top, pad_bottom,
                                     pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (PALM_INPUT_SIZE, PALM_INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)

    # FIX 1: correct normalisation — model expects [0, 1] not [-1, 1]
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    pad = (
        pad_left   / sq,
        pad_top    / sq,
        pad_right  / sq,
        pad_bottom / sq,
    )

    return tensor, pad, sq


# FIX 2: correct anchor stride list — must be [8, 16, 16, 16] to produce
# exactly 2016 anchors. [8, 16] only produces 1440.
def generate_anchors():
    strides = [8, 16, 16, 16]
    input_h = input_w = PALM_INPUT_SIZE
    anchors = []

    for stride in strides:
        rows = int(np.ceil(input_h / stride))
        cols = int(np.ceil(input_w / stride))
        for r in range(rows):
            for c in range(cols):
                for _ in range(2):
                    cx = (c + 0.5) / cols
                    cy = (r + 0.5) / rows
                    anchors.append([cx, cy])

    return np.array(anchors, dtype=np.float32)  # [2016, 2] normalised [0,1]

ANCHORS = generate_anchors()

palm_interpreter = tflite.Interpreter(
    model_path="extracted_models/hand_detector.tflite")
palm_interpreter.allocate_tensors()
palm_in  = palm_interpreter.get_input_details()[0]['index']
palm_out = palm_interpreter.get_output_details()

lm_interpreter = tflite.Interpreter(
    model_path="extracted_models/hand_landmarks_detector.tflite")
lm_interpreter.allocate_tensors()
lm_in  = lm_interpreter.get_input_details()[0]['index']
lm_out = lm_interpreter.get_output_details()
lm_out_map = {d['name']: d['index'] for d in lm_out}

def run_palm_detector(tensor):
    palm_interpreter.set_tensor(palm_in, tensor)
    palm_interpreter.invoke()
    regressors = palm_interpreter.get_tensor(palm_out[0]['index'])[0]       # [2016, 18]
    scores_raw = palm_interpreter.get_tensor(palm_out[1]['index'])[0, :, 0] # [2016]
    return regressors, scores_raw

SCORE_THRESH = 0.5
NMS_THRESH   = 0.3

# CHANGED — decode_detections: add [:MAX_HANDS] at the end
def decode_detections(regressors, scores_raw, pad):
    scores = 1.0 / (1.0 + np.exp(-scores_raw.clip(-88, 88)))
    keep = np.where(scores > SCORE_THRESH)[0]
    if len(keep) == 0:
        return []
    kp_scores = scores[keep]
    kp_regs   = regressors[keep]
    kp_anch   = ANCHORS[keep]
    anchor_cx_px = kp_anch[:, 0] * PALM_INPUT_SIZE
    anchor_cy_px = kp_anch[:, 1] * PALM_INPUT_SIZE
    cx = (anchor_cx_px + kp_regs[:, 0]) / PALM_INPUT_SIZE
    cy = (anchor_cy_px + kp_regs[:, 1]) / PALM_INPUT_SIZE
    w  = (kp_regs[:, 2] * 2)            / PALM_INPUT_SIZE
    h  = (kp_regs[:, 3] * 2)            / PALM_INPUT_SIZE
    kps = []
    for i in range(7):
        kx = (anchor_cx_px + kp_regs[:, 4 + 2*i]) / PALM_INPUT_SIZE
        ky = (anchor_cy_px + kp_regs[:, 5 + 2*i]) / PALM_INPUT_SIZE
        kps.append(np.stack([kx, ky], axis=1))
    kps = np.stack(kps, axis=1)
    boxes_nms = []
    for i in range(len(keep)):
        x1 = int((cx[i] - w[i]/2) * 1000)
        y1 = int((cy[i] - h[i]/2) * 1000)
        bw = int(w[i] * 1000)
        bh = int(h[i] * 1000)
        boxes_nms.append([x1, y1, bw, bh])
    indices = cv2.dnn.NMSBoxes(boxes_nms, kp_scores.tolist(), 0, NMS_THRESH)
    if len(indices) == 0:
        return []
    pad_l, pad_t, pad_r, pad_b = pad
    detections = []
    for idx in indices.flatten():
        def unpad_x(v): return (v - pad_l) / (1.0 - pad_l - pad_r)
        def unpad_y(v): return (v - pad_t) / (1.0 - pad_t - pad_b)
        detections.append({
            'cx':    unpad_x(float(cx[idx])),
            'cy':    unpad_y(float(cy[idx])),
            'w':     float(w[idx])  / (1.0 - pad_l - pad_r),
            'h':     float(h[idx])  / (1.0 - pad_t - pad_b),
            'kps':   np.stack([unpad_x(kps[idx,:,0]),
                               unpad_y(kps[idx,:,1])], axis=1),
            'score': float(kp_scores[idx]),
        })
    return detections[:MAX_HANDS]



# CHANGED CONSTANTS — replace the originals
LM_INPUT_SIZE = 224
ROI_SCALE     = 1.5   # was 2.6
MAX_HANDS     = 2

POS_SMOOTH   = 0.5
SIZE_SMOOTH  = 0.7
ANGLE_SMOOTH = 0.5

# NEW — replaces compute_roi_affine
def build_affine(cx, cy, size, angle):
    cos_a = np.cos(angle);  sin_a = np.sin(angle)
    half  = size / 2.0
    src_pts = np.float32([[-half,-half],[half,-half],[-half,half]])
    rot     = np.array([[cos_a,-sin_a],[sin_a,cos_a]], dtype=np.float32)
    dst_pts = (rot @ src_pts.T).T + np.array([cx, cy], dtype=np.float32)
    dst_224 = np.float32([[0,0],[LM_INPUT_SIZE,0],[0,LM_INPUT_SIZE]])
    affine_matrix     = cv2.getAffineTransform(dst_224, dst_pts)
    inv_affine_matrix = cv2.getAffineTransform(dst_pts, dst_224)
    return affine_matrix, inv_affine_matrix

def compute_roi_affine(detection, frame_W, frame_H):
    kps = detection['kps']
    # FIX: atan2(dy, dx) not atan2(dx, dy)
    dx    = kps[2,0]*frame_W - kps[0,0]*frame_W
    dy    = kps[2,1]*frame_H - kps[0,1]*frame_H
    angle = np.arctan2(dy, dx) - np.pi / 2
    cx    = detection['cx'] * frame_W
    cy    = detection['cy'] * frame_H
    size  = max(detection['w'] * frame_W, detection['h'] * frame_H) * ROI_SCALE
    return build_affine(cx, cy, size, angle)


LM_INPUT_SIZE = 224

# CHANGED — crop_hand_region: add vertical flip
def crop_hand_region(frame, inv_affine_matrix):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped = cv2.warpAffine(rgb, inv_affine_matrix,
                             (LM_INPUT_SIZE, LM_INPUT_SIZE),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    cropped = cv2.flip(cropped, 0)  # FIX: vertical flip required
    tensor  = cropped.astype(np.float32) / 255.0
    tensor  = np.expand_dims(tensor, axis=0)
    return tensor


def send_tensor_to_pynq(input_tensor):
    lm_interpreter.set_tensor(lm_in, input_tensor)
    lm_interpreter.invoke()
    landmarks       = lm_interpreter.get_tensor(lm_out_map['Identity']  )[0]
    presence        = lm_interpreter.get_tensor(lm_out_map['Identity_1'])[0]
    handedness      = lm_interpreter.get_tensor(lm_out_map['Identity_2'])[0]
    world_landmarks = lm_interpreter.get_tensor(lm_out_map['Identity_3'])[0]
    return [landmarks, presence, handedness, world_landmarks]


PRESENCE_THRESHOLD = 0.5

# CHANGED — postprocess_landmarks: undo vertical flip on y coords
def postprocess_landmarks(raw_outputs, affine_matrix, frame_W, frame_H):
    presence   = float(raw_outputs[1][0])
    handedness = float(raw_outputs[2][0])
    if presence < PRESENCE_THRESHOLD:
        return None
    lms_roi   = raw_outputs[0].reshape(21, 3).copy()
    world_lms = raw_outputs[3].reshape(21, 3)
    lms_roi[:, 1] = LM_INPUT_SIZE - lms_roi[:, 1]  # FIX: undo vertical flip
    xy_roi = lms_roi[:, :2]
    ones   = np.ones((21, 1), dtype=np.float32)
    xy_h   = np.hstack([xy_roi, ones])
    xy_px  = (affine_matrix @ xy_h.T).T
    landmarks_px   = np.hstack([xy_px, lms_roi[:, 2:3]])
    landmarks_norm = landmarks_px.copy()
    landmarks_norm[:, 0] /= frame_W
    landmarks_norm[:, 1] /= frame_H
    return landmarks_px, landmarks_norm, presence, handedness, world_lms


class _Landmark:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class _HandResult:
    def __init__(self):
        self.hand_landmarks = []

def build_result(all_hand_data):
    result = _HandResult()
    for hand in all_hand_data:
        if hand is None:
            continue
        landmarks_norm = hand[1]
        lm_list = [_Landmark(float(landmarks_norm[i, 0]),
                             float(landmarks_norm[i, 1]),
                             float(landmarks_norm[i, 2]))
                   for i in range(21)]
        result.hand_landmarks.append(lm_list)
    return result


tracked_rois = []

# NEW — replaces update_tracked_roi_from_landmarks
class SmoothedROI:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cx = self.cy = self.size = self.angle = None
        self._size_buf = []

    def update(self, landmarks_px):
        wrist     = landmarks_px[0,  :2]
        mid_mcp   = landmarks_px[9,  :2]
        idx_mcp   = landmarks_px[5,  :2]
        pinky_mcp = landmarks_px[17, :2]

        # stable centre: mean of knuckle bases + wrist
        palm_pts = landmarks_px[[0,5,9,13,17], :2]
        cx = float(np.mean(palm_pts[:,0]))
        cy = float(np.mean(palm_pts[:,1]))

        # stable angle: wrist -> middle MCP
        dx    = float(mid_mcp[0] - wrist[0])
        dy    = float(mid_mcp[1] - wrist[1])
        angle = np.arctan2(dy, dx) - np.pi / 2

        # stable size: palm width invariant to finger pose
        palm_width = float(np.linalg.norm(idx_mcp - pinky_mcp))
        size = palm_width * 3.5

        # 5-frame median to kill size spikes
        self._size_buf.append(size)
        if len(self._size_buf) > 5:
            self._size_buf.pop(0)
        size = float(np.median(self._size_buf))

        if self.cx is None:
            self.cx    = cx;   self.cy    = cy
            self.size  = size; self.angle = angle
        else:
            self.cx   = POS_SMOOTH  * self.cx   + (1-POS_SMOOTH)  * cx
            self.cy   = POS_SMOOTH  * self.cy   + (1-POS_SMOOTH)  * cy
            self.size = SIZE_SMOOTH * self.size + (1-SIZE_SMOOTH) * size
            diff = angle - self.angle
            if diff >  np.pi: diff -= 2*np.pi
            if diff < -np.pi: diff += 2*np.pi
            self.angle = self.angle + (1-ANGLE_SMOOTH) * diff

        return build_affine(self.cx, self.cy, self.size, self.angle)

# one smoother per hand slot — put this near the other state variables
smoothers = [SmoothedROI() for _ in range(MAX_HANDS)]


def send_stroke_to_pynq(points):
    data    = {"stroke": [[int(p[0]), int(p[1])] for p in points]}
    payload = json.dumps(data).encode("utf-8")
    sock.sendto(payload, pynq_addr)
    print(f"[udp] sent stroke with {len(points)} points to PYNQ")


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
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return a, b
    ang     = np.degrees(np.arctan2(dy, dx))
    snapped = round(ang / angle_step_deg) * angle_step_deg
    rad     = np.radians(snapped)
    bx2     = int(round(ax + length * np.cos(rad)))
    by2     = int(round(ay + length * np.sin(rad)))
    return a, (bx2, by2)

def resample_polyline(points, step=8.0):
    if len(points) < 2:
        return points[:]
    out  = [points[0]]
    acc  = 0.0
    prev = np.array(points[0], dtype=float)
    for p in points[1:]:
        cur = np.array(p, dtype=float)
        seg = np.linalg.norm(cur - prev)
        if seg < 1e-6:
            continue
        while acc + seg >= step:
            t    = (step - acc) / seg
            newp = prev + t * (cur - prev)
            out.append((int(round(newp[0])), int(round(newp[1]))))
            prev = newp
            seg  = np.linalg.norm(cur - prev)
            acc  = 0.0
        acc += seg
        prev = cur
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out

def rdp(points, eps):
    if len(points) < 3:
        return points
    a   = np.array(points[0],  dtype=float)
    b   = np.array(points[-1], dtype=float)
    ab  = b - a
    ab2 = float(ab @ ab)
    max_d = -1.0
    idx   = -1
    for i in range(1, len(points) - 1):
        p = np.array(points[i], dtype=float)
        if ab2 < 1e-9:
            d = np.linalg.norm(p - a)
        else:
            t    = float(((p - a) @ ab) / ab2)
            proj = a + np.clip(t, 0.0, 1.0) * ab
            d    = np.linalg.norm(p - proj)
        if d > max_d:
            max_d = d
            idx   = i
    if max_d > eps:
        left  = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]

def fit_circle_kasa(points):
    pts = np.array(points, dtype=float)
    x   = pts[:, 0];  y = pts[:, 1]
    A   = np.column_stack([x, y, np.ones_like(x)])
    b   = x * x + y * y
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx   = 0.5 * c[0];  cy = 0.5 * c[1]
    r    = np.sqrt(max(1e-9, cx*cx + cy*cy + c[2]))
    d    = np.sqrt((x - cx)**2 + (y - cy)**2)
    rmse = float(np.sqrt(np.mean((d - r)**2)))
    return float(cx), float(cy), float(r), rmse

def angle_deg(u, v):
    u  = np.array(u, dtype=float);  v = np.array(v, dtype=float)
    nu = np.linalg.norm(u);          nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def try_rectangle(points, eps=12.0, right_angle_tol=50.0):
    simp = rdp(points, eps)
    if len(simp) >= 2 and simp[0] == simp[-1]:
        simp = simp[:-1]
    while len(simp) > 4:
        n    = len(simp)
        lens = []
        for i in range(n):
            a = np.array(simp[i],           float)
            b = np.array(simp[(i + 1) % n], float)
            lens.append(np.linalg.norm(b - a))
        simp.pop(int(np.argmin(lens)))
    if len(simp) != 4:
        return None
    for i in range(4):
        p_prev = np.array(simp[(i - 1) % 4], dtype=float)
        p      = np.array(simp[i],           dtype=float)
        p_next = np.array(simp[(i + 1) % 4], dtype=float)
        ang    = angle_deg(p_prev - p, p_next - p)
        if abs(ang - 90.0) > right_angle_tol:
            return None
    return simp

def draw_circle(canvas, cx, cy, r, thickness=3, fill=False):
    center = (int(round(cx)), int(round(cy)))
    rr     = int(round(r))
    if fill:
        cv2.circle(canvas, center, rr, (255, 255, 255), -1)
    cv2.circle(canvas, center, rr, (255, 255, 255), thickness)

def draw_rect(canvas, corners, thickness=3, fill=False):
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    if fill:
        cv2.fillPoly(canvas, [pts], (255, 255, 255))
    cv2.polylines(canvas, [pts], True, (255, 255, 255), thickness)

def to_world(p_screen, scale, ox, oy):
    return (int(round((p_screen[0] - ox) / scale)), int(round((p_screen[1] - oy) / scale)))

def to_screen(p_world, scale, ox, oy):
    return (int(round(p_world[0] * scale + ox)), int(round(p_world[1] * scale + oy)))

def snap_to_grid(p, g):
    return (int(round(p[0] / g) * g), int(round(p[1] / g) * g))

def is_open_palm(hand, W, H):
    wrist = np.array([hand[0].x * W, hand[0].y * H], dtype=float)
    tips  = [4, 8, 12, 16, 20]
    d     = []
    for i in tips:
        p = np.array([hand[i].x * W, hand[i].y * H], dtype=float)
        d.append(np.linalg.norm(p - wrist))
    return float(np.mean(d)) > 170.0


# =========================================================
# Params
# =========================================================
SMOOTH_ALPHA         = 0.25
PINCH_THRESH_PX      = 80
PAUSE_SPEED_PX       = 30.0
PAUSE_TIME_S         = 0.25
SNAP_WINDOW          = 20
LINE_TOL_PX          = 6.0
CLOSE_THRESH_PX      = 60.0
ANGLE_STEP_DEG       = 45
SHAPE_PAUSE_TIME     = 0.35
CIRCLE_REL_TOL       = 0.18
RECT_RIGHT_ANGLE_TOL = 50.0
ERASER_RADIUS_WORLD  = 35
GRID_SIZE            = 25
SHOW_GRID            = False
GRID_SNAP            = False
VIEW_SCALE_MIN       = 0.6
VIEW_SCALE_MAX       = 3.0


# =========================================================
# Camera
# =========================================================
def open_camera():
    for idx in range(2):
        cap = cv2.VideoCapture(idx)
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
# Flask
# =========================================================
app = Flask(__name__)
latest_frame = None

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_server():
    app.run(host="0.0.0.0", port=8000, threaded=True)

threading.Thread(target=run_server, daemon=True).start()


# =========================================================
# State
# =========================================================
canvas           = None
current          = []
strokes          = []
still_time       = 0.0
last_t           = None
last_pt_screen   = None
pen_down_prev    = False
xf = yf          = None
shape_buffer     = None
shape_timer      = 0.0
undo_stack       = []
redo_stack       = []
eraser_active_prev = False
view_scale       = 1.0
view_ox          = 0.0
view_oy          = 0.0
twohand_prev_mid  = None
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
    H, W  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    t = time.time()
    if last_t is None:
        last_t = t
    dt     = t - last_t
    last_t = t

    # ---- palm detection or use tracked ROIs ----
 # CHANGED — main loop tracking block
# Replace everything from "---- palm detection or use tracked ROIs ----"
# through to "---- build result object ----" with this:

    # ---- palm detection when we don't have enough hands ----
    if len(tracked_rois) < MAX_HANDS:
        for i in range(len(tracked_rois), MAX_HANDS):
            smoothers[i].reset()
        pd_tensor, pad, sq = preprocess_palm_detector(frame)
        regressors, scores_raw = run_palm_detector(pd_tensor)
        detections = decode_detections(regressors, scores_raw, pad)
        tracked_rois = []
        for i, det in enumerate(detections):
            affine, inv_affine = compute_roi_affine(det, W, H)
            tracked_rois.append((affine, inv_affine, i))

    # ---- landmark inference + postprocessing ----
    all_hand_data    = []
    new_tracked_rois = []

    for affine, inv_affine, hand_idx in tracked_rois:
        lm_tensor   = crop_hand_region(frame, inv_affine)
        raw_outputs = send_tensor_to_pynq(lm_tensor)
        hand_data   = postprocess_landmarks(raw_outputs, affine, W, H)

        if hand_data is not None:
            all_hand_data.append(hand_data)
            new_affine, new_inv = smoothers[hand_idx].update(hand_data[0])
            new_tracked_rois.append((new_affine, new_inv, hand_idx))
        else:
            smoothers[hand_idx].reset()

    tracked_rois = new_tracked_rois

    # ---- build result object ----
    result = build_result(all_hand_data)

    cursor_screen = None
    pen_down      = False
    hands_n       = 0
    open_palm_1   = False
    open_palm_2   = False
    hand1         = None
    hand2         = None

    if result.hand_landmarks:
        hands_n     = len(result.hand_landmarks)
        hand1       = result.hand_landmarks[0]
        open_palm_1 = is_open_palm(hand1, W, H)

        ix, iy = hand1[8].x, hand1[8].y
        tx, ty = hand1[4].x, hand1[4].y
        cursor_screen = (int(ix * W), int(iy * H))

        pinch_px = np.hypot((ix - tx) * W, (iy - ty) * H)
        pen_down = pinch_px < PINCH_THRESH_PX

        if hands_n >= 2:
            hand2       = result.hand_landmarks[1]
            open_palm_2 = is_open_palm(hand2, W, H)

    # =====================================================
    # Two-hand zoom/pan
    # =====================================================
    if hands_n >= 2 and open_palm_1 and open_palm_2 and (not pen_down):
        p1  = (int(hand1[8].x * W), int(hand1[8].y * H))
        p2  = (int(hand2[8].x * W), int(hand2[8].y * H))
        mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        d12 = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

        if twohand_prev_mid is not None and twohand_prev_dist is not None:
            dx      = mid[0] - twohand_prev_mid[0]
            dy      = mid[1] - twohand_prev_mid[1]
            view_ox += dx
            view_oy += dy

            if twohand_prev_dist > 1e-6:
                ratio      = d12 / twohand_prev_dist
                new_scale  = float(np.clip(view_scale * ratio, VIEW_SCALE_MIN, VIEW_SCALE_MAX))
                wx         = (mid[0] - view_ox) / max(1e-6, view_scale)
                wy         = (mid[1] - view_oy) / max(1e-6, view_scale)
                view_scale = new_scale
                view_ox    = mid[0] - wx * view_scale
                view_oy    = mid[1] - wy * view_scale

        twohand_prev_mid  = mid
        twohand_prev_dist = d12
    else:
        twohand_prev_mid  = None
        twohand_prev_dist = None

    # =====================================================
    # Smooth cursor
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

    if cursor_f_screen is not None and last_pt_screen is not None and dt > 1e-6:
        speed = dist(cursor_f_screen, last_pt_screen) / dt
    else:
        speed = 0.0
    if cursor_f_screen is not None:
        last_pt_screen = cursor_f_screen

    cursor_world = None
    if cursor_f_screen is not None:
        cursor_world = to_world(cursor_f_screen, view_scale, view_ox, view_oy)
        if GRID_SNAP and pen_down:
            cursor_world = snap_to_grid(cursor_world, GRID_SIZE)

    # =====================================================
    # Eraser
    # =====================================================
    eraser_active = (hands_n >= 1 and open_palm_1 and (not pen_down) and cursor_world is not None)

    if eraser_active and (not eraser_active_prev):
        undo_stack.append(canvas.copy())
        redo_stack.clear()

    if eraser_active and cursor_world is not None:
        cv2.circle(canvas, cursor_world, ERASER_RADIUS_WORLD, (0, 0, 0), -1)

    eraser_active_prev = eraser_active

    # =====================================================
    # Drawing
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

        if still_time >= PAUSE_TIME_S and len(current) >= SNAP_WINDOW:
            seg   = current[-SNAP_WINDOW:]
            a, b  = seg[0], seg[-1]
            max_d = max(point_line_distance(p, a, b) for p in seg)
            if max_d <= LINE_TOL_PX:
                a2, b2  = snap_angle_endpoint(a, b, ANGLE_STEP_DEG)
                current = current[:-SNAP_WINDOW] + [a2, b2]
            still_time = 0.0

    # =====================================================
    # Pen-up
    # =====================================================
    if (not pen_down) and pen_down_prev and current:
        shape_buffer = current[:]
        shape_timer  = time.time()
        strokes.append(current)
        print("[stroke] finished drawing")
        send_stroke_to_pynq(current)
        current    = []
        still_time = 0.0

    pen_down_prev = pen_down

    # =====================================================
    # Shape recognition
    # =====================================================
    shape_label = "none"
    if shape_buffer is not None and (time.time() - shape_timer) >= SHAPE_PAUSE_TIME:
        pts = resample_polyline(shape_buffer, step=8.0)

        closed = (len(pts) >= 10 and dist(pts[0], pts[-1]) < CLOSE_THRESH_PX)
        if closed and pts[0] != pts[-1]:
            pts.append(pts[0])

        xs    = [p[0] for p in pts];  ys = [p[1] for p in pts]
        scale = max(20.0, max(max(xs) - min(xs), max(ys) - min(ys)))

        drew = False
        undo_stack.append(canvas.copy())
        redo_stack.clear()

        if closed and len(pts) >= 25:
            fit = fit_circle_kasa(pts[:-1])
            if fit is not None:
                cx, cy, r, rmse = fit
                rel = rmse / max(1.0, r)
                if 15 <= r <= 0.9 * scale and rel <= CIRCLE_REL_TOL:
                    draw_circle(canvas, cx, cy, r, thickness=3, fill=True)
                    shape_label = "circle"
                    drew        = True

        if (not drew) and closed:
            eps  = 0.06 * scale
            rect = try_rectangle(pts, eps=eps, right_angle_tol=RECT_RIGHT_ANGLE_TOL)
            if rect is not None:
                draw_rect(canvas, rect, thickness=3, fill=True)
                shape_label = "rect"
                drew        = True

        if not drew:
            for i in range(1, len(shape_buffer)):
                cv2.line(canvas, shape_buffer[i - 1], shape_buffer[i], (255, 255, 255), 3)
            shape_label = "stroke"

        shape_buffer = None

    # =====================================================
    # Display
    # =====================================================
    display = frame.copy()

    M = np.array([[view_scale, 0, view_ox],
                  [0, view_scale, view_oy]], dtype=np.float32)
    warped_canvas = cv2.warpAffine(canvas, M, (W, H),
                                   flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))

    mask = warped_canvas[:, :, 0] > 0
    display[mask] = warped_canvas[mask]

    if SHOW_GRID:
        step = GRID_SIZE
        for x in range(0, W, step):
            cv2.line(display, (x, 0), (x, H), (30, 30, 30), 1)
        for y in range(0, H, step):
            cv2.line(display, (0, y), (W, y), (30, 30, 30), 1)

    if len(current) >= 2:
        for i in range(1, len(current)):
            a = to_screen(current[i - 1], view_scale, view_ox, view_oy)
            b = to_screen(current[i],     view_scale, view_ox, view_oy)
            cv2.line(display, a, b, (200, 200, 200), 2)

    if cursor_f_screen is not None:
        color = (0, 255, 0) if pen_down else (0, 0, 255)
        if eraser_active:
            color = (255, 0, 0)
        cv2.circle(display, cursor_f_screen, 8, color, -1)

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

    cv2.putText(display,
                f"PYNQ -> {PYNQ_IP}:{PYNQ_PORT}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    latest_frame = display

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    if k in (ord('c'), ord('C')):
        undo_stack.append(canvas.copy())
        redo_stack.clear()
        canvas[:]    = 0
        strokes.clear()
        current      = []
        shape_buffer = None
        still_time   = 0.0

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

cap.release()
cv2.destroyAllWindows()