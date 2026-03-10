import cv2
import numpy as np
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import socket 
import json

from flask import Flask, Response
import threading

import tflite_runtime.interpreter as tflite   # or tensorflow.lite

PYNQ_IP = "192.168.2.99"
PYNQ_PORT = 5005      # UDP - stroke analysis
PYNQ_TCP_PORT = 5006  # TCP - TFLite inference

# UDP socket (stroke analysis)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pynq_addr = (PYNQ_IP, PYNQ_PORT)

# TCP socket (TFLite inference)
sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_tcp.connect((PYNQ_IP, PYNQ_TCP_PORT))


# =========================================================
# Helpers
# =========================================================

# stage 1
PALM_INPUT_SIZE = 192

def preprocess_palm_detector(frame):
    """
    Input:  BGR frame from camera e.g. (720, 1280, 3)
    Output: float32 tensor [1, 192, 192, 3] in RGB [-1, 1]
            letterbox padding (left, top, right, bottom) normalised to [0,1]
    """
    H, W = frame.shape[:2]

    # convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # letterbox: pad shorter side to make square, keeping aspect ratio
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

    # resize to 192x192
    resized = cv2.resize(padded, (PALM_INPUT_SIZE, PALM_INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)

    # normalise to [-1, 1] as float32
    tensor = (resized.astype(np.float32) / 127.5) - 1.0
    tensor = np.expand_dims(tensor, axis=0)   # [1, 192, 192, 3]

    # padding fractions (needed later to map detections back to frame coords)
    pad = (
        pad_left   / sq,   # left
        pad_top    / sq,   # top
        pad_right  / sq,   # right
        pad_bottom / sq    # bottom
    )

    return tensor, pad, sq   # sq = side length of padded square in pixels

# stage 2

# ---- anchor generation (exact MediaPipe params) ----
def generate_anchors():
    """
    Produces 2016 anchors matching MediaPipe's SsdAnchorsCalculator.
    num_layers=4, strides=[8,16,16,16], input=192x192,
    fixed_anchor_size=True, anchor_offset=0.5
    """
    strides      = [8, 16, 16, 16]
    input_h = input_w = PALM_INPUT_SIZE
    anchors = []

    for stride in strides:
        rows = int(np.ceil(input_h / stride))
        cols = int(np.ceil(input_w / stride))
        for r in range(rows):
            for c in range(cols):
                # two anchors per cell (aspect_ratios=[1.0],
                # interpolated_scale_aspect_ratio=1.0)
                for _ in range(2):
                    cx = (c + 0.5) / cols
                    cy = (r + 0.5) / rows
                    anchors.append([cx, cy])

    return np.array(anchors, dtype=np.float32)   # [2016, 2]

ANCHORS = generate_anchors()

# ---- run palm detector ----
palm_interpreter = tflite.Interpreter(model_path="extracted_models/hand_detector.tflite")
palm_interpreter.allocate_tensors()
palm_in  = palm_interpreter.get_input_details()[0]['index']
palm_out = palm_interpreter.get_output_details()
# output 0: regressors [1, 2016, 18]
# output 1: scores     [1, 2016, 1]

lm_interpreter = tflite.Interpreter(model_path="extracted_models/hand_landmarks_detector.tflite")
lm_interpreter.allocate_tensors()
lm_in  = lm_interpreter.get_input_details()[0]['index']
lm_out = lm_interpreter.get_output_details()
# map output names to indices
lm_out_map = {d['name']: d['index'] for d in lm_out}

def run_palm_detector(tensor):
    palm_interpreter.set_tensor(palm_in, tensor)
    palm_interpreter.invoke()
    regressors = palm_interpreter.get_tensor(palm_out[0]['index'])[0]  # [2016, 18]
    scores_raw = palm_interpreter.get_tensor(palm_out[1]['index'])[0, :, 0]  # [2016]
    return regressors, scores_raw

# ---- decode + NMS ----
SCORE_THRESH = 0.5
NMS_THRESH   = 0.3
X_SCALE = Y_SCALE = W_SCALE = H_SCALE = 192.0

def decode_detections(regressors, scores_raw, pad):
    """
    Returns list of dicts, each with:
        cx, cy, w, h  (normalised to original frame, 0..1)
        kps           (7 keypoints, each [x, y] normalised to original frame)
        score
    """
    # sigmoid scores
    scores = 1.0 / (1.0 + np.exp(-scores_raw))

    keep = np.where(scores > SCORE_THRESH)[0]
    if len(keep) == 0:
        return []

    kp_scores = scores[keep]
    kp_regs   = regressors[keep]   # [N, 18]
    kp_anch   = ANCHORS[keep]      # [N, 2]

    # decode box centre and size (all in [0,1] relative to 192x192 input)

    ANCHORS_WH = np.ones((len(ANCHORS), 2), dtype=np.float32)  # w=1, h=1

    cx = kp_regs[:, 0] / X_SCALE * ANCHORS_WH[keep, 0] + kp_anch[:, 0]
    cy = kp_regs[:, 1] / Y_SCALE * ANCHORS_WH[keep, 1] + kp_anch[:, 1]
    w  = kp_regs[:, 2] / W_SCALE * ANCHORS_WH[keep, 0]
    h  = kp_regs[:, 3] / H_SCALE * ANCHORS_WH[keep, 1]
    
    # decode 7 keypoints (indices 4..17, pairs)
    kps = []
    for i in range(7):
        kx = kp_regs[:, 4 + 2*i] / X_SCALE * ANCHORS_WH[keep, 0] + kp_anch[:, 0]
        ky = kp_regs[:, 4 + 2*i + 1] / Y_SCALE * ANCHORS_WH[keep, 1] + kp_anch[:, 1]
        kps.append(np.stack([kx, ky], axis=1))
    kps = np.stack(kps, axis=1)   # [N, 7, 2]

    # NMS using cv2 (expects pixel boxes; multiply by 1000 as int trick)
    pad_l, pad_t, pad_r, pad_b = pad
    boxes_nms = []
    for i in range(len(keep)):
        x1 = int((cx[i] - w[i]/2) * 1000)
        y1 = int((cy[i] - h[i]/2) * 1000)
        bw = int(w[i] * 1000)
        bh = int(h[i] * 1000)
        boxes_nms.append([x1, y1, bw, bh])

    indices = cv2.dnn.NMSBoxes(boxes_nms,
                               kp_scores.tolist(),
                               0,
                               NMS_THRESH)
    if len(indices) == 0:
        return []

    detections = []
    for idx in indices.flatten():
        # map from letterboxed [0,1] back to original frame [0,1]
        # by removing the padding fraction
        def unpad_x(v):
            return (v - pad_l) / (1.0 - pad_l - pad_r)
        def unpad_y(v):
            return (v - pad_t) / (1.0 - pad_t - pad_b)

        detections.append({
            'cx':    unpad_x(float(cx[idx])),
            'cy':    unpad_y(float(cy[idx])),
            'w':     float(w[idx])  / (1.0 - pad_l - pad_r),
            'h':     float(h[idx])  / (1.0 - pad_t - pad_b),
            'kps':   np.stack([unpad_x(kps[idx,:,0]),
                               unpad_y(kps[idx,:,1])], axis=1),
            'score': float(kp_scores[idx])
        })

    return detections

ROI_SCALE = 2.6   # how much larger than the palm box to make the crop

def compute_roi_affine(detection, frame_W, frame_H):
    """
    Replicates MediaPipe's DetectionsToRectsCalculator +
    RectTransformationCalculator.

    Rotation: aligns wrist (kp 0) -> middle-finger MCP (kp 2) with Y axis.
    Returns:
        affine_matrix      (2x3 float32) — maps ROI space -> frame pixels
        inv_affine_matrix  (2x3 float32) — maps frame pixels -> ROI space
                           (used later to crop the 224x224 input tensor)
    """
    kps = detection['kps']   # [7, 2] in normalised frame coords [0,1]

    # wrist and middle-MCP in pixel coords
    wrist_x  = kps[0, 0] * frame_W;  wrist_y  = kps[0, 1] * frame_H
    mcp_x    = kps[2, 0] * frame_W;  mcp_y    = kps[2, 1] * frame_H

    # rotation angle so wrist->MCP aligns with Y axis
    angle = np.arctan2(mcp_x - wrist_x, mcp_y - wrist_y) - np.pi / 2   # note: x/y order

    # ROI centre = detection box centre in pixels
    cx = detection['cx'] * frame_W
    cy = detection['cy'] * frame_H

    # ROI size = max(w,h) * ROI_SCALE in pixels
    size = max(detection['w'] * frame_W,
               detection['h'] * frame_H) * ROI_SCALE

    # build affine: rotation + scale + translation
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # 3 destination points of the square ROI in frame pixel space
    # (top-left, top-right, bottom-left) of a size×size square centred at cx,cy
    half = size / 2.0
    src_pts = np.float32([
        [-half, -half],
        [ half, -half],
        [-half,  half],
    ])
    # rotate each point and translate to centre
    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]], dtype=np.float32)
    dst_pts = (rot @ src_pts.T).T + np.array([cx, cy], dtype=np.float32)

    # destination in 224x224 space
    lm_input = 224.0
    dst_224 = np.float32([
        [0,          0       ],
        [lm_input,   0       ],
        [0,          lm_input],
    ])

    # affine: 224 space -> frame pixel space
    affine_matrix     = cv2.getAffineTransform(dst_224, dst_pts)
    # inverse: frame pixel space -> 224 space  (used for cropping)
    inv_affine_matrix = cv2.getAffineTransform(dst_pts, dst_224)

    return affine_matrix, inv_affine_matrix

LM_INPUT_SIZE = 224

def crop_hand_region(frame, inv_affine_matrix):
    """
    Applies the inverse affine transform to warp the hand region
    into a 224x224 image, then normalises to [0, 1] float32 RGB.
    This is the tensor you send to the PYNQ.
    """
    # warpAffine with the inverse matrix maps the rotated/scaled
    # hand region into the 224x224 output image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cropped = cv2.warpAffine(rgb, inv_affine_matrix,
                             (LM_INPUT_SIZE, LM_INPUT_SIZE),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)

    # normalise to [0, 1] float32 — landmark model expects this
    tensor = cropped.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)   # [1, 224, 224, 3]

    return tensor

def send_tensor_to_pynq(input_tensor):
#    """Send [1,224,224,3] float32 tensor, receive 4 output tensors."""
#    payload = input_tensor.astype(np.float32).tobytes()

#   sock_tcp.sendall(len(payload).to_bytes(4, 'big'))
#    sock_tcp.sendall(payload)

#    def recv_exact(n):
#        buf = b''
#        while len(buf) < n:
#            chunk = sock_tcp.recv(n - len(buf))
#            if not chunk:
#                raise ConnectionError("PYNQ disconnected")
#            buf += chunk
#        return buf

#    out_sizes = [63, 1, 1, 63]
#    outputs = []
#    for s in out_sizes:
#        raw = recv_exact(s * 4)
#        outputs.append(np.frombuffer(raw, dtype=np.float32))

#   return outputs  # [landmarks, presence, handedness, world_landmarks]

        """
    Temporary: runs the landmark tflite model locally instead of on PYNQ.
    Drop-in replacement — same inputs and outputs as the real function.
    """
    lm_interpreter.set_tensor(lm_in, input_tensor)
    lm_interpreter.invoke()

    landmarks      = lm_interpreter.get_tensor(lm_out_map['Identity']  )[0]   # [63]
    presence       = lm_interpreter.get_tensor(lm_out_map['Identity_1'])[0]   # [1]
    handedness     = lm_interpreter.get_tensor(lm_out_map['Identity_2'])[0]   # [1]
    world_landmarks = lm_interpreter.get_tensor(lm_out_map['Identity_3'])[0]  # [63]

    return [landmarks, presence, handedness, world_landmarks]

PRESENCE_THRESHOLD = 0.5

def postprocess_landmarks(raw_outputs, affine_matrix, frame_W, frame_H):
    """
    raw_outputs: list of 4 arrays returned by send_tensor_to_pynq:
        [landmarks [63], presence [1], handedness [1], world_landmarks [63]]
    affine_matrix: 2x3 matrix from compute_roi_affine (ROI space -> frame pixels)

    Returns:
        landmarks_px   [21, 3]  x,y in frame pixels, z raw
        landmarks_norm [21, 3]  x,y normalised to [0,1] by frame size
        presence       float
        handedness     float    (>0.5 = right hand)
        world_lms      [21, 3]  metric world coords (origin = hand centre)
    """
    presence   = float(raw_outputs[1][0])
    handedness = float(raw_outputs[2][0])

    if presence < PRESENCE_THRESHOLD:
        return None

    # reshape [63] -> [21, 3]
    lms_roi       = raw_outputs[0].reshape(21, 3)   # x,y in [0,224], z raw
    world_lms     = raw_outputs[3].reshape(21, 3)

    # map x,y from 224x224 ROI space back to frame pixel space
    # using the affine_matrix (2x3): dst = M * [x, y, 1]^T
    xy_roi  = lms_roi[:, :2]                          # [21, 2]
    ones    = np.ones((21, 1), dtype=np.float32)
    xy_h    = np.hstack([xy_roi, ones])               # [21, 3] homogeneous
    xy_px   = (affine_matrix @ xy_h.T).T              # [21, 2] frame pixels

    landmarks_px = np.hstack([xy_px,
                               lms_roi[:, 2:3]])      # [21, 3]

    # normalise x,y to [0,1]
    landmarks_norm = landmarks_px.copy()
    landmarks_norm[:, 0] /= frame_W
    landmarks_norm[:, 1] /= frame_H

    return landmarks_px, landmarks_norm, presence, handedness, world_lms

class _Landmark:
    """Mimics mediapipe NormalizedLandmark with .x .y .z attributes."""
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class _HandResult:
    """Mimics the object returned by landmarker.detect_for_video."""
    def __init__(self):
        self.hand_landmarks = []   # list of lists of _Landmark (one per hand)

def build_result(all_hand_data):
    """
    all_hand_data: list of postprocess_landmarks() return values,
                   one entry per detected hand (up to 2).
    Returns a _HandResult whose .hand_landmarks matches what your
    existing code expects.
    """
    result = _HandResult()
    for hand in all_hand_data:
        if hand is None:
            continue
        landmarks_norm = hand[1]   # [21, 3] normalised
        lm_list = [_Landmark(float(landmarks_norm[i, 0]),
                             float(landmarks_norm[i, 1]),
                             float(landmarks_norm[i, 2]))
                   for i in range(21)]
        result.hand_landmarks.append(lm_list)
    return result

tracked_rois = []   # list of (affine_matrix, inv_affine_matrix) per hand

def update_tracked_roi_from_landmarks(landmarks_px, frame_W, frame_H):
    """
    Derive next frame's ROI directly from current landmarks,
    using wrist (0) and middle-MCP (9) — same as MediaPipe.
    Returns (affine_matrix, inv_affine_matrix).
    """
    wrist = landmarks_px[0, :2]
    mcp   = landmarks_px[9, :2]

    angle = np.arctan2(mcp[0] - wrist[0], mcp[1] - wrist[1]) - np.pi / 2

    cx = float(np.mean(landmarks_px[:, 0]))
    cy = float(np.mean(landmarks_px[:, 1]))

    xs = landmarks_px[:, 0];  ys = landmarks_px[:, 1]
    size = max(np.max(xs) - np.min(xs),
               np.max(ys) - np.min(ys)) * ROI_SCALE

    cos_a = np.cos(angle);  sin_a = np.sin(angle)
    half  = size / 2.0
    src_pts = np.float32([[-half,-half],[half,-half],[-half,half]])
    rot     = np.array([[cos_a,-sin_a],[sin_a,cos_a]], dtype=np.float32)
    dst_pts = (rot @ src_pts.T).T + np.array([cx, cy], dtype=np.float32)
    dst_224 = np.float32([[0,0],[224,0],[0,224]])

    affine     = cv2.getAffineTransform(dst_224, dst_pts)
    inv_affine = cv2.getAffineTransform(dst_pts, dst_224)
    return affine, inv_affine

def send_stroke_to_pynq(points): 
    #Send stroke points to the pynq as JSON

    data = { 
        "stroke": [[int(p[0]), int(p[1])] for p in points]
    }

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
SHOW_GRID = False
GRID_SNAP = False

# Zoom/pan
VIEW_SCALE_MIN = 0.6
VIEW_SCALE_MAX = 3.0

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
    app.run(host="0.0.0.0", port=8000, threaded=True)

threading.Thread(target=run_server, daemon=True).start()

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
    #print("main loop running")

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

    H, W = frame.shape[:2]
    
    # ---- palm detection or use tracked ROIs ----
    if not tracked_rois:
        pd_tensor, pad, sq = preprocess_palm_detector(frame)
        regressors, scores_raw = run_palm_detector(pd_tensor)
        detections = decode_detections(regressors, scores_raw, pad)
        tracked_rois = []
        for det in detections:
            affine, inv_affine = compute_roi_affine(det, W, H)
            tracked_rois.append((affine, inv_affine))
    
    # ---- landmark inference + postprocessing ----
    all_hand_data = []
    new_tracked_rois = []
    
    for affine, inv_affine in tracked_rois:
        lm_tensor = crop_hand_region(frame, inv_affine)
        raw_outputs = send_tensor_to_pynq(lm_tensor)
        hand_data = postprocess_landmarks(raw_outputs, affine, W, H)
    
        if hand_data is not None:
            all_hand_data.append(hand_data)
            new_affine, new_inv_affine = update_tracked_roi_from_landmarks(hand_data[0], W, H)
            new_tracked_rois.append((new_affine, new_inv_affine))
        # if hand_data is None, presence was low — drop this ROI
    
    tracked_rois = new_tracked_rois
    
    # ---- build result object ----
    result = build_result(all_hand_data)

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

        print("[stroke] finished drawing")
        send_stroke_to_pynq(current)

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
    cv2.putText(display,
            f"PYNQ → {PYNQ_IP}:{PYNQ_PORT}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2)

    latest_frame = display

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
