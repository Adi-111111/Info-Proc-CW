import cv2
import numpy as np
import tensorflow as tf

tflite = tf.lite

PALM_INPUT_SIZE  = 192
LM_INPUT_SIZE    = 224
ROI_SCALE        = 1.5
SCORE_THRESH     = 0.5
NMS_THRESH       = 0.3
PRESENCE_THRESH  = 0.5
MAX_HANDS        = 2

POS_SMOOTH   = 0.5
SIZE_SMOOTH  = 0.7
ANGLE_SMOOTH = 0.5

# ── Anchors ────────────────────────────────────────────────────────────────────

def generate_anchors():
    strides = [8, 16, 16, 16]
    anchors = []
    for stride in strides:
        rows = cols = int(np.ceil(PALM_INPUT_SIZE / stride))
        for r in range(rows):
            for c in range(cols):
                for _ in range(2):
                    anchors.append([(c + 0.5) / cols, (r + 0.5) / rows])
    return np.array(anchors, dtype=np.float32)

ANCHORS = generate_anchors()

# ── Models ─────────────────────────────────────────────────────────────────────

palm_interpreter = tflite.Interpreter(
    model_path="extracted_models/hand_detector.tflite", num_threads=4)
palm_interpreter.allocate_tensors()
palm_in  = palm_interpreter.get_input_details()[0]['index']
palm_out = palm_interpreter.get_output_details()

lm_interpreter = tflite.Interpreter(
    model_path="extracted_models/hand_landmarks_detector.tflite", num_threads=4)
lm_interpreter.allocate_tensors()
lm_in      = lm_interpreter.get_input_details()[0]['index']
lm_out_map = {d['name']: d['index']
              for d in lm_interpreter.get_output_details()}

# ── Palm detector ──────────────────────────────────────────────────────────────

def preprocess_palm(frame):
    H, W = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if W > H:
        pad_top = (W-H)//2;  pad_bottom = W-H-pad_top
        pad_left = pad_right = 0;  sq = W
    else:
        pad_left = (H-W)//2;  pad_right = H-W-pad_left
        pad_top = pad_bottom = 0;  sq = H
    padded  = cv2.copyMakeBorder(rgb, pad_top, pad_bottom,
                                      pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (PALM_INPUT_SIZE, PALM_INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    tensor  = resized.astype(np.float32) / 255.0
    pad     = (pad_left/sq, pad_top/sq, pad_right/sq, pad_bottom/sq)
    return np.expand_dims(tensor, 0), pad

def run_palm(tensor):
    palm_interpreter.set_tensor(palm_in, tensor)
    palm_interpreter.invoke()
    regressors = palm_interpreter.get_tensor(palm_out[0]['index'])[0]
    scores_raw = palm_interpreter.get_tensor(palm_out[1]['index'])[0, :, 0]
    return regressors, scores_raw

def decode_detections(regressors, scores_raw, pad):
    scores = 1.0 / (1.0 + np.exp(-scores_raw.clip(-88, 88)))
    keep   = np.where(scores > SCORE_THRESH)[0]
    if len(keep) == 0:
        return []
    kp_scores    = scores[keep]
    kp_regs      = regressors[keep]
    kp_anch      = ANCHORS[keep]
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
    boxes_nms = [[int((cx[i]-w[i]/2)*1000), int((cy[i]-h[i]/2)*1000),
                  int(w[i]*1000), int(h[i]*1000)] for i in range(len(keep))]
    indices = cv2.dnn.NMSBoxes(boxes_nms, kp_scores.tolist(), 0, NMS_THRESH)
    if len(indices) == 0:
        return []
    pad_l, pad_t, pad_r, pad_b = pad
    detections = []
    for idx in indices.flatten():
        def ux(v): return (v - pad_l) / (1.0 - pad_l - pad_r)
        def uy(v): return (v - pad_t) / (1.0 - pad_t - pad_b)
        detections.append({
            'cx':    ux(float(cx[idx])),
            'cy':    uy(float(cy[idx])),
            'w':     float(w[idx])  / (1.0 - pad_l - pad_r),
            'h':     float(h[idx])  / (1.0 - pad_t - pad_b),
            'kps':   np.stack([ux(kps[idx,:,0]), uy(kps[idx,:,1])], axis=1),
            'score': float(kp_scores[idx]),
        })
    return detections[:MAX_HANDS]

# ── ROI ────────────────────────────────────────────────────────────────────────

def build_affine(cx, cy, size, angle):
    cos_a   = np.cos(angle);  sin_a = np.sin(angle)
    half    = size / 2.0
    src_pts = np.float32([[-half,-half],[half,-half],[-half,half]])
    rot     = np.array([[cos_a,-sin_a],[sin_a,cos_a]], dtype=np.float32)
    dst_pts = (rot @ src_pts.T).T + np.array([cx, cy], dtype=np.float32)
    dst_224 = np.float32([[0,0],[LM_INPUT_SIZE,0],[0,LM_INPUT_SIZE]])
    affine     = cv2.getAffineTransform(dst_224, dst_pts)
    inv_affine = cv2.getAffineTransform(dst_pts, dst_224)
    return affine, inv_affine

def compute_roi_from_detection(det, W, H):
    kps   = det['kps']
    dx    = kps[2,0]*W - kps[0,0]*W
    dy    = kps[2,1]*H - kps[0,1]*H
    angle = np.arctan2(dy, dx) - np.pi/2
    cx    = det['cx']*W
    cy    = det['cy']*H
    size  = max(det['w']*W, det['h']*H) * ROI_SCALE
    return build_affine(cx, cy, size, angle)

# ── Crop + landmarks ───────────────────────────────────────────────────────────

def crop_hand(frame, inv_affine):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped = cv2.warpAffine(rgb, inv_affine, (LM_INPUT_SIZE, LM_INPUT_SIZE),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cropped = cv2.flip(cropped, 0)
    return np.expand_dims(cropped.astype(np.float32) / 255.0, 0), cropped

def run_landmarks(tensor):
    lm_interpreter.set_tensor(lm_in, tensor)
    lm_interpreter.invoke()
    return [
        lm_interpreter.get_tensor(lm_out_map['Identity']  )[0],
        lm_interpreter.get_tensor(lm_out_map['Identity_1'])[0],
        lm_interpreter.get_tensor(lm_out_map['Identity_2'])[0],
        lm_interpreter.get_tensor(lm_out_map['Identity_3'])[0],
    ]

def postprocess_landmarks(raw, affine):
    if float(raw[1][0]) < PRESENCE_THRESH:
        return None
    lms_roi = raw[0].reshape(21, 3).copy()
    lms_roi[:, 1] = LM_INPUT_SIZE - lms_roi[:, 1]
    xy_h  = np.hstack([lms_roi[:,:2], np.ones((21,1), dtype=np.float32)])
    xy_px = (affine @ xy_h.T).T
    return np.hstack([xy_px, lms_roi[:,2:3]])

# ── Smoothed ROI ───────────────────────────────────────────────────────────────

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

        # stable centre: mean of the 5 knuckle bases + wrist
        palm_pts = landmarks_px[[0,5,9,13,17], :2]
        cx = float(np.mean(palm_pts[:,0]))
        cy = float(np.mean(palm_pts[:,1]))

        # stable angle: wrist -> middle MCP
        dx    = float(mid_mcp[0] - wrist[0])
        dy    = float(mid_mcp[1] - wrist[1])
        angle = np.arctan2(dy, dx) - np.pi/2

        # stable size: palm width (index MCP -> pinky MCP), scaled to cover hand
        palm_width = float(np.linalg.norm(idx_mcp - pinky_mcp))
        size = palm_width * 3.5

        # 5-frame median to kill size spikes
        self._size_buf.append(size)
        if len(self._size_buf) > 5:
            self._size_buf.pop(0)
        size = float(np.median(self._size_buf))

        if self.cx is None:
            self.cx    = cx
            self.cy    = cy
            self.size  = size
            self.angle = angle
        else:
            self.cx   = POS_SMOOTH  * self.cx   + (1-POS_SMOOTH)  * cx
            self.cy   = POS_SMOOTH  * self.cy   + (1-POS_SMOOTH)  * cy
            self.size = SIZE_SMOOTH * self.size + (1-SIZE_SMOOTH) * size
            diff = angle - self.angle
            if diff >  np.pi: diff -= 2*np.pi
            if diff < -np.pi: diff += 2*np.pi
            self.angle = self.angle + (1-ANGLE_SMOOTH) * diff

        return build_affine(self.cx, self.cy, self.size, self.angle)

# ── Drawing ────────────────────────────────────────────────────────────────────

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
# two distinct colour palettes, one per hand
HAND_COLOURS = [
    [(0,200,255),(0,255,100),(255,140,0),(255,0,140),(80,80,255)],
    [(0,255,200),(200,255,0),(255,80,80),(180,0,255),(255,200,0)],
]

def finger_colours(hand_idx):
    cols = HAND_COLOURS[hand_idx % len(HAND_COLOURS)]
    # wrist, thumb x4, index x4, middle x4, ring x4, pinky x4
    return ([cols[0]] +
            [cols[0]]*4 + [cols[1]]*4 +
            [cols[2]]*4 + [cols[3]]*4 + [cols[4]]*4)

def draw_landmarks(frame, landmarks_px, hand_idx=0):
    pts  = landmarks_px[:,:2].astype(int)
    cols = finger_colours(hand_idx)
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (180,180,180), 1)
    for i, (x, y) in enumerate(pts):
        r = 7 if i in (4,8,12,16,20) else 4
        cv2.circle(frame, (x,y), r, cols[i], -1)
        cv2.circle(frame, (x,y), r, (255,255,255), 1)
    for i, label in [(0,'W'),(4,'T'),(8,'I'),(12,'M'),(16,'R'),(20,'P')]:
        cv2.putText(frame, label, (pts[i,0]+6, pts[i,1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

# ── Camera ─────────────────────────────────────────────────────────────────────

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

cap, _ = open_camera()
if cap is None:
    raise SystemExit("No camera found.")

print("Camera ready. Q to quit.")

smoothers    = [SmoothedROI() for _ in range(MAX_HANDS)]
tracked_rois = []   # list of (affine, inv_affine, hand_idx)

# ── Main loop ──────────────────────────────────────────────────────────────────

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    frame = np.ascontiguousarray(cv2.flip(frame, 1))
    H, W  = frame.shape[:2]

    # always run palm detector when we don't have enough hands
    if len(tracked_rois) < MAX_HANDS:
        for i in range(len(tracked_rois), MAX_HANDS):
            smoothers[i].reset()

        tensor, pad = preprocess_palm(frame)
        regressors, scores_raw = run_palm(tensor)
        detections = decode_detections(regressors, scores_raw, pad)

        # rebuild tracked_rois from fresh detections
        tracked_rois = []
        for i, det in enumerate(detections):
            affine, inv_affine = compute_roi_from_detection(det, W, H)
            tracked_rois.append((affine, inv_affine, i))

    # landmark inference
    new_tracked_rois = []
    for affine, inv_affine, hand_idx in tracked_rois:
        tensor_lm, crop_rgb  = crop_hand(frame, inv_affine)
        raw                  = run_landmarks(tensor_lm)
        landmarks_px         = postprocess_landmarks(raw, affine)

        if landmarks_px is not None:
            draw_landmarks(frame, landmarks_px, hand_idx)
            new_affine, new_inv = smoothers[hand_idx].update(landmarks_px)
            new_tracked_rois.append((new_affine, new_inv, hand_idx))
        else:
            smoothers[hand_idx].reset()

    tracked_rois = new_tracked_rois

    # crop debug window for first hand only
    if tracked_rois:
        _, inv, _ = tracked_rois[0]
        _, crop_rgb = crop_hand(frame, inv)
        cv2.imshow("crop", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

    cv2.putText(frame, f"hands={len(tracked_rois)}  Q=quit",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()