import cv2
import mediapipe as mp
import time
import math
import os
import numpy as np 

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'hand_landmarker.task')
PERSISTENCE_TIME = 15 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
points = [] 
angle = 0   
prev_palm_x = None
smooth_buffer = []

def apply_rotation(x, y, z, angle_deg):
    rad = math.radians(angle_deg)
    nx = x * math.cos(rad) + z * math.sin(rad)
    nz = -x * math.sin(rad) + z * math.cos(rad)
    return nx, y, nz

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2 
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        current_time = time.time()
        
        if result.hand_world_landmarks and result.hand_landmarks:
            world_lms = result.hand_world_landmarks[0]
            screen_lms = result.hand_landmarks[0]
            
            itip, ttip = screen_lms[8], screen_lms[4]
            pinch_dist = math.sqrt((itip.x - ttip.x)**2 + (itip.y - ttip.y)**2)
            
            is_open = screen_lms[12].y < screen_lms[10].y and screen_lms[20].y < screen_lms[18].y
            
            if is_open:
                curr_x = screen_lms[0].x
                if prev_palm_x is not None:
                    angle += (curr_x - prev_palm_x) * 180 
                prev_palm_x = curr_x
                cv2.putText(frame, "ROTATE MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                prev_palm_x = None
                if pinch_dist < 0.1:
                    idx_w = world_lms[8]
                    
                    smooth_buffer.append([idx_w.x, idx_w.y, idx_w.z])
                    if len(smooth_buffer) > 5: smooth_buffer.pop(0)
                    
                    avg_x = sum(p[0] for p in smooth_buffer) / len(smooth_buffer)
                    avg_y = sum(p[1] for p in smooth_buffer) / len(smooth_buffer)
                    avg_z = sum(p[2] for p in smooth_buffer) / len(smooth_buffer)
                    
                    points.append([avg_x, avg_y, avg_z, current_time])

        points = [p for p in points if current_time - p[3] < PERSISTENCE_TIME]

        for i in range(1, len(points)):
            if points[i][3] - points[i-1][3] < 0.2:
                x1, y1, z1 = apply_rotation(points[i-1][0], points[i-1][1], points[i-1][2], angle)
                x2, y2, z2 = apply_rotation(points[i][0], points[i][1], points[i][2], angle)
                
                u1, v1 = int(cx + x1 * 800), int(cy + y1 * 800)
                u2, v2 = int(cx + x2 * 800), int(cy + y2 * 800)
                
                color_val = int(np.clip(255 + (z2 * 1000), 100, 255))
                cv2.line(frame, (u1, v1), (u2, v2), (color_val, 0, 255), 3)

        cv2.imshow("Spatial AI Desk", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()