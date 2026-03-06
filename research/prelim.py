import cv2
import mediapipe as mp
import time
import math
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'hand_landmarker.task')



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

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        current_time = time.time()
        
        if result.hand_landmarks:

            itip = result.hand_landmarks[0][8]
            ttip = result.hand_landmarks[0][4]
            
            ix, iy = int(itip.x * w), int(itip.y * h)
            tx, ty = int(ttip.x * w), int(ttip.y * h)
            
            dist = math.sqrt((ix - tx)**2 + (iy - ty)**2)
            
            if dist < 50:
                points.append([ix, iy, current_time])

        points = [p for p in points if current_time - p[2] < 5]

        for i in range(1, len(points)):
            if points[i][2] - points[i-1][2] < 0.2:
                cv2.line(frame, (points[i-1][0], points[i-1][1]), 
                         (points[i][0], points[i][1]), (255, 0, 255), 5)

        cv2.imshow("Iron Man AI Desk", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or \
           cv2.getWindowProperty("Iron Man AI Desk", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()