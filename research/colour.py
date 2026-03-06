import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

points = []

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1) 
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_time = time.time()
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest) 
        if M["m00"] > 1000: 
            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"]) 
            points.append([cx, cy, current_time])

    points = [p for p in points if current_time - p[2] < 5]

    for i in range(1, len(points)):
        if points[i][2] - points[i-1][2] < 0.1: 
            cv2.line(frame, (points[i-1][0], points[i-1][1]), 
                     (points[i][0], points[i][1]), (0, 0, 255), 5)

    cv2.imshow("Iron Man Desk - Red Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Iron Man Desk - Red Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()