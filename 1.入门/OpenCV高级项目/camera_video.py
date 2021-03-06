import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

cap = cv2.VideoCapture()
cap.open(0)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        # 是否获得帧
        if not ret:
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
