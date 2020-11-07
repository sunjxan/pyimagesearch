import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

cap = cv2.VideoCapture()

if cap.open("vtest.avi"):
    while True:
        ret, frame = cap.read()
        # 是否获得帧
        if not ret:
            break
        cv2.imshow("Frame", frame)
        # 等待时间为 1000 / fps
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
