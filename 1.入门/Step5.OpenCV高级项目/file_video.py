import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

cap = cv2.VideoCapture()
cap.open("vtest.avi")

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        # 是否获得帧
        if not ret:
            break
        cv2.imshow("Frame", frame)
        # 等待时间为 1000 / fps
        if cv2.waitKey(100) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
