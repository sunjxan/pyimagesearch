import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
import math
from collections import deque

import imutils
from imutils.video_capture import playVideo
from imutils.video_capture import captureCamera

greenLower = (15, 100, 60)
greenUpper = (35, 160, 120)
ptSize = 128
pts = deque(maxlen=ptSize)

def captureFunc(frame, frameIndex):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)

    cnts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
        maxContour = max(cnts, key=cv2.contourArea)
        M = cv2.moments(maxContour)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        center = (cX, cY)
        (x, y), r = cv2.minEnclosingCircle(maxContour)
        x, y, r = round(x), round(y), round(r)

        if r > 10:
            cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    pts.appendleft(center)

    count = len(pts)
    for i in range(1, count):
        if i > 20:
            break
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = round(math.sqrt(ptSize / (i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow('', frame)

def endFunc():
    cv2.destroyAllWindows()

# playVideo("../1.入门/Step5.OpenCV高级项目/vtest.avi", captureFunc=captureFunc)
captureCamera(0, captureFunc=captureFunc, endFunc=endFunc)