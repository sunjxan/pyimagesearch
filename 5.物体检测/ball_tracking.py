import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2

import imutils
from imutils.video_capture import playVideo
from imutils.video_capture import captureCamera

def captureFunc(frame, frameIndex):
    greenLower = (10, 80, 40)
    greenUpper = (40, 180, 140)

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)

    cnts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxContour = max(cnts, key=cv2.contourArea)
        M = cv2.moments(maxContour)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        x, y, r = cv2.minEnclosingCircle(maxContour)
        print(type(x), type(y), type(r))

        if r > 10:
            cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
            cv2.circel(frame, (cX, cY), 5, (0, 0, 255), -1)

    cv2.imshow('', frame)

def endFunc():
    cv2.destroyAllWindows()

# playVideo("../1.入门/Step5.OpenCV高级项目/vtest.avi", captureFunc=captureFunc)
captureCamera(0, captureFunc=captureFunc, endFunc=endFunc)