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

    output = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('', output)

def endFunc():
    cv2.destroyAllWindows()

# playVideo("../1.入门/Step5.OpenCV高级项目/vtest.avi", captureFunc=captureFunc)
captureCamera(0, captureFunc=captureFunc, endFunc=endFunc)