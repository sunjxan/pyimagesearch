import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2

import imutils
from imutils.video_capture import playVideo
from imutils.object_detection import non_max_suppression

hog = cv2.HOGDescriptor()
detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(detector)

def detect(image):
    rects, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # 非极大值抑制
    rects = non_max_suppression(rects, overlapThresh=.65)

    output = image.copy()
    for x, y, w, h in rects:
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)

    return output

def captureFunc(frame, frameIndex):
    result = detect(frame)
    cv2.imshow("Frame", result)
    return result

playVideo("../1.入门/OpenCV高级项目/vtest.avi", captureFunc=captureFunc)