import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2

import imutils
from imutils.video_capture import playVideo
from imutils.video_capture import captureCamera

trackers = None

def captureFunc(frame, frameIndex):
    global trackers
    if trackers is not None and len(trackers):
        for tracker in trackers:
            ret, box = tracker.update(frame)
            if ret:
                x, y, w, h = [round(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    return frame

def ROIFunc(frame, ROIs):
    global trackers
    trackers = []
    for ROI in ROIs:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(ROI))
        trackers.append(tracker)

# playVideo("../1.入门/Step5.OpenCV高级项目/vtest.avi", captureFunc=captureFunc, ROIFunc=ROIFunc)
captureCamera(0, captureFunc=captureFunc, ROIFunc=ROIFunc)