import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
import dlib

import imutils
from imutils.video_capture import playVideo
from imutils.video_capture import captureCamera

trackers = []

def captureFunc(frame, frameIndex):
    global trackers
    if len(trackers):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for index, tracker in enumerate(trackers):
            tracker.update(rgb)
            pos = tracker.get_position()
            if pos is not None:
                top = round(pos.top())
                bottom = round(pos.bottom())
                left = round(pos.left())
                right = round(pos.right())
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "ID {}".format(index + 1), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    return frame

def ROIFunc(frame, ROIs):
    global trackers
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for ROI in ROIs:
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(ROI[0], ROI[1], ROI[0] + ROI[2], ROI[1] + ROI[3])
        tracker.start_track(rgb, rect)
        trackers.append(tracker)

playVideo("../1.入门/OpenCV高级项目/vtest.avi", captureFunc=captureFunc, ROIFunc=ROIFunc)
# captureCamera(0, captureFunc=captureFunc, ROIFunc=ROIFunc)