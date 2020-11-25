import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2

import face_recognition

from imutils import face_utils
from imutils.video_capture import captureCamera
from imutils.video_capture import playVideo


def captureFunc(frame, frameIndex):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='cnn')
    landmarks = face_recognition.face_landmarks(rgb, boxes)

    for index, box in enumerate(boxes):
        for landmark in landmarks[index].values():
            for ptX, ptY in landmark:
                cv2.circle(frame, (ptX, ptY), 1, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    return frame

playVideo('Megamind.avi', fps=24, captureFunc=captureFunc)
# captureCamera(0, captureFunc=captureFunc)