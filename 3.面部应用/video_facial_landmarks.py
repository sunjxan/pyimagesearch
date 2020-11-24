import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

import dlib
import face_recognition

from imutils import face_utils
from imutils.video_capture import captureCamera

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def captureFunc(frame, frameIndex):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    for index, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for ptX, ptY in shape:
            cv2.circle(frame, (ptX, ptY), 1, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    return frame

captureCamera(0, captureFunc=captureFunc)