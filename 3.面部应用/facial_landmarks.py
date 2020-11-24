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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread('faces.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(1, 2, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title('original')

rects = detector(gray, 1)
for index, rect in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    x, y, w, h = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    for ptX, ptY in shape:
        cv2.circle(image, (ptX, ptY), 1, (0, 0, 255), -1)
    if len(rects) > 1:
        cv2.putText(image, 'Face #{}'.format(index + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

plt.subplot(1, 2, 2)
plt.imshow(image[..., (2, 1, 0)])
plt.title('landmarks')
plt.show()