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

colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread('faces.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

output = image.copy()
for index, rect in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    x, y, w, h = face_utils.rect_to_bb(rect)

    # 画透明部分的备份
    img = output.copy()
    for ix, name in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
        l, r = face_utils.FACIAL_LANDMARKS_IDXS[name]
        pts = shape[l:r]
        # 画颚线
        if name == "jaw":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(img, ptA, ptB, colors[ix], 2)
        # 画其他部分的区域
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(img, [hull], -1, colors[ix], -1)
        output = cv2.addWeighted(img, .5, output, .5, 0)
    if len(rects) > 1:
        cv2.putText(output, 'Face #{}'.format(index + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

plt.subplot(1, 2, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title('original')
plt.subplot(1, 2, 2)
plt.imshow(output[..., (2, 1, 0)])
plt.title('landmarks')
plt.show()