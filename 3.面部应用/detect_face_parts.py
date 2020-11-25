import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

import face_recognition

image = cv2.imread('faces.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model='cnn')
landmarks = face_recognition.face_landmarks(rgb, boxes)

colors = np.random.randint(0, 256, (9, 3), dtype=np.uint8).tolist()

for index, box in enumerate(boxes):
    top, right, bottom, left = box
    for ix, landmark in enumerate(landmarks[index].keys()):
        # 画透明部分的备份
        backup = image.copy()
        pts = np.array(landmarks[index][landmark])
        if landmark in ("left_eye", "right_eye", "top_lip", "bottom_lip"):
            # 画眼睛和嘴唇的封闭区域
            hull = cv2.convexHull(pts)
            cv2.drawContours(image, [hull], -1, colors[ix], -1)
        else:
            # 画眉毛、鼻子和颚线的线条
            for i in range(1, len(pts)):
                ptA = tuple(pts[i - 1])
                ptB = tuple(pts[i])
                cv2.line(image, ptA, ptB, colors[ix], 2)
        image = cv2.addWeighted(image, .75, backup, .25, 0)
    if len(boxes) > 1:
        cv2.putText(image, 'Face #{}'.format(index + 1), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title('original')
plt.subplot(1, 2, 2)
plt.imshow(image[..., (2, 1, 0)])
plt.title('landmarks')
plt.show()