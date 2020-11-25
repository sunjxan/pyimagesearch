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

for index, box in enumerate(boxes):
    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    for landmark in landmarks[index].values():
        for ptX, ptY in landmark:
            cv2.circle(image, (ptX, ptY), 1, (0, 0, 255), -1)
    if len(boxes) > 1:
        cv2.putText(image, 'Face #{}'.format(index + 1), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title('original')
plt.subplot(1, 2, 2)
plt.imshow(image[..., (2, 1, 0)])
plt.title('landmarks')
plt.show()