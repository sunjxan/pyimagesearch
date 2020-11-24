import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

import face_recognition

image_model = cv2.imread('model.jpg')
rgb_model = cv2.cvtColor(image_model, cv2.COLOR_BGR2RGB)

boxes_model = face_recognition.face_locations(rgb_model, model='cnn')
encodings_model = face_recognition.face_encodings(rgb_model, boxes_model)

image = cv2.imread('faces.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model='cnn')
encodings = face_recognition.face_encodings(rgb, boxes)

for index, encoding in enumerate(encodings):
    matches = face_recognition.compare_faces(encodings_model, encoding, .6)
    if True in matches:
        top, right, bottom, left = boxes[index]
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

plt.subplot(1, 2, 1)
plt.imshow(image_model[..., (2, 1, 0)])
plt.title('model')
plt.subplot(1, 2, 2)
plt.imshow(image[..., (2, 1, 0)])
plt.title('faces')
plt.show()