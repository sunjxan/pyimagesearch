import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

detector = cv2.CascadeClassifier("/home/sunjxan/opencv/data/haarcascades/haarcascade_frontalcatface.xml")

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray)
    output = image.copy()
    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(output, (x, y), (x + w, y +  h), (0, 0, 255), 2)
        cv2.putText(output, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    return output

image = cv2.imread("cat_01.jpg")
image2 = cv2.imread("cat_02.jpg")
image3 = cv2.imread("cat_03.jpg")
image4 = cv2.imread("cat_04.jpg")

plt.subplot(2, 2, 1)
plt.imshow(detect(image)[..., (2, 1, 0)])
plt.subplot(2, 2, 2)
plt.imshow(detect(image2)[..., (2, 1, 0)])
plt.subplot(2, 2, 3)
plt.imshow(detect(image3)[..., (2, 1, 0)])
plt.subplot(2, 2, 4)
plt.imshow(detect(image4)[..., (2, 1, 0)])
plt.show()