import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

import imutils

image = cv2.imread("jp.png")
h, w, d = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

B, G, R = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

plt.subplot(1, 2, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title('Image')

roi = image[60:160, 320:420]
plt.subplot(1, 2, 2)
plt.imshow(roi[..., (2, 1, 0)])
plt.title('ROI')
plt.show()

resized = cv2.resize(image, (200, 200))
plt.subplot(1, 3, 1)
plt.imshow(resized[..., (2, 1, 0)])
plt.title('Fixed Resizing')

ratio = .5
resized = cv2.resize(image, (round(w * ratio), round(h * ratio)))
plt.subplot(1, 3, 2)
plt.imshow(resized[..., (2, 1, 0)])
plt.title('Aspect Ratio Resize')

resized = imutils.resize(image, width=300)
plt.subplot(1, 3, 3)
plt.imshow(resized[..., (2, 1, 0)])
plt.title('Imutils Resize')
plt.show()

center = round(w / 2), round(h / 2)
# 得到旋转矩阵
M = cv2.getRotationMatrix2D(center, -45, 1.0)
# 仿射运算
# (x, y) => (M[0, 0]x + M[0, 1]y + M[0, 2], M[1, 0]x + M[1, 1]y + M[1, 2])
# 可以设置边框
rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_WRAP)
plt.subplot(1, 3, 1)
plt.imshow(rotated[..., (2, 1, 0)])
plt.title('OpenCV Rotation')

rotated = imutils.rotate(image, -45)
plt.subplot(1, 3, 2)
plt.imshow(rotated[..., (2, 1, 0)])
plt.title('Imutils Rotation')

rotated = imutils.rotate_bound(image, -45)
plt.subplot(1, 3, 3)
plt.imshow(rotated[..., (2, 1, 0)])
plt.title('Imutils Bound Rotation')
plt.show()

blurred = cv2.GaussianBlur(image, (11, 11), 0)
plt.imshow(blurred[..., (2, 1, 0)])
plt.title('Blurred')
plt.show()

output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
plt.subplot(2, 2, 1)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Rectangle')

output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
plt.subplot(2, 2, 2)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Circle')

output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
plt.subplot(2, 2, 3)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Line')

output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
plt.subplot(2, 2, 4)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Text')
plt.show()