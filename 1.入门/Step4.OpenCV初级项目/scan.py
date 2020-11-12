import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

import imutils
from imutils.transform import four_point_transform
from skimage.filters import threshold_local

image = cv2.imread("receipt.jpg")
original = image.copy()
h, w = image.shape[:2]
ratio = 500 / h
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

print("STEP 1: Edge Detection")
plt.subplot(1, 4, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title("Original")
plt.subplot(1, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title("Gray")
plt.subplot(1, 4, 3)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred")
plt.subplot(1, 4, 4)
plt.imshow(edged, cmap='gray')
plt.title("Edged")
plt.show()

cnts, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
screenCnt = None

for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, .02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

print("SETP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
plt.subplot(1, 4, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title("Outline")

print("STEP 3: Apply perspective transform")
# 除以ration进行还原
warped = four_point_transform(original, screenCnt.squeeze() / ratio)
plt.subplot(1, 4, 2)
plt.imshow(warped[..., (2, 1, 0)])
plt.title("Warped")

# 使用scikit-image包里的threshold_local将灰度图片转换为阈值
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 4, 3)
plt.imshow(warped, cmap='gray')
plt.title("Gray")

T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
plt.subplot(1, 4, 4)
plt.imshow(warped, cmap='gray')
plt.title("Scanned")
plt.show()