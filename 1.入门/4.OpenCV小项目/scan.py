import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append('../..')

import cv2
import numpy as np

import imutils
from imutils.transform import four_point_transform
from skimage.filters import threshold_local

image = cv2.imread("receipt.jpg")
original = image.copy()
h, w, d = image.shape
ratio = 500 / h
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey()
cv2.destroyAllWindows()

cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
cv2.imshow("Outline", image)
cv2.waitKey()
cv2.destroyAllWindows()

# 除以ration进行还原
warped = four_point_transform(original, screenCnt.squeeze() / ratio)

# 使用scikit-image包里的threshold_local将灰度图片转换为阈值
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(original, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey()
cv2.destroyAllWindows()